/* Offline Streamlit component. Text uses textContent; canvas is view state only. */
(() => {
  "use strict";
  const model = window.NectivonMindmapModel;
  const svg = document.getElementById("canvas"), viewportElement = document.getElementById("viewport");
  const nodesElement = document.getElementById("nodes"), edgesElement = document.getElementById("edges");
  const status = document.getElementById("status"), counts = document.getElementById("counts");
  const kinds = {knowledge: "知识点", question: "历史题目", family: "历史归纳族", document: "来源文档", page: "来源页"};
  let payload = null, positions = {}, viewport = {x: 0, y: 0, scale: 1};
  let collapsed = new Set(), selected = null, gesture = null, sequence = 0, dirty = false;
  function applyTheme(theme) {
    if (!theme || !document.documentElement?.style) return;
    for (const key of ["background", "surface", "subtle", "raised", "border", "edge", "text", "muted", "accent", "soft"]) {
      if (/^#[0-9a-f]{6}$/i.test(theme[key] || "")) {
        document.documentElement.style.setProperty(`--nv-${key}`, theme[key]);
      }
    }
    document.documentElement.style.colorScheme = theme.scheme === "dark" ? "dark" : "light";
  }
  const send = (type, extra = {}) => window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, "*");
  const emit = value => send("streamlit:setComponentValue", {value: {...value, scope: payload.scope,
    event_id: `${Date.now()}:${++sequence}`}, dataType: "json"});
  function element(name, attrs = {}, text = null) {
    const result = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => result.setAttribute(key, value));
    if (text !== null) result.textContent = text;
    return result;
  }
  function changed() { dirty = true; status.textContent = "布局已修改，点击“保存布局”持久保存"; }
  function camera() {
    viewportElement.setAttribute("transform", `translate(${viewport.x} ${viewport.y}) scale(${viewport.scale})`);
    svg.dataset.scale = viewport.scale;
    svg.dataset.panX = viewport.x;
    svg.dataset.panY = viewport.y;
  }
  function visible() { return model.visibleNodes(payload.nodes, payload.children, collapsed); }
  function render() {
    if (!payload) return;
    const shown = visible(), identifiers = new Set(shown.map(node => node.id));
    const links = payload.links.filter(edge => identifiers.has(edge.source) && identifiers.has(edge.target));
    edgesElement.replaceChildren();
    for (const edge of links) {
      const a = positions[edge.source], b = positions[edge.target];
      const ax = a.x + model.WIDTH / 2, ay = a.y + model.HEIGHT / 2;
      const bx = b.x + model.WIDTH / 2, by = b.y + model.HEIGHT / 2;
      const dx = bx - ax, dy = by - ay;
      const trim = Math.max(Math.abs(dx) / (model.WIDTH / 2 + 5), Math.abs(dy) / (model.HEIGHT / 2 + 5), 1);
      const line = element("line", {x1: ax, y1: ay, x2: bx - dx / trim, y2: by - dy / trim,
        class: "edge", "marker-end": "url(#arrow)", "data-source": edge.source, "data-target": edge.target});
      line.append(element("title", {}, `${edge.label}${edge.note ? " · " + edge.note : ""}`));
      edgesElement.append(line);
      edgesElement.append(element("text", {x: (ax + bx) / 2, y: (ay + by) / 2 - 6,
        "text-anchor": "middle", class: "edge-label"}, edge.label));
    }
    nodesElement.replaceChildren();
    for (const node of shown) {
      const p = positions[node.id];
      const group = element("g", {class: `node ${node.kind}${node.id === selected ? " selected" : ""}`,
        transform: `translate(${p.x} ${p.y})`, tabindex: 0, role: "button", "aria-label": `${node.title} ${node.status}`,
        "data-node-id": node.id, "data-x": p.x, "data-y": p.y});
      group.append(element("rect", {width: model.WIDTH, height: model.HEIGHT, rx: 9}));
      group.append(element("title", {}, `${node.title}\n${node.id}${node.status ? " · " + node.status : ""}`));
      const title = [...node.title];
      group.append(element("text", {x: 12, y: 23}, title.slice(0, 15).join("")));
      group.append(element("text", {x: 12, y: 41}, title.slice(15, 30).join("") + (title.length > 30 ? "…" : "")));
      group.append(element("text", {x: 12, y: 59, class: "meta"}, `${kinds[node.kind] || node.kind}${node.status ? " · " + node.status : ""}`));
      if (payload.children[node.id]?.length) {
        const fold = element("g", {class: "fold", transform: "translate(170 45)", role: "button",
          tabindex: 0, "aria-label": `${collapsed.has(node.id) ? "展开" : "折叠"}来源页`, "data-fold-id": node.id});
        fold.append(element("rect", {width: 21, height: 21, rx: 4}));
        fold.append(element("text", {x: 6, y: 16}, collapsed.has(node.id) ? "+" : "−"));
        group.append(fold);
      }
      nodesElement.append(group);
    }
    camera();
    counts.textContent = `画布显示 ${shown.length} / ${payload.nodes.length} 个节点（含来源），${links.length} / ${payload.links.length} 条关联；已折叠 ${payload.nodes.length - shown.length} 个来源页。知识与历史节点 ${payload.shown} / ${payload.total} 个。`;
  }
  function select(id) { selected = id; render(); emit({action: "select", node_id: id}); }
  function toggle(id) {
    if (collapsed.has(id)) collapsed.delete(id); else collapsed.add(id);
    changed(); render();
  }
  svg.addEventListener("pointerdown", event => {
    if (!payload || event.button !== 0) return;
    const fold = event.target.closest("[data-fold-id]");
    if (fold) { toggle(fold.dataset.foldId); event.preventDefault(); return; }
    const node = event.target.closest("[data-node-id]");
    gesture = {id: node?.dataset.nodeId, startX: event.clientX, startY: event.clientY,
      point: node ? {...positions[node.dataset.nodeId]} : {...viewport}, moved: false};
    svg.setPointerCapture(event.pointerId);
    svg.classList.toggle("panning", !node);
    event.preventDefault();
  });
  svg.addEventListener("pointermove", event => {
    if (!gesture) return;
    const dx = event.clientX - gesture.startX, dy = event.clientY - gesture.startY;
    if (!gesture.moved && Math.hypot(dx, dy) < 4) return;
    gesture.moved = true;
    if (gesture.id) positions[gesture.id] = {x: gesture.point.x + dx / viewport.scale, y: gesture.point.y + dy / viewport.scale};
    else viewport = {...viewport, x: gesture.point.x + dx, y: gesture.point.y + dy};
    changed(); render();
  });
  function endGesture(event) {
    if (!gesture) return;
    const ended = gesture; gesture = null; svg.classList.remove("panning");
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    if (event.type === "pointerup" && ended.id && !ended.moved) select(ended.id);
  }
  svg.addEventListener("pointerup", endGesture);
  svg.addEventListener("pointercancel", endGesture);
  svg.addEventListener("keydown", event => {
    if (!["Enter", " "].includes(event.key)) return;
    const fold = event.target.closest("[data-fold-id]");
    const node = event.target.closest("[data-node-id]");
    if (fold) toggle(fold.dataset.foldId); else if (node) select(node.dataset.nodeId);
    event.preventDefault();
  });
  svg.addEventListener("wheel", event => {
    if (!payload) return;
    event.preventDefault();
    const bounds = svg.getBoundingClientRect();
    viewport = model.zoomAt(viewport, Math.exp(-event.deltaY * .0015), event.clientX - bounds.left, event.clientY - bounds.top);
    changed(); camera();
  }, {passive: false});
  function fit() { viewport = model.fitViewport(visible(), positions, svg.clientWidth, svg.clientHeight); camera(); }
  document.getElementById("zoom-in").onclick = () => {
    viewport = model.zoomAt(viewport, 1.25, svg.clientWidth / 2, svg.clientHeight / 2); changed(); camera();
  };
  document.getElementById("zoom-out").onclick = () => {
    viewport = model.zoomAt(viewport, .8, svg.clientWidth / 2, svg.clientHeight / 2); changed(); camera();
  };
  document.getElementById("fit").onclick = () => { if (payload) { fit(); changed(); } };
  document.getElementById("reset").onclick = () => {
    if (!payload) return;
    positions = model.initialPositions(payload.nodes); collapsed.clear(); render(); fit(); changed();
  };
  document.getElementById("expand").onclick = () => { collapsed.clear(); changed(); render(); };
  document.getElementById("save").onclick = () => {
    if (!payload) return;
    status.textContent = "正在保存布局…";
    const currentPositions = Object.fromEntries(payload.nodes.map(node => [node.id, positions[node.id]]));
    emit({action: "save", layout: {positions: currentPositions, viewport,
      collapsed: [...collapsed].filter(id => payload.children[id])}});
  };
  window.addEventListener("message", event => {
    if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
    applyTheme(event.data.args?.theme);
    const incoming = event.data.args?.payload;
    if (!incoming || !Array.isArray(incoming.nodes)) return;
    const fresh = !payload || payload.scope !== incoming.scope;
    const saved = !fresh && incoming.saved_at && incoming.saved_at !== payload.saved_at;
    if (fresh) {
      positions = model.initialPositions(incoming.nodes, incoming.layout.positions);
      viewport = {...incoming.layout.viewport}; collapsed = new Set(incoming.layout.collapsed);
      selected = null; dirty = false;
    } else {
      positions = model.reconcilePositions(incoming.nodes, incoming.layout.positions, positions);
      if (saved) dirty = false;
    }
    payload = incoming; render();
    if (fresh && !incoming.has_saved_layout) fit();
    status.textContent = dirty ? "布局已修改，点击“保存布局”持久保存" : incoming.has_saved_layout ? "布局已保存到本地" : "默认布局；调整后可保存";
  });
  window.addEventListener("error", () => { status.textContent = "导图显示失败，请使用下方节点列表查看内容与来源。"; });
  send("streamlit:componentReady", {apiVersion: 1});
  send("streamlit:setFrameHeight", {height: 635});
})();
