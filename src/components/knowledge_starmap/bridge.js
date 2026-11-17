/* Streamlit protocol bridge. Knowledge data is supplied by the local SQLite service. */
(() => {
  "use strict";
  const send = (type, extra = {}) => window.parent.postMessage(
    {isStreamlitMessage: true, type, ...extra}, "*"
  );
  let resolveInitial;
  let initialized = false;
  let revision = "";
  let sequence = 0;
  let payload = null;
  let editing = false;
  function applyTheme(theme) {
    if (!theme || !document.documentElement?.style) return;
    for (const key of ['background', 'surface', 'border', 'text', 'muted', 'accent', 'soft']) {
      if (/^#[0-9a-f]{6}$/i.test(theme[key] || '')) {
        document.documentElement.style.setProperty(`--nv-${key}`, theme[key]);
      }
    }
  }
  function canonicalPayload(value) {
    const data = value?.knowledge_graph;
    if (!data) return value; // Historical envelope remains readable.
    if (data.schema_version !== 1 || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
      throw new Error('星图数据契约无效');
    }
    const ids = new Set(data.nodes.map(node => node.id));
    const preview = value.is_preview === true && data.mode === 'preview';
    if (data.mode === 'preview' && !preview) throw new Error('候选预览缺少明确模式');
    if (!preview && data.nodes.some(node => node.id.startsWith('preview:') ||
      node.confirmation_status === 'ai_candidate')) throw new Error('正式图不能包含候选预览');
    if (preview && data.nodes.some(node => !node.id.startsWith('preview:') ||
      node.confirmation_status !== 'ai_candidate')) throw new Error('候选预览身份或状态无效');
    if (ids.size !== data.nodes.length || data.edges.some(edge =>
      !ids.has(edge.source) || !ids.has(edge.target) ||
      edge.confirmation_status !== (preview ? 'ai_candidate' : 'persisted'))) {
      throw new Error('星图包含重复节点、悬空关系或未确认关系');
    }
    return {...value, vault: {...value.vault, notes: data.nodes, links: data.edges}};
  }
  function previewBanner() {
    let banner = document.getElementById('nvPreviewBanner');
    if (!payload?.is_preview) { if (banner) banner.remove(); return; }
    if (!banner) {
      banner = document.createElement('div'); banner.id = 'nvPreviewBanner';
      banner.className = 'nv-preview-banner'; document.body.append(banner);
    }
    banner.textContent = 'AI 候选预览 · 尚未写入正式知识库 · 请核对后在页面确认';
  }
  function applyPositions() {
    const graph = window.__starmap?.graph;
    if (!graph || !payload) return false;
    const nodes = graph.graphData()?.nodes || [];
    if (!nodes.length && payload.vault.notes.length) return false;
    const positions = payload.layout?.positions || {};
    for (const node of nodes) {
      const point = positions[node.id];
      if (node.hub || !point || !['x', 'y', 'z'].every(axis => Number.isFinite(point[axis]))) continue;
      for (const axis of ['x', 'y', 'z']) node[axis] = node[`f${axis}`] = point[axis];
    }
    graph.enableNodeDrag(editing);
    graph.onNodeDragEnd(node => {
      if (node.hub) return;
      for (const axis of ['x', 'y', 'z']) node[`f${axis}`] = node[axis];
    });
    if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
    return true;
  }
  function layoutControls() {
    if (document.getElementById('nvLayoutControls') || !payload?.layout_editing) return;
    const panel = document.createElement('div');
    panel.id = 'nvLayoutControls'; panel.className = 'nv-layout-controls';
    const edit = document.createElement('button');
    edit.className = 'nv-btn'; edit.type = 'button'; edit.textContent = '调整位置';
    edit.addEventListener('click', () => {
      editing = !editing;
      window.__starmap.graph.enableNodeDrag(editing);
      edit.textContent = editing ? '结束调整' : '调整位置';
      edit.classList.toggle('on', editing);
      edit.setAttribute('aria-pressed', String(editing));
    });
    const save = document.createElement('button');
    save.className = 'nv-btn'; save.type = 'button'; save.textContent = '保存星图布局';
    save.addEventListener('click', () => {
      const positions = {};
      for (const node of window.__starmap.graph.graphData().nodes) {
        if (!node.hub && ['x', 'y', 'z'].every(axis => Number.isFinite(node[axis]))) {
          positions[node.id] = {x: node.x, y: node.y, z: node.z};
        }
      }
      send('streamlit:setComponentValue', {value: {action: 'save_star_layout',
        scope: payload.scope, layout: {positions}, event_id: `${Date.now()}:${++sequence}`},
        dataType: 'json'});
    });
    panel.append(edit, save); document.body.append(panel);
  }
  function initializePositions(attempt = 0) {
    if (applyPositions()) { layoutControls(); return; }
    if (attempt < 300) window.setTimeout(() => initializePositions(attempt + 1), 100);
  }
  window.nectivonMapHeight = () => Math.max(240, window.innerHeight - (
    parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nv-footer-height')) || 0
  ));
  window.nectivonInitialPayload = new Promise(resolve => { resolveInitial = resolve; });
  window.nectivonSelected = node => {
    const id = node && !node.hub ? node.id : null;
    if (!id) return;
    send("streamlit:setComponentValue", {
      value: {node_id: id, event_id: `${Date.now()}:${++sequence}`}, dataType: "json"
    });
  };
  // 星图关系动作（连接/断开）交回父页面持久化；本组件从不直接写库。
  window.nectivonRelationChange = request => {
    if (!request || !request.action || payload?.is_preview) return;
    send("streamlit:setComponentValue", {
      value: {relation: request, event_id: `${Date.now()}:${++sequence}`}, dataType: "json"
    });
  };
  window.addEventListener("message", event => {
    if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
    const args = event.data.args;
    // Skin colours are independent of data revision, positions and simulation.
    applyTheme(args?.theme);
    if (!args?.payload?.vault || args.revision === revision) return;
    payload = canonicalPayload(args.payload);
    previewBanner();
    revision = args.revision;
    if (!initialized) {
      initialized = true;
      resolveInitial(payload);
      initializePositions();
    } else if (window.__starmap) {
      window.dispatchEvent(new Event('nectivon:payload-changed'));
      window.__starmap.setPayload(payload);
      applyPositions();
      layoutControls();
    }
  });
  window.addEventListener("error", () => {
    let panel = document.getElementById("nectivon-error");
    if (!panel) {
      panel = document.createElement("p");
      panel.id = "nectivon-error";
      panel.textContent = "星图渲染遇到问题，请使用页面下方的节点列表查看关联与来源。";
      document.body.append(panel);
    }
  });
  send("streamlit:componentReady", {apiVersion: 1});
  send("streamlit:setFrameHeight", {height: 620});
})();
