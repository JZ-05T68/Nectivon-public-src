/* Pure view geometry; no graph mutation or I/O. */
((root) => {
  "use strict";
  const WIDTH = 196, HEIGHT = 70, MIN_SCALE = .00001;
  function initialPositions(nodes, stored = {}) {
    const positions = {}, columns = {document: 0, page: 1, knowledge: 2, question: 3, family: 4};
    const rows = {}, occupied = Object.values(stored);
    [...nodes].sort((a, b) => a.id.localeCompare(b.id)).forEach(node => {
      if (stored[node.id]) { positions[node.id] = {...stored[node.id]}; return; }
      const column = columns[node.kind] ?? 2;
      let row = rows[column] || 0;
      let point = {x: 40 + column * 265, y: 40 + row * 115};
      while (occupied.some(p => Math.abs(p.x - point.x) < WIDTH + 24 && Math.abs(p.y - point.y) < HEIGHT + 24)) {
        row += 1;
        point = {x: 40 + column * 265, y: 40 + row * 115};
      }
      rows[column] = row + 1;
      positions[node.id] = point;
      occupied.push(point);
    });
    return positions;
  }
  function visibleNodes(nodes, children, collapsed) {
    const hidden = new Set();
    for (const parent of collapsed) for (const child of children[parent] || []) hidden.add(child);
    return nodes.filter(node => !hidden.has(node.id));
  }
  function reconcilePositions(nodes, stored, local) {
    return {...local, ...initialPositions(nodes, {...stored, ...local})};
  }
  function fitViewport(nodes, positions, width, height) {
    if (!nodes.length) return {x: 0, y: 0, scale: 1};
    const points = nodes.map(node => positions[node.id]);
    const left = Math.min(...points.map(p => p.x)), top = Math.min(...points.map(p => p.y));
    const right = Math.max(...points.map(p => p.x)) + WIDTH;
    const bottom = Math.max(...points.map(p => p.y)) + HEIGHT;
    const scale = Math.min(1.4, Math.max(MIN_SCALE, Math.min((width - 60) / (right - left), (height - 60) / (bottom - top))));
    return {scale, x: (width - (right - left) * scale) / 2 - left * scale,
      y: (height - (bottom - top) * scale) / 2 - top * scale};
  }
  function zoomAt(viewport, factor, x, y) {
    const scale = Math.max(MIN_SCALE, Math.min(4, viewport.scale * factor));
    const ratio = scale / viewport.scale;
    return {scale, x: x - (x - viewport.x) * ratio, y: y - (y - viewport.y) * ratio};
  }
  const api = {WIDTH, HEIGHT, MIN_SCALE, initialPositions, reconcilePositions, visibleNodes, fitViewport, zoomAt};
  if (typeof module !== "undefined") module.exports = api;
  else root.NectivonMindmapModel = api;
})(typeof window === "undefined" ? globalThis : window);
