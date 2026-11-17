/* NON_REAL_TEST_DATA: geometry does not infer or edit knowledge relations. */
const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('../../src/components/knowledge_mindmap/model.js');
test('existing positions survive new nodes; deterministic fresh nodes', () => {
  const nodes = [{id: 'knowledge:1', kind: 'knowledge'}, {id: 'knowledge:2', kind: 'knowledge'}];
  const positions = model.initialPositions(nodes, {'knowledge:1': {x: 47, y: -19}});
  assert.deepEqual(positions['knowledge:1'], {x: 47, y: -19});
  assert.notDeepEqual(positions['knowledge:2'], positions['knowledge:1']);
  assert.deepEqual(model.initialPositions(nodes), model.initialPositions([...nodes].reverse()));
});
test('folding hides only explicit member pages, leaving cyclic knowledge alone', () => {
  const nodes = ['document:1', 'page:1', 'knowledge:1', 'knowledge:2'].map(id => ({id}));
  assert.deepEqual(model.visibleNodes(nodes, {'document:1': ['page:1']}, new Set(['document:1']))
    .map(node => node.id), ['document:1', 'knowledge:1', 'knowledge:2']);
  assert.equal(nodes.length, 4);
});
test('filter round trip retains saved and unsaved coordinates without inventing nodes', () => {
  const nodes = [{id: 'knowledge:1', kind: 'knowledge'}, {id: 'knowledge:2', kind: 'knowledge'}];
  const saved = {'knowledge:1': {x: 99, y: 88}, 'knowledge:2': {x: -55, y: 12}};
  const local = model.reconcilePositions(nodes.slice(0, 1), saved, {'knowledge:1': {x: 100, y: 101}});
  const returned = model.reconcilePositions(nodes, saved, local);
  assert.deepEqual(returned['knowledge:1'], {x: 100, y: 101});
  assert.deepEqual(returned['knowledge:2'], {x: -55, y: 12});
});
test('a new lexically earlier node cannot cover any saved node', () => {
  const initial = Array.from({length: 9}, (_, i) => ({id: `knowledge:${i + 1}`, kind: 'knowledge'}));
  const saved = model.initialPositions(initial);
  const next = model.initialPositions([...initial, {id: 'knowledge:10', kind: 'knowledge'}], saved);
  for (const node of initial) {
    assert.deepEqual(next[node.id], saved[node.id]);
    assert.notDeepEqual(next[node.id], next['knowledge:10']);
  }
  const dragged = {...saved, 'knowledge:2': {x: 570, y: 1075}};
  const after = model.initialPositions([...initial, {id: 'knowledge:10', kind: 'knowledge'}], dragged);
  for (const point of Object.values(dragged)) {
    assert.ok(Math.abs(point.x - after['knowledge:10'].x) >= model.WIDTH + 24 ||
      Math.abs(point.y - after['knowledge:10'].y) >= model.HEIGHT + 24);
  }
});
test('fit and zoom are bounded, finite and cursor anchored', () => {
  const fit = model.fitViewport([{id: 'knowledge:1'}], {'knowledge:1': {x: -10000, y: 10000}}, 1000, 500);
  assert.ok(Number.isFinite(fit.x) && Number.isFinite(fit.y));
  assert.ok(fit.scale >= model.MIN_SCALE && fit.scale <= 4);
  const zoom = model.zoomAt({x: 0, y: 0, scale: 1}, 2, 50, 70);
  assert.deepEqual(zoom, {x: -50, y: -70, scale: 2});
  assert.equal(model.zoomAt(zoom, 999, 0, 0).scale, 4);
});
test('fit really contains 300/1000 nodes and widely dragged saved nodes', () => {
  for (const count of [300, 1000]) {
    const nodes = Array.from({length: count}, (_, i) => ({id: `knowledge:${i}`, kind: 'knowledge'}));
    const positions = model.initialPositions(nodes);
    positions['knowledge:0'] = {x: -999999, y: -999999};
    positions['knowledge:1'] = {x: 999999, y: 999999};
    const fit = model.fitViewport(nodes, positions, 1000, 500);
    for (const point of Object.values(positions)) {
      assert.ok(point.x * fit.scale + fit.x >= 0);
      assert.ok(point.y * fit.scale + fit.y >= 0);
      assert.ok((point.x + model.WIDTH) * fit.scale + fit.x <= 1000);
      assert.ok((point.y + model.HEIGHT) * fit.scale + fit.y <= 500);
    }
  }
});

test('live canvas keeps unsaved zoom, node coordinates and dirty status across skin changes', () => {
  const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
  const elements = new Map(), styles = new Map(), handlers = new Map(), sent = [];
  function element() {
    return {attrs: {}, dataset: {}, children: [], listeners: new Map(),
      clientWidth: 900, clientHeight: 520,
      setAttribute(key, value) {this.attrs[key] = value;},
      append(child) {this.children.push(child);},
      replaceChildren() {this.children = [];},
      addEventListener(name, fn) {this.listeners.set(name, fn);},
      getBoundingClientRect: () => ({left: 0, top: 0})};
  }
  const document = {documentElement: {style: {setProperty: (key, value) => styles.set(key, value)}},
    getElementById: id => {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    }, createElementNS: () => element()};
  const parent = {postMessage: message => sent.push(message)};
  const window = {parent, NectivonMindmapModel: model,
    addEventListener: (name, fn) => handlers.set(name, fn)};
  vm.runInContext(fs.readFileSync(path.join(__dirname,
    '../../src/components/knowledge_mindmap/mindmap.js'), 'utf8'), vm.createContext({window, document}));
  const payload = {scope: 'NON_REAL_TEST_DATA', has_saved_layout: true, shown: 1, total: 1,
    nodes: [{id: 'knowledge:1', kind: 'knowledge', title: '测试知识', status: '人工确认'}],
    links: [], children: {}, layout: {positions: {'knowledge:1': {x: 47, y: -19}},
      viewport: {x: 21, y: 34, scale: 1}, collapsed: []}};
  const deliver = theme => handlers.get('message')({source: parent,
    data: {type: 'streamlit:render', args: {payload, theme}}});
  deliver({surface: '#ffffff', scheme: 'light'});
  const svg = elements.get('canvas');
  svg.listeners.get('wheel')({preventDefault() {}, deltaY: -200, clientX: 100, clientY: 100});
  const zoomed = {...svg.dataset}, beforeMessages = sent.length;
  deliver({surface: '#1b2330', text: '#e5ebf4', scheme: 'dark'});
  assert.deepEqual(svg.dataset, zoomed, 'Skin changes must not restore the old saved viewport');
  assert.equal(elements.get('nodes').children[0].attrs.transform, 'translate(47 -19)');
  assert.match(elements.get('status').textContent, /布局已修改/);
  assert.equal(styles.get('--nv-surface'), '#1b2330');
  assert.equal(sent.length, beforeMessages, 'Skin changes cannot save or select nodes');
});
