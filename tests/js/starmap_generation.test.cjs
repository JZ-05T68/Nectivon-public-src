// Canonical data and saved 3D positions enter the original renderer through its bridge.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const script = fs.readFileSync(path.join(__dirname,
  '../../src/components/knowledge_starmap/bridge.js'), 'utf8');

function bridge() {
  const listeners = new Map(), sent = [], calls = [];
  const nodes = [{id: 'knowledge:1', x: 0, y: 0, z: 0},
    {id: 'knowledge:2', x: 3, y: 4, z: 5}];
  const graph = {
    graphData: () => ({nodes}),
    enableNodeDrag: value => calls.push(['drag', value]),
    onNodeDragEnd: callback => {graph.dragEnd = callback;},
    d3ReheatSimulation: () => {},
  };
  const parent = {postMessage: message => sent.push(message)};
  const window = {parent, __starmap: {graph, setPayload: value => calls.push(['payload', value])},
    addEventListener: (name, fn) => listeners.set(name, fn),
    dispatchEvent: () => {}, setTimeout: () => {}};
  const elements = new Map(), styles = new Map();
  const document = {documentElement: {style: {setProperty: (key, value) => styles.set(key, value)}},
    getElementById: id => elements.get(id),
    createElement: () => ({remove() {elements.delete(this.id);}}),
    body: {append: element => elements.set(element.id, element)}};
  const context = vm.createContext({window, document, Event});
  vm.runInContext(script, context);
  const deliver = (payload, appearance = {}) => listeners.get('message')({source: parent,
    data: {type: 'streamlit:render', args: {payload, revision: String(calls.length), ...appearance}}});
  return {window, deliver, calls, nodes, graph, sent, elements, styles};
}

function payload() {
  return {scope: 'library', mode: 'nectivon', layout_editing: false,
    layout: {positions: {'knowledge:1': {x: 10, y: 20, z: 30}}},
    vault: {notes: [], links: []},
    knowledge_graph: {schema_version: 1,
      nodes: [{id: 'knowledge:1'}, {id: 'knowledge:2'}],
      edges: [{source: 'knowledge:2', target: 'knowledge:1', relation_type: 'requires',
        confirmation_status: 'persisted'}]}};
}

test('canonical stable IDs, typed direction and saved coordinates reach renderer', () => {
  const b = bridge();
  b.deliver(payload());
  assert.equal(b.nodes[0].fx, 10);
  assert.equal(b.nodes[0].fy, 20);
  assert.equal(b.nodes[0].fz, 30);
  b.deliver({...payload(), vaultName: 'updated'});
  const passed = b.calls.find(call => call[0] === 'payload')[1];
  assert.equal(passed.vault.links[0].relation_type, 'requires');
  assert.equal(passed.vault.links[0].source, 'knowledge:2');
  assert.equal(passed.vault.notes[0].id, 'knowledge:1');
  b.nodes[0].x = 99;
  b.graph.dragEnd(b.nodes[0]);
  assert.equal(b.nodes[0].fx, 99, 'Manual drag is pinned, rather than immediately undone by forces');
});

test('candidate edges or duplicate IDs are rejected before renderer update', () => {
  const b = bridge();
  const candidate = payload();
  candidate.knowledge_graph.edges[0].confirmation_status = 'ai_candidate';
  assert.throws(() => b.deliver(candidate), /未确认/);
  const duplicate = payload();
  duplicate.knowledge_graph.nodes.push({id: 'knowledge:1'});
  assert.throws(() => b.deliver(duplicate), /重复/);
  assert.equal(b.calls.filter(call => call[0] === 'payload').length, 0);
});

test('relations and node selections still go to the parent service bridge', () => {
  const b = bridge();
  const relation = {action: 'unlink', source: 'knowledge:2', target: 'knowledge:1'};
  b.window.nectivonRelationChange(relation);
  b.window.nectivonSelected({id: 'knowledge:1'});
  assert.equal(b.sent.at(-2).value.relation, relation);
  assert.equal(b.sent.at(-1).value.node_id, 'knowledge:1');
});

test('explicit isolated preview permits candidate lines with banner but blocks relation writes', () => {
  const b = bridge(), p = payload();
  p.is_preview = true; p.knowledge_graph.mode = 'preview';
  p.knowledge_graph.nodes = [{id: 'preview:revision:1', confirmation_status: 'ai_candidate'},
    {id: 'preview:revision:2', confirmation_status: 'ai_candidate'}];
  p.knowledge_graph.edges = [{source: 'preview:revision:2', target: 'preview:revision:1',
    relation_type: 'requires', confirmation_status: 'ai_candidate'}];
  b.deliver(p);
  assert.match(b.elements.get('nvPreviewBanner').textContent, /尚未写入正式知识库/);
  const before = b.sent.length;
  b.window.nectivonRelationChange({action: 'link', source: 'preview:revision:1',
    target: 'preview:revision:2'});
  assert.equal(b.sent.length, before);
  b.deliver(payload());
  assert.equal(b.elements.has('nvPreviewBanner'), false);
});

test('theme-only updates preserve unsaved positions and do not restart graph rendering', () => {
  const b = bridge(), p = payload();
  b.deliver(p, {revision: 'same-data', theme: {surface: '#ffffff', text: '#202c3b'}});
  b.nodes[0].x = 92;
  const beforeCalls = b.calls.length, beforeMessages = b.sent.length;
  b.deliver(p, {revision: 'same-data', theme: {surface: '#1b2330', text: '#e5ebf4'}});
  assert.equal(b.styles.get('--nv-surface'), '#1b2330');
  assert.equal(b.styles.get('--nv-text'), '#e5ebf4');
  assert.equal(b.nodes[0].x, 92, 'Theme switch cannot apply the older saved position');
  assert.equal(b.calls.length, beforeCalls, 'No setPayload or node pinning on theme change');
  assert.equal(b.sent.length, beforeMessages, 'Theme changes never emit persistence events');
  b.deliver(p, {revision: 'same-data', theme: {surface: 'url(https://invalid.test/x)', text: 'red'}});
  assert.equal(b.styles.get('--nv-surface'), '#1b2330', 'Only local palette colors are accepted');
});
