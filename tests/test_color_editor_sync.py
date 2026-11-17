"""Deterministic stale-render and atomic-save races, with local fake DOM and SQLite data."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

RACE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),listeners={},out=[],elements={};
const body={value:'',style:{},setAttribute(){},addEventListener(t,f){listeners[t]=f}};
Object.assign(elements,{body,status:{},label:{}});
const parent={postMessage(value){out.push(value)}};
let render;
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{
 window:{parent,addEventListener(t,f){render=f}},
 document:{body:{scrollHeight:220},getElementById(id){return elements[id]},
 querySelectorAll(){return[]}},ResizeObserver:class{observe(){}},
 setTimeout(){return 1},clearTimeout(){}
});
function send(value,ack=0,sequence=1,flush=null,external=0){
 render({source:parent,data:{type:'streamlit:render',args:{label:'Body',value,
 height:260,max_chars:30000,ack_serial:ack,render_serial:sequence,
 flush_token:flush,external_revision:external}}});
}
send('SAVED_BEFORE_TYPING');
if(listeners.focus)listeners.focus();
body.value='NEW_UNACKNOWLEDGED_INPUT';listeners.input();listeners.blur();
const transmitted=out.filter(x=>x.type==='streamlit:setComponentValue').at(-1).value.text;
send('SAVED_BEFORE_TYPING',0,2);
const afterStale=body.value;
send('SAVED_BEFORE_TYPING',0,3,77);
const flush=out.filter(x=>x.type==='streamlit:setComponentValue'&&x.value.flush_token===77).at(-1);
send('EXPLICIT_SOURCE_EDIT',0,4,77,1);
const sourceFlush=out.filter(x=>x.type==='streamlit:setComponentValue'
 &&x.value.flush_token===77).at(-1);
const serial=sourceFlush?sourceFlush.value.serial:0;
send('EXPLICIT_SOURCE_EDIT',serial,5,null,1);
send('SAVED_BEFORE_TYPING',0,2);
const afterAckAndOlder=body.value;
process.stdout.write(JSON.stringify({transmitted,after_stale_render:afterStale,
 lost_pending_input:afterStale!=='NEW_UNACKNOWLEDGED_INPUT',
 flush:flush?flush.value:null,after_ack_and_older:afterAckAndOlder,
 after_explicit_source:body.value,source_flush:sourceFlush?sourceFlush.value:null}));
"""


def test_stale_rerender_cannot_replace_pending_input_and_flush_carries_current_body():
    node = shutil.which('node')
    if not node:
        pytest.skip('本地Node未安装，浏览器流程另行验收')
    script = Path(__file__).parents[1] / 'src/study_loop/color_editor_component/editor.js'
    result = subprocess.run([node, '-e', RACE_HARNESS, str(script)], capture_output=True,
                            text=True, encoding='utf-8', timeout=15, check=True)
    observed = json.loads(result.stdout)
    assert observed['transmitted'] == 'NEW_UNACKNOWLEDGED_INPUT'
    assert observed['lost_pending_input'] is False
    assert observed['after_ack_and_older'] == 'EXPLICIT_SOURCE_EDIT'
    assert observed['flush']['text'] == 'NEW_UNACKNOWLEDGED_INPUT'
    assert observed['flush']['flush_token'] == 77
    assert observed['after_explicit_source'] == 'EXPLICIT_SOURCE_EDIT'
    assert observed['source_flush']['text'] == 'EXPLICIT_SOURCE_EDIT'
    assert observed['source_flush']['external_revision'] == 1


def test_identical_ack_does_not_reassign_textarea_or_reset_real_selection():
    node = shutil.which('node')
    if not node:
        pytest.skip('本地Node未安装，浏览器流程另行验收')
    script = Path(__file__).parents[1] / 'src/study_loop/color_editor_component/editor.js'
    harness = r"""
const vm=require('node:vm'),fs=require('node:fs'),listeners={},out=[];
let text='', assignments=0, render;
const body={style:{},selectionStart:0,selectionEnd:0,setAttribute(){},
 get value(){return text},set value(v){text=v;assignments++;
 this.selectionStart=this.selectionEnd=v.length},addEventListener(t,f){listeners[t]=f}};
const parent={postMessage(v){out.push(v)}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{
 window:{parent,addEventListener(t,f){render=f}},
 document:{body:{scrollHeight:200},getElementById(id){return id==='body'?body:{}},
 querySelectorAll(){return[]}},ResizeObserver:class{observe(){}}
});
function send(value,ack,sequence,external=0){render({source:parent,
 data:{type:'streamlit:render',args:{label:'Body',value,height:200,max_chars:30000,
 ack_serial:ack,render_serial:sequence,external_revision:external}}});}
send('ORIGINAL',0,1);
body.value='LATEST_TEXT';listeners.input();body.selectionStart=1;body.selectionEnd=5;
const before=assignments;
const serial=out.filter(x=>x.type==='streamlit:setComponentValue').at(-1).value.serial;
send('LATEST_TEXT',serial,2);
const after={assignments,selection:[body.selectionStart,body.selectionEnd]};
send('LATEST_TEXT',serial,3,1);
const sameSource={assignments,selection:[body.selectionStart,body.selectionEnd]};
send('EXPLICIT_DIFFERENT',serial,4,2);
process.stdout.write(JSON.stringify({before,after,sameSource,final:body.value}));
"""
    result = subprocess.run([node, '-e', harness, str(script)], capture_output=True,
                            text=True, encoding='utf-8', timeout=15, check=True)
    value = json.loads(result.stdout)
    assert value['after']['assignments'] == value['before']
    assert value['after']['selection'] == [1, 5]
    assert value['sameSource'] == value['after']
    assert value['final'] == 'EXPLICIT_DIFFERENT'


def test_theme_repaint_preserves_pending_input_selection_and_revision():
    """An iframe palette update cannot acknowledge or rewrite an unsent human draft."""
    node = shutil.which('node')
    if not node:
        pytest.skip('本地Node未安装，浏览器流程另行验收')
    script = Path(__file__).parents[1] / 'src/study_loop/color_editor_component/editor.js'
    harness = r"""
const vm=require('node:vm'),fs=require('node:fs'),listeners={},out=[],styles={};
let render,text='',assignments=0;
const body={style:{},selectionStart:0,selectionEnd:0,setAttribute(){},
 get value(){return text},set value(v){text=v;assignments++},
 addEventListener(t,f){listeners[t]=f}};
const parent={postMessage(v){out.push(v)}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{
 window:{parent,addEventListener(t,f){render=f}},
 document:{body:{scrollHeight:200},documentElement:{style:{setProperty(k,v){styles[k]=v}}},
 getElementById(id){return id==='body'?body:{}},querySelectorAll(){return[]}},
 ResizeObserver:class{observe(){}}
});
function send(palette,sequence){render({source:parent,data:{type:'streamlit:render',args:{
 label:'Body',value:'SAVED',height:200,max_chars:30000,ack_serial:0,
 render_serial:sequence,external_revision:0,palette}}});}
send({scheme:'light',surface:'#ffffff',text:'#202c3b'},1);
body.value='UNACKNOWLEDGED_HUMAN_DRAFT';body.selectionStart=2;body.selectionEnd=9;
listeners.input();const before=assignments,count=out.length;
send({scheme:'dark',surface:'#1b2330',text:'#e5ebf4',note_red:'#ff9292',note_blue:'#8bbcff'},2);
process.stdout.write(JSON.stringify({text:body.value,before,assignments,
 selection:[body.selectionStart,body.selectionEnd],newMessages:out.length-count,styles}));
"""
    result = subprocess.run([node, '-e', harness, str(script)], capture_output=True,
                            text=True, encoding='utf-8', timeout=15, check=True)
    value = json.loads(result.stdout)
    assert value['text'] == 'UNACKNOWLEDGED_HUMAN_DRAFT'
    assert value['assignments'] == value['before']
    assert value['selection'] == [2, 9]
    assert value['styles']['--note-surface'] == '#1b2330'
    assert value['styles']['--note-text'] == '#e5ebf4'
    assert value['styles']['--note-note_red'] == '#ff9292'
    assert value['styles']['--note-note_blue'] == '#8bbcff'
    assert value['newMessages'] == 1  # Only frame height; no extra text/ack packet.


COLOR_BUTTON_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),out=[],listeners={},buttons=[];
let render;
const body={value:'',style:{},selectionStart:0,selectionEnd:0,setAttribute(){},
 addEventListener(t,f){listeners[t]=f},focus(){},setSelectionRange(a,b){
 this.selectionStart=a;this.selectionEnd=b},
 setRangeText(value,a,b,mode){this.value=value;this.selectionStart=0;
 this.selectionEnd=value.length}};
for(const color of ['D62828','2563EB']){
 const handlers={};buttons.push({dataset:{color},handlers,
 addEventListener(t,f){handlers[t]=f}});
}
const status={},parent={postMessage(v){out.push(v)}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{
 window:{parent,addEventListener(t,f){render=f}},
 document:{body:{scrollHeight:250},querySelectorAll(){return buttons},
 getElementById(id){return id==='body'?body:id==='status'?status:{}}},
 NectivonColors:require(process.argv[2]),ResizeObserver:class{observe(){}}
});
const original='UNSELECTED_START\n重点 $x=y$\nUNSELECTED_END';
render({source:parent,data:{type:'streamlit:render',args:{label:'Body',value:original,
 height:250,max_chars:30000,ack_serial:0,render_serial:1,external_revision:0}}});
body.selectionStart=original.indexOf('重点');body.selectionEnd=original.indexOf('\nUNSELECTED_END');
const start=body.selectionStart;buttons[0].handlers.click();
const first=body.value,selection=[body.selectionStart,body.selectionEnd];
buttons[1].handlers.click();
process.stdout.write(JSON.stringify({original,start,first,selection,second:body.value,
 changed_without_reselection:first!==body.value,status:status.textContent}));
"""


def test_color_button_clears_selection_before_next_button_can_recolor_the_whole_note():
    node = shutil.which('node')
    if not node:
        pytest.skip('本地Node未安装，实际浏览器选区另行验收')
    directory = Path(__file__).parents[1] / 'src/study_loop/color_editor_component'
    result = subprocess.run([node, '-e', COLOR_BUTTON_HARNESS, str(directory / 'editor.js'),
                             str(directory / 'selection.js')], capture_output=True,
                            text=True, encoding='utf-8', timeout=15, check=True)
    observed = json.loads(result.stdout)
    assert '<span style="color:#D62828">重点 $x=y$</span>' in observed['first']
    assert observed['selection'] == [observed['start'], observed['start']]
    assert observed['changed_without_reselection'] is False
    assert observed['second'] == observed['first']
    assert observed['status'] == '请先选中要标注的文字或公式。'


def test_real_review_save_waits_for_atomic_current_text_and_executes_once(tmp_path, monkeypatch):
    from test_ai_review_support import RecordingProvider, review_case

    from src.study_loop import ai_review_ui as ui
    from src.study_loop import color_editor as editor
    from src.study_loop.ai_review import AIReviewService

    case = review_case(tmp_path / 'NON_REAL_ATOMIC_SAVE')
    provider = RecordingProvider('NON_REAL_ORIGINAL_AI')
    service = AIReviewService(case.store)
    document = service.generate([case.selected.id], provider)
    identity = document['id']
    key = ui.PREFIX + 'body_' + identity
    component_key = key + '_color_component'
    state = {
        ui.PREFIX + 'document_id': identity, ui.PREFIX + 'view_' + identity: '编辑',
        ui.PREFIX + 'editor_live_' + identity: True, key: document['working'],
        ui.PREFIX + 'loaded_' + identity: document['working'],
        ui.PREFIX + 'revision_' + identity: document['working_revision'],
        key + '_color_ready': True, key + '_color_ack': 1,
        'sl_active_color_editor': key,
    }
    fake = SimpleNamespace(session_state=state)
    monkeypatch.setattr(ui, 'st', fake)
    monkeypatch.setattr(editor, 'st', fake)
    monkeypatch.setattr(ui, 'review_service', lambda store: service)
    ui._save(case.store)
    assert service.versions(identity) == []  # Native click cannot toast/save stale input.
    token = state[key + '_color_pending']['token']
    changed = '<span style="color:#D62828">NON_REAL_NEW_INPUT $x=y$</span>\n'
    state[component_key] = {'text': changed, 'serial': 3, 'external_revision': 0,
                            'flush_token': token}
    callback = partial(ui._persist, user_edit=True, document_id=identity)
    editor._changed(key, component_key, 200000, callback, (case.store,))
    assert service.document(identity)['working'] == changed
    assert [v['content'] for v in service.versions(identity)] == [changed]
    assert service.document(identity)['raw_ai'] == 'NON_REAL_ORIGINAL_AI'
    editor._changed(key, component_key, 200000, callback, (case.store,))
    assert len(service.versions(identity)) == 1 and len(provider.prompts) == 1
    assert key + '_color_pending' not in state


def test_old_events_cannot_overwrite_newer_serial_or_explicit_source_edit(monkeypatch):
    from src.study_loop import color_editor as editor

    state = {'body': 'CURRENT', 'body_color_ack': 4, 'body_color_external': 2}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    calls = []
    for event in [
        {'text': 'OLDER', 'serial': 3, 'external_revision': 2},
        {'text': 'OLD_FRAME', 'serial': 5, 'external_revision': 1},
    ]:
        state['component'] = event
        editor._changed('body', 'component', 200, lambda: calls.append(True), ())
    assert state['body'] == 'CURRENT' and calls == []


def test_source_only_fallback_does_not_wait_for_missing_iframe(monkeypatch):
    from src.study_loop import color_editor as editor

    state = {'body': 'SOURCE_ONLY'}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    assert editor.defer_color_action('body', lambda: None) is False
    assert 'body_color_pending' not in state


def test_mounted_but_unready_iframe_waits_instead_of_saving_old_source(monkeypatch):
    from src.study_loop import color_editor as editor

    state = {'body': 'OLD_SOURCE', 'body_color_mounted': True,
             'body_color_source_active': True, 'body_color_source_open': True,
             'body_color_source_owner': False}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    assert editor.defer_color_action('body', lambda: None) is True
    assert state['body_color_pending']['token'] == 1


def test_legacy_native_flags_cannot_authorize_or_replace_single_editor_input(monkeypatch):
    from src.study_loop import color_editor as editor
    blue = '<span style="color:#2563EB">CURRENT_INPUT $x>0$</span>'
    state = {'body': blue, 'body_color_mounted': True,
             'body_color_source_open': True, 'body_color_source_owner': True,
             'body_color_source_generation': 2, 'body_color_source': 'STALE_NATIVE',
             'body_color_source_2': 'ANOTHER_STALE_NATIVE'}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    assert editor.confirmed_color_body('body') == blue
    assert state['body'] == blue
    assert editor.defer_color_action('body', lambda: None) is True
    assert state['body_color_pending']['token'] == 1


def test_flushed_action_uses_exact_receipt_even_if_other_code_replayed_old_body(monkeypatch):
    from src.study_loop import color_editor as editor
    from src.study_loop.models import StudyError

    blue = '<span style="color:#2563EB">BLUE $x>0$</span>'
    state = {'body': 'OLD_SOURCE', 'body_color_ack': 5, 'body_color_external': 2,
             'body_color_flushed': {'text': blue, 'serial': 5,
                                    'external_revision': 2, 'token': 17}}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    assert editor.confirmed_color_body('body', flushed=True) == blue
    assert state['body'] == blue
    state['body_color_ack'] = 6
    with pytest.raises(StudyError, match='尚未确认'):
        editor.confirmed_color_body('body', flushed=True)


def test_unmounted_read_view_uses_canonical_body_not_any_obsolete_auxiliary_value(monkeypatch):
    from src.study_loop import color_editor as editor
    state = {'body': 'CURRENT_ACKNOWLEDGED_BODY', 'body_color_mounted': False,
             'body_color_ready': False, 'body_color_source_open': True,
             'body_color_source_owner': True, 'body_color_source': 'OBSOLETE_NATIVE'}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    assert editor.confirmed_color_body('body') == 'CURRENT_ACKNOWLEDGED_BODY'
    assert editor.defer_color_action('body', lambda: None) is False
    assert 'body_color_pending' not in state


def test_single_editor_has_no_native_writer_or_mode_feedback_loop(tmp_path):
    case, app, identity, prefix = _memory_component_case(tmp_path)
    editor_id = app.get('component_instance')[0].proto.id
    assert len(app.get('component_instance')) == 1
    assert not any(area.label == '记忆正文' for area in app.text_area)
    assert not any(panel.label == 'Markdown 源码编辑' for panel in app.expander)
    for _ in range(3):
        app.run(timeout=30)
        assert app.get('component_instance')[0].proto.id == editor_id
        assert case.chats.memory(identity)['revision'] == 1
    assert app.session_state[prefix + '_draft'] == 'NON_REAL_OLD'


def _memory_component_case(tmp_path):
    from streamlit.testing.v1 import AppTest
    from test_ai_review_support import RecordingProvider, review_case

    from src.study_loop.ai_review import AIReviewService
    from src.study_loop.knowledge_memory import KnowledgeMemoryService
    from src.study_loop.memory_editor_ui import _prefix

    case = review_case(tmp_path / 'NON_REAL_WIDGET_REPLAY')
    service = AIReviewService(case.store)
    document = service.generate([case.selected.id], RecordingProvider('NON_REAL_RAW_AI'))
    service.save_version(document['id'])
    memories = KnowledgeMemoryService(case.store)
    extraction = memories.extract_review(document['id'], RecordingProvider('NON_REAL_OLD'))
    identity = memories.confirm(extraction['id'], 'NON_REAL_OLD', confirmed=True)
    app = AppTest.from_string(
        'from pathlib import Path\nimport streamlit as st\n'
        'from src.study_loop.storage import StudyStore\n'
        'from src.study_loop.memory_editor_ui import render_memory_editor\n'
        'render_memory_editor(StudyStore(Path(st.session_state["root"])), '
        'st.session_state["memory"], key_prefix="atomic")\n'
    )
    app.session_state['root'] = str(case.store.root)
    app.session_state['memory'] = identity
    app.run(timeout=30)
    next(button for button in app.button if button.label == '编辑记忆').click().run(timeout=30)
    prefix = _prefix(case.store, identity, 'atomic')
    return case, app, identity, prefix


def _component_event(app, component_id, event, stale_source=None):
    states = app._tree.get_widget_states()
    state = next((value for value in states.widgets if value.id == component_id), None)
    if state is None:
        state = states.widgets.add()
        state.id = component_id
    state.json_value = json.dumps(event)
    if stale_source is not None:
        identity, content = stale_source
        native = next((value for value in states.widgets if value.id == identity), None)
        if native is None:
            native = states.widgets.add()
            native.id = identity
        native.string_value = content
    return app._run(widget_state=states, timeout=30)


def test_real_widget_old_hidden_source_cannot_undo_blue_save_or_duplicate_revision(tmp_path):
    """Replay the actual component/native widget conflict, not a dictionary approximation."""
    case, app, identity, prefix = _memory_component_case(tmp_path)
    component_id = app.get('component_instance')[0].proto.id
    source_id = '$$ID-OBSOLETE_NATIVE-' + prefix + '_body_color_source'
    key = prefix + '_body'
    _component_event(app, component_id, {'text': 'NON_REAL_OLD', 'serial': 1})
    blue = '<span style="color:#2563EB">NON_REAL_BLUE $q=a/b$</span>\n'
    _component_event(app, component_id, {'text': blue, 'serial': 2})
    assert app.session_state[prefix + '_draft'] == blue
    assert key not in {area.key for area in app.text_area}
    next(button for button in app.button if button.label == '保存修改').click()
    _component_event(app, component_id, {'text': blue, 'serial': 2},
                     (source_id, 'NON_REAL_OLD'))
    assert case.chats.memory(identity)['revision'] == 1
    assert not any('修改已保存' in value.value for value in app.success)
    token = app.session_state[key + '_color_pending']['token']
    event = {'text': blue, 'serial': 3, 'flush_token': token}
    _component_event(app, component_id, event, (source_id, 'NON_REAL_OLD'))
    assert not app.exception and not app.error
    memory = case.chats.memory(identity)
    assert memory['revision'] == 2 and memory['content'] == blue
    assert app.session_state[prefix + '_draft'] == blue
    assert [value['content'] for value in case.chats.memory_revisions(identity)] == [
        blue, 'NON_REAL_OLD',
    ]
    _component_event(app, component_id, event, (source_id, 'NON_REAL_OLD'))
    assert len(case.chats.memory_revisions(identity)) == 2
    assert case.chats.memory(identity)['content'] == blue


def test_real_widget_save_before_first_ready_event_waits_for_current_payload(tmp_path):
    case, app, identity, prefix = _memory_component_case(tmp_path)
    component_id = app.get('component_instance')[0].proto.id
    next(button for button in app.button if button.label == '保存修改').click().run(timeout=30)
    assert case.chats.memory(identity)['revision'] == 1
    assert not any('修改已保存' in value.value for value in app.success)
    key = prefix + '_body'
    token = app.session_state[key + '_color_pending']['token']
    blue = '<span style="color:#2563EB">NON_REAL_EARLY_BLUE</span>'
    _component_event(app, component_id, {'text': blue, 'serial': 1, 'flush_token': token})
    assert not app.exception and not app.error
    assert case.chats.memory(identity)['revision'] == 2
    assert case.chats.memory(identity)['content'] == blue


def test_obsolete_auxiliary_payload_cannot_fake_ready_or_replace_first_blue_ack(tmp_path):
    case, app, identity, prefix = _memory_component_case(tmp_path)
    component_id = app.get('component_instance')[0].proto.id
    key = prefix + '_body'
    obsolete = '$$ID-OBSOLETE_NATIVE-' + key + '_color_source'
    _component_event(app, component_id, {'text': 'NO_VALID_ACK', 'serial': 0},
                     (obsolete, 'STALE_NATIVE_ONE'))
    assert app.session_state[prefix + '_draft'] == 'NON_REAL_OLD'
    assert not app.session_state.filtered_state.get(key + '_color_ready')
    next(button for button in app.button if button.label == '保存修改').click().run(timeout=30)
    token = app.session_state[key + '_color_pending']['token']
    _component_event(app, component_id, {'text': 'NO_VALID_ACK', 'serial': 0},
                     (obsolete, 'STALE_NATIVE_TWO'))
    assert app.session_state[key + '_color_pending']['token'] == token
    assert case.chats.memory(identity)['revision'] == 1
    blue = '<span style="color:#2563EB">NON_REAL_FIRST_BLUE_ACK</span>'
    _component_event(app, component_id, {'text': blue, 'serial': 1, 'flush_token': token})
    assert not app.exception and not app.error
    assert case.chats.memory(identity)['revision'] == 2
    assert case.chats.memory(identity)['content'] == blue


def test_legacy_auxiliary_mode_flags_do_not_bypass_unready_save(tmp_path):
    case, app, identity, prefix = _memory_component_case(tmp_path)
    component_id = app.get('component_instance')[0].proto.id
    key = prefix + '_body'
    app.session_state[key + '_color_source_open'] = True
    app.session_state[key + '_color_source_owner'] = True
    app.session_state[key + '_color_source'] = 'STALE_PREVIOUS_VERSION'
    next(button for button in app.button if button.label == '保存修改').click().run(timeout=30)
    assert case.chats.memory(identity)['revision'] == 1
    assert not any('修改已保存' in value.value for value in app.success)
    blue = '<span style="color:#2563EB">NON_REAL_CURRENT_INPUT</span>'
    token = app.session_state[key + '_color_pending']['token']
    _component_event(app, component_id, {'text': blue, 'serial': 1, 'flush_token': token})
    assert case.chats.memory(identity)['content'] == blue
    assert app.session_state[prefix + '_draft'] == blue


def test_obsolete_native_payload_after_preview_cannot_rewrite_acknowledged_body(tmp_path):
    case, app, identity, prefix = _memory_component_case(tmp_path)
    component_id = app.get('component_instance')[0].proto.id
    key = prefix + '_body'
    blue = '<span style="color:#2563EB">NON_REAL_PREVIEW $q=a/b$</span>'
    _component_event(app, component_id, {'text': blue, 'serial': 1})
    next(button for button in app.button if button.label == '预览修改').click().run(timeout=30)
    token = app.session_state[key + '_color_pending']['token']
    event = {'text': blue, 'serial': 2, 'flush_token': token}
    _component_event(app, component_id, event)
    assert app.session_state[prefix + '_mode'] == '预览'
    assert case.chats.memory(identity)['revision'] == 1
    obsolete = '$$ID-OBSOLETE_NATIVE-' + key + '_color_source_12'
    _component_event(app, component_id, event, (obsolete, 'STALE_AFTER_PREVIEW'))
    assert app.session_state[prefix + '_draft'] == blue
    assert app.session_state[key] == blue
    next(button for button in app.button if button.label == '保存修改').click().run(timeout=30)
    assert not app.exception and not app.error
    assert case.chats.memory(identity)['content'] == blue
    assert case.chats.memory(identity)['revision'] == 2


def test_ready_and_unchanged_flush_do_not_report_a_fictitious_user_edit(monkeypatch):
    from src.study_loop import color_editor as editor

    state = {'body': 'UNCHANGED', 'component': {'text': 'UNCHANGED', 'serial': 1}}
    monkeypatch.setattr(editor, 'st', SimpleNamespace(session_state=state))
    edits, actions = [], []
    editor._changed('body', 'component', 200, lambda: edits.append(True), ())
    assert state['body_color_ready'] is True and state['body_color_ack'] == 1
    assert edits == []
    assert editor.defer_color_action('body', lambda: actions.append(state['body'])) is True
    state['component'] = {'text': 'UNCHANGED', 'serial': 2, 'flush_token': 1}
    editor._changed('body', 'component', 200, lambda: edits.append(True), ())
    assert edits == [] and actions == ['UNCHANGED']


@pytest.mark.parametrize('ready', [False, True])
def test_page_departure_remounts_editor_until_flush_then_switches_outside_callback(
    monkeypatch, ready,
):
    from src.study_loop import ai_review_ui as ui
    from src.study_loop import color_editor as editor

    class PageSwitch(BaseException):
        pass

    switches = []
    state = {'sl_active_color_editor': 'body', 'body': 'OLD', 'body_color_ready': ready,
             'body_color_mounted': True}
    def switch_page(page):
        switches.append(page)
        raise PageSwitch
    fake = SimpleNamespace(session_state=state, switch_page=switch_page)
    monkeypatch.setattr(ui, 'st', fake)
    monkeypatch.setattr(editor, 'st', fake)
    target = 'pages/NON_REAL_OTHER.py'
    learning = 'pages/18_学习整理.py'
    with pytest.raises(PageSwitch):
        ui.persist_review_route(target)
    assert switches == [learning]
    token = state['body_color_pending']['token']
    state['component'] = {'text': 'LATEST_BEFORE_LEAVING', 'serial': 1, 'flush_token': token}
    editor._changed('body', 'component', 200, None, ())
    assert switches == [learning]  # The callback itself never calls switch_page/rerun.
    with pytest.raises(PageSwitch):
        ui.persist_review_route(learning)
    assert switches == [learning, target]
    assert state['body'] == 'LATEST_BEFORE_LEAVING'


def test_delayed_editor_callback_is_bound_to_its_original_document(tmp_path, monkeypatch):
    from test_ai_review_support import RecordingProvider, review_case

    from src.study_loop import ai_review_ui as ui
    from src.study_loop.ai_review import AIReviewService

    case = review_case(tmp_path / 'NON_REAL_BOUND_DOCUMENT')
    service = AIReviewService(case.store)
    first = service.generate([case.selected.id], RecordingProvider('FIRST_AI'))
    second = service.generate([case.selected.id], RecordingProvider('SECOND_AI'))
    state = {ui.PREFIX + 'document_id': second['id'],
             ui.PREFIX + 'body_' + first['id']: 'FIRST_EDIT_ARRIVED_LATE',
             ui.PREFIX + 'revision_' + first['id']: first['working_revision']}
    monkeypatch.setattr(ui, 'st', SimpleNamespace(session_state=state))
    monkeypatch.setattr(ui, 'review_service', lambda store: service)
    assert ui._persist(case.store, user_edit=True, document_id=first['id']) is True
    assert service.document(first['id'])['working'] == 'FIRST_EDIT_ARRIVED_LATE'
    assert service.document(second['id']) == second
    assert state[ui.PREFIX + 'document_id'] == second['id']
