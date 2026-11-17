"""Non-learning synthetic parser fixtures; never evidence of learning effectiveness.

All message bodies below are parser markers/configuration labels, not actual user
dialogue, academic questions, answers, marks or historical records.
"""

from __future__ import annotations

import hashlib
import io
import json
import socket

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.study_loop.components import import_source
from src.study_loop.conversation_import import html_pages
from src.study_loop.models import Page, Question, Source, StudyError
from src.study_loop.storage import StudyStore
from src.study_loop.structure import local_drafts
from src.study_loop.vesti import MAX_TEXT, decode_text, json_pages, text_pages


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Parser and storage regression tests may not contact remote resources."""
    def deny(*args, **kwargs):
        raise AssertionError("解析器回归禁止外网调用")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def source(kind: str = "json", *, transcript: bool = False) -> Source:
    return Source("synthetic-parser-only", kind, f"parser.{kind}", "", "", transcript=transcript)


def encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Keep path ownership records and materials outside real runtime locations."""
    import src.study_loop.storage as storage

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "local"))
    monkeypatch.setattr(
        storage, "configured_platform_storage_paths", lambda: (tmp_path / "platform",)
    )
    return StudyStore(tmp_path / "study", create=True)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32", "gb18030"])
def test_common_text_encodings_preserve_unicode(encoding):
    text = "非学习解析测试标记\r\n配置键：绿色\n"
    assert decode_text(text.encode(encoding)) == text


@pytest.mark.parametrize("data", [b"abc\x00def", b"abc\x01def", b"\xff\xfeA"])
def test_binary_or_broken_encoding_is_explicit_error(data):
    with pytest.raises(StudyError, match="编码|控制字符"):
        decode_text(data)


def test_short_text_retains_existing_snapshot_identity_and_metadata():
    parsed = text_pages(source("txt"), b"parser-marker\n")
    assert len(parsed) == 1 and parsed[0].id == "synthetic-parser-only:t0"
    assert parsed[0].text == "parser-marker\n"
    assert parsed[0].metadata == {
        "format": "txt",
        "line_start": 1,
        "line_end": 1,
        "content_mode": "unknown",
        "record_kind": "text",
    }


def test_long_text_chunks_losslessly_with_offsets_and_stable_ids():
    text = "parser-marker\n" * 40_000
    parsed = text_pages(source("txt"), text.encode())
    assert len(parsed) > 1
    assert "".join(page.text for page in parsed) == text
    assert all(len(page.text) <= MAX_TEXT for page in parsed)
    for page in parsed:
        meta = page.metadata
        assert text[meta["character_start"] : meta["character_end"]] == page.text
    assert parsed == text_pages(source("txt"), text.encode())


def test_generic_json_preserves_role_time_pointer_and_never_promotes_answers():
    raw = {
        "id": "fixture",
        "title": "非学习解析测试",
        "platform": "fixture-only",
        "messages": [
            {
                "id": "m1",
                "role": "assistant",
                "timestamp": "fixture-time",
                "content": "1. 配置键\n答案：marker-a\n解析：marker-b",
            },
        ],
    }
    page = json_pages(source(), encoded(raw))[0]
    assert page.metadata["pointer"] == "/messages/0"
    assert page.metadata["role"] == "assistant"
    assert page.metadata["created_at"] == "fixture-time"
    assert page.metadata["conversation"]["platform"] == "fixture-only"
    draft = local_drafts(source(), page)[0]
    assert draft.material_kind == "note"
    assert draft.question.content == raw["messages"][0]["content"]
    assert draft.question.observed_answer == draft.question.reference_answer == ""
    assert draft.verification_status == "pending"


@pytest.mark.parametrize(
    "shape,pointer",
    [
        (lambda m: m, "/0"),
        (lambda m: {"messages": m}, "/messages/0"),
        (lambda m: [{"id": "c", "messages": m}], "/0/messages/0"),
        (lambda m: {"conversations": [{"messages": m}]}, "/conversations/0/messages/0"),
    ],
)
def test_explicit_generic_json_container_shapes(shape, pointer):
    parsed = json_pages(source(), encoded(shape([{"role": "user", "content": "parser marker"}])))
    assert len(parsed) == 1 and parsed[0].metadata["pointer"] == pointer


def test_mapping_retains_all_branches_and_escaped_pointers():
    raw = [
        {
            "title": "非学习分支测试",
            "current_node": "child-two",
            "mapping": {
                "root": {"message": None, "parent": None, "children": ["a/b~c", "child-two"]},
                "a/b~c": {
                    "parent": "root",
                    "message": {
                        "id": "first",
                        "author": {"role": "assistant"},
                        "create_time": 123,
                        "content": {"parts": ["branch-one"], "content_type": "text"},
                    },
                },
                "child-two": {
                    "parent": "root",
                    "message": {
                        "id": "second",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["branch-two"]},
                    },
                },
            },
        }
    ]
    parsed = json_pages(source(), encoded(raw))
    assert [page.text for page in parsed] == ["branch-one", "branch-two"]
    assert parsed[0].metadata["pointer"] == "/0/mapping/a~1b~0c/message"
    assert parsed[0].metadata["branch"]["parent"] == "root"
    assert parsed[0].metadata["created_at"] == 123
    assert all("所有分支" in page.metadata["warning"] for page in parsed)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"schema_version": "vesti_export.v999"},
        {"messages": "wrong"},
        {"messages": [{"role": []}]},
        [{"unknown": "value"}],
        {"mapping": {"node": []}},
        {"mapping": {"node": {"message": "wrong"}}},
        {"messages": []},
    ],
)
def test_unknown_or_malformed_json_does_not_guess_a_transcript(raw):
    with pytest.raises(StudyError):
        json_pages(source(), encoded(raw))


@pytest.mark.parametrize(
    "data",
    [
        b'{"messages":[],"messages":[]}',
        b'{"messages":[],"x":NaN}',
        b'{"messages":',
    ],
)
def test_duplicate_fields_nonstandard_constants_and_truncation_fail(data):
    with pytest.raises(StudyError):
        json_pages(source(), data)


def test_json_utf16_and_nontext_attachment_descriptions_are_retained():
    raw = {
        "messages": [
            {
                "role": "user",
                "content": {
                    "parts": [{"type": "image", "url": "https://example.invalid/never-fetch"}],
                },
            }
        ]
    }
    page = json_pages(source(), json.dumps(raw).encode("utf-16"))[0]
    assert "never-fetch" in page.text


def test_vesti_full_archive_keeps_existing_pointer_and_snapshot_identity():
    raw = {
        "schema_version": "vesti_export.v1",
        "data": {
            "conversations": [{"id": 1, "title": "非学习 Vesti 解析测试"}],
            "messages": [
                {
                    "id": 2,
                    "conversation_id": 1,
                    "role": "user",
                    "content_text": "parser marker",
                    "created_at_iso": "fixture-time",
                }
            ],
        },
    }
    parsed = json_pages(source(), encoded(raw))
    page = parsed[1]
    expected = hashlib.sha256(b"vesti-v1:synthetic-parser-only:/data/messages/0").hexdigest()
    assert page.id == expected
    assert page.metadata["pointer"] == "/data/messages/0"
    assert page.metadata["created_at"] == "fixture-time"
    assert page.metadata["conversation"]["id"] == 1
    assert "parser marker" in page.text


def test_oversized_message_is_chunked_without_truncation():
    message = "parser-marker " * 30_000
    parsed = json_pages(
        source(), encoded({"messages": [{"role": "assistant", "content": message}]})
    )
    assert len(parsed) > 1 and "".join(page.text for page in parsed) == message
    assert all(page.metadata["pointer"] == "/messages/0" for page in parsed)


def test_html_is_inert_and_keeps_visible_text_only():
    original = (
        "<html><head><script>hidden-code()</script><style>hidden-style</style></head>"
        '<body><p>parser &amp; marker</p><img src="https://example.invalid/no-fetch">'
        "<script>hidden-body()</script><div>second-marker</div></body></html>"
    )
    parsed = html_pages(source("html"), original.encode())
    text = "".join(page.text for page in parsed)
    assert "parser & marker" in text and "second-marker" in text
    assert "hidden" not in text and "no-fetch" not in text
    assert parsed[0].metadata["format"] == "html_text_v1"
    assert "精确依据" in parsed[0].metadata["text_basis"]


def test_source_bytes_duplicate_and_reimport_stay_unchanged(store):
    original = encoded({"messages": [{"role": "user", "content": "parser-only marker"}]})
    received = store.receive("synthetic.json", original, "json", transcript=True)
    import_source(store, received.id)
    pages_before, drafts_before = store.pages(received.id), store.drafts()
    assert store.receive("renamed.json", original, "json", transcript=True) == received
    import_source(store, received.id)
    assert store.read_source(received) == original
    assert store.pages(received.id) == pages_before
    assert store.drafts() == drafts_before
    assert len(store.sources()) == 1 and store.items() == []


def test_html_route_preserves_raw_file_and_does_not_mark_verified(store):
    original = b"<p>parser marker</p><script>do-not-run()</script>"
    received = store.receive("synthetic.html", original, "html", transcript=True)
    import_source(store, received.id)
    assert store.read_source(received) == original
    assert len(store.drafts()) == 1
    assert store.drafts()[0].verification_status == "pending"


def test_interrupted_draft_creation_resumes_without_overwriting_sources(store, monkeypatch):
    original = encoded({"messages": [
        {"role": "user", "content": "parser-first"},
        {"role": "assistant", "content": "parser-second"},
    ]})
    received = store.receive("interruption.json", original, "json")
    persist = store.add_draft
    calls = 0

    def stop_second(draft):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interruption")
        return persist(draft)

    monkeypatch.setattr(store, "add_draft", stop_second)
    with pytest.raises(OSError, match="synthetic interruption"):
        import_source(store, received.id)
    before = store.pages(received.id)
    assert len(store.drafts()) == 1
    monkeypatch.setattr(store, "add_draft", persist)
    import_source(store, received.id)
    assert store.read_source(received) == original
    assert store.pages(received.id) == before
    assert len(store.drafts()) == 2 and store.items() == []


def test_malformed_input_keeps_original_and_creates_no_drafts(store):
    original = b'{"messages":'
    received = store.receive("bad.json", original, "json")
    with pytest.raises(StudyError):
        import_source(store, received.id)
    assert store.read_source(received) == original
    assert store.pages(received.id) == [] and store.drafts() == []


def test_synthetic_subnumber_and_answer_fields_stay_separate():
    text = "1. 解析器字段校验\n(1) 首段标记\n答案：marker-a\n(2) 次段标记\n解析：marker-b"
    page = text_pages(source("txt"), text.encode())[0]
    draft = local_drafts(source("txt"), page)[0]
    assert [child.number for child in draft.question.children] == ["1", "2"]
    assert draft.question.children[0].observed_answer.strip() == "marker-a"
    assert draft.question.children[0].observed_analysis == ""
    assert draft.question.children[1].observed_answer == ""
    assert draft.question.children[1].observed_analysis.strip() == "marker-b"


def test_revision_editor_initializes_existing_reference_values(monkeypatch):
    import src.study_loop.import_ui as ui

    question = Question(
        content="非学习编辑器字段测试",
        answer_status="user_note",
        reference_answer="retained marker",
        reference_analysis="retained detail",
    )
    monkeypatch.setattr(ui, "_parser_question", question, raising=False)
    app = AppTest.from_string(
        "from src.study_loop.import_ui import _edit_question, _parser_question\n"
        "_edit_question(_parser_question, 'parser-fixture')\n"
    ).run(timeout=20)
    assert not app.exception
    assert question.answer_status == "user_note"
    assert question.reference_answer == "retained marker"
    assert question.reference_analysis == "retained detail"


def test_manual_paste_can_explicitly_mark_dialogue_without_role_heuristics(store, monkeypatch):
    import src.study_loop.import_ui as ui

    monkeypatch.setattr(ui, "_parser_store", store, raising=False)
    monkeypatch.setattr(
        ui, "_enqueue", lambda saved_store, identity: import_source(saved_store, identity)
    )
    app = AppTest.from_string(
        "from src.study_loop.import_ui import render_input, _parser_store\n"
        "render_input(_parser_store)\n"
    ).run(timeout=20)
    assert not app.radio
    assert not app.exception
    app.text_area[0].set_value("1. parser config\n答案：non-learning-marker")
    app.checkbox(key="sl_manual_dialogue").check()
    next(b for b in app.button if b.label == "添加文字").click().run(timeout=20)
    assert not app.exception
    assert store.sources()[0].transcript is True
    assert store.drafts()[0].material_kind == "note"
    assert store.drafts()[0].question.observed_answer == ""


@pytest.mark.parametrize("data", [
    b'{"messages":[{"role":"user","content":"\\ud800"}]}',
    b'{"messages":[{"role":"user","content":"marker","\\udfff":1}]}',
])
def test_lone_json_surrogate_fails_before_partial_persistence(store, data):
    received = store.receive("surrogate.json", data, "json")
    with pytest.raises(StudyError, match="孤立代理字符"):
        import_source(store, received.id)
    assert store.read_source(received) == data
    assert store.pages(received.id) == [] and store.drafts() == []


def test_paired_json_surrogates_are_valid_unicode():
    data = b'{"messages":[{"role":"user","content":"\\ud83d\\ude00"}]}'
    assert json_pages(source(), data)[0].text == "\U0001f600"


def test_long_vesti_offsets_explicitly_refer_to_generated_display_text():
    raw = {"schema_version": "vesti_export.v1", "data": {
        "conversations": [{"id": 1}],
        "messages": [{"id": 2, "conversation_id": 1, "role": "assistant",
                      "content_text": "parser-marker " * 30_000}],
    }}
    pages = [page for page in json_pages(source(), encoded(raw))
             if page.metadata.get("record_kind") == "message"]
    displayed = "".join(page.text for page in pages)
    assert displayed.startswith("【AI 原话")
    for page in pages:
        metadata = page.metadata
        assert "展示文本" in metadata["character_basis"]
        assert "不是原 JSON" in metadata["character_basis"]
        assert displayed[metadata["character_start"]:metadata["character_end"]] == page.text


def test_image_only_source_view_keeps_frame_provenance_in_backend(store, monkeypatch):
    """UI-only non-learning image fixture; actual MPO decoding has separate tests."""
    import src.study_loop.import_ui as ui

    image = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(image, format="PNG")
    received = store.receive("ui-marker.png", image.getvalue(), "image")
    store.put_page(Page(
        "ui-frame", received.id, None, image=received.blob, image_sha256=received.sha256,
        metadata={"source_frame": 2, "warning": "NON-LEARNING FRAME PROVENANCE MARKER"},
    ))
    monkeypatch.setattr(ui, "_parser_store", store, raising=False)
    app = AppTest.from_string(
        "from src.study_loop.import_ui import render_source, _parser_store\n"
        f"render_source(_parser_store, {received.id!r}, ['ui-frame'], prefix='ui-fixture')\n"
    ).run(timeout=20)
    assert not app.exception
    captions = [element.value for element in app.caption]
    assert len(app.get("image")) == 1
    assert not captions
    assert store.pages(received.id)[0].metadata["source_frame"] == 2
    assert store.pages(received.id)[0].metadata["warning"] == (
        "NON-LEARNING FRAME PROVENANCE MARKER"
    )


def test_docx_rendered_page_label_does_not_claim_original_pdf_physical_page():
    from src.study_loop.import_ui import _page_label

    page = Page("parser-page", "parser-source", 1)
    assert _page_label(page, source_kind="docx") == "第 1 页"
    assert _page_label(page, source_kind="pdf") == "第 1 页"
