"""NON_REAL_TEST_DATA: real current services, temporary stores, no legacy facade."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.study_loop.components import import_source
from src.study_loop.models import Draft, Page, Question
from src.study_loop.storage import StudyStore


def image_bytes(color: str = "blue", *, jpeg: bool = False) -> bytes:
    """Generate a marked test image; EXIF is synthetic, never personal metadata."""
    output = io.BytesIO()
    picture = Image.new("RGB", (80, 60), color)
    if jpeg:
        exif = Image.Exif()
        exif[0x010F] = "NON_REAL_TEST_DATA_CAMERA"
        picture.save(output, format="JPEG", exif=exif)
    else:
        picture.save(output, format="PNG")
    return output.getvalue()


def image_case(root: Path, *, jpeg: bool = False):
    """Exercise the real receipt, Import Core and local draft extraction."""
    store = StudyStore(root, create=True)
    raw = image_bytes(jpeg=jpeg)
    source = store.receive("NON_REAL_TEST_DATA" + (".jpg" if jpeg else ".png"), raw, "image")
    import_source(store, source.id)
    page = store.pages(source.id)[0]
    return store, source, page, store.drafts()[0], raw


def pdf_case(root: Path):
    """Two distinguishable synthetic physical pages, imported through Import Core."""
    with fitz.open() as document:
        for number in (1, 2):
            page = document.new_page(width=240, height=180)
            page.insert_text((20, 30), f"NON_REAL_TEST_DATA SOURCE_PAGE_{number}")
        raw = document.tobytes()
    store = StudyStore(root, create=True)
    source = store.receive("NON_REAL_TEST_DATA.pdf", raw, "pdf")
    import_source(store, source.id)
    pages = store.pages(source.id)
    draft = store.drafts()[0]
    draft.page_ids = [page.id for page in pages]
    return store, source, pages, draft, raw


def text_case(root: Path, *, subject: str = "TEST_COURSE"):
    """Use explicit source metadata and real local parsing, without model calls."""
    store = StudyStore(root, create=True)
    raw = (
        f"科目：{subject}\n章节：TEST_CHAPTER\n"
        "1. NON_REAL_TEST_DATA SHARED_CONTEXT\n"
        "(1) NON_REAL_TEST_DATA UNIT_A\n(2) NON_REAL_TEST_DATA UNIT_B"
    ).encode()
    source = store.receive("NON_REAL_TEST_DATA.txt", raw, "txt")
    import_source(store, source.id)
    return store, source, store.drafts()[0]


def panel(store: StudyStore, name: str, *, advanced: bool = False) -> AppTest:
    """Render a real panel against the supplied temporary store; no service mocks."""
    assert name in {"drafts", "review", "summary"}
    module = "import_ui" if name == "drafts" else "review_ui"
    extra = (
        "\nfrom src.study_loop.review_ui import render_review_advanced\n"
        "render_review_advanced(store)\n"
        if advanced and name == "review" else ""
    )
    if advanced and name == "drafts":
        extra = (
            "\nfrom src.study_loop.import_ui import render_draft_advanced\n"
            "render_draft_advanced(store, section='source')\n"
        )
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.models import StudyError\n"
        f"from src.study_loop.{module} import render_{name}\n"
        "store = StudyStore(Path(st.session_state['test_root']))\n"
        "try:\n"
        f"    render_{name}(store)\n"
        "except StudyError as error:\n    st.warning(str(error))\n"
        + extra
    )
    app.session_state["test_root"] = str(store.root)
    return app.run(timeout=30)


def submit_attempt(store: StudyStore, session: str, item: Draft, *, index: int = 1) -> None:
    """A synthetic contract event, not a claim about the user's learning."""
    store.record_attempt(
        session,
        item.id,
        item.question.units()[0].id,
        outcome="partial",
        answer=f"NON_REAL_TEST_DATA RESPONSE_{index}",
        note="NON_REAL_TEST_DATA",
        independent=True,
        attempted=True,
        error_tags=["TEST_TAG"],
        submission_id=f"TEST_SUBMISSION_{index}",
    )


def compound_image_draft(store: StudyStore, source_id: str, page: Page) -> Draft:
    """Two leaf units share a single verified source image in the current model."""
    draft = Draft(
        "TEST_COMPOUND",
        source_id,
        [page.id],
        Question(
            id="TEST_PARENT",
            number="6",
            content="NON_REAL_TEST_DATA SHARED_FIGURE",
            children=[
                Question(id="TEST_A", number="1", content="NON_REAL_TEST_DATA OPTION_A"),
                Question(id="TEST_B", number="2", content="NON_REAL_TEST_DATA OPTION_B"),
            ],
        ),
    )
    store.add_draft(draft)
    return draft


def capture_images(monkeypatch) -> list[bytes]:
    """Observe bytes while still exercising Streamlit's real image renderer."""
    import streamlit as st

    original = st.image
    images = []

    def capture(data, *args, **kwargs):
        images.append(data)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(st, "image", capture)
    return images


def source_panel(store: StudyStore, source_id: str, page_ids: list[str]) -> AppTest:
    """Exercise the actual source renderer, including the application's error boundary."""
    app = AppTest.from_string(
        "from pathlib import Path\nimport streamlit as st\n"
        "from src.study_loop.storage import StudyStore\n"
        "from src.study_loop.models import StudyError\n"
        "from src.study_loop.import_ui import render_source\n"
        "store = StudyStore(Path(st.session_state['test_root']))\n"
        "try:\n"
        "    render_source(store, st.session_state['source_id'], "
        "st.session_state['page_ids'], prefix='test')\n"
        "except StudyError as error:\n    st.warning(str(error))\n"
    )
    app.session_state["test_root"] = str(store.root)
    app.session_state["source_id"] = source_id
    app.session_state["page_ids"] = page_ids
    return app.run(timeout=30)
