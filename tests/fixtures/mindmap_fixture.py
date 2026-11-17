"""NON_REAL_TEST_DATA: explicit-only infrastructure fixture, never startup seed.

Every caller must provide a disposable database and material directory. This
fixture contains no subject matter, answers, scores or user learning history.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from src.database import Database
from src.knowledge_object_service import KnowledgeObjectService


def seed_mindmap(database: Database, material_root: Path, *, large_count: int = 0) -> dict:
    """Create source anchors, a cycle, multiple relation parents and an isolate."""
    material_root.mkdir(parents=True, exist_ok=True)
    pdf_path = material_root / "NON_REAL_TEST_DATA.pdf"
    with fitz.open() as document:
        for number in (1, 2):
            page = document.new_page(width=420, height=300)
            page.insert_text((40, 60), f"NON_REAL_TEST_DATA / TEST_SOURCE_PAGE_{number}",
                             fontsize=12)
        document.save(pdf_path)
    document = database.create_document(
        title="NON_REAL_TEST_DATA · TEST_DOCUMENT_01", filename=pdf_path.name,
        source_path=pdf_path, sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        page_count=2,
    )
    pages = []
    with fitz.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            image_path = material_root / f"TEST_SOURCE_PAGE_{index + 1}.png"
            page.get_pixmap().save(image_path)
            pages.append(database.create_page(
                document_id=document.id, page_number=index + 1, image_path=image_path,
                extracted_text=page.get_text(), status="ready",
            ))
    service = KnowledgeObjectService(database)
    items = {}
    titles = {
        "A": "TEST_NODE_A",
        "B": "TEST_NODE_B",
        "C": "TEST_NODE_C",
        "LONG": "TEST_NODE_LONG · " + "基础设施长中文标题用于验证安全截断与完整回查" * 3,
        "ISOLATED": "TEST_NODE_ISOLATED <script>alert('NON_REAL_TEST_DATA')</script>",
    }
    for key, title in titles.items():
        sources = [("page", pages[0 if key == "A" else 1].id, "NON_REAL_TEST_DATA")]
        if key == "ISOLATED":
            sources = []
        item = service.create(
            kind="concept", title=title,
            content=f"NON_REAL_TEST_DATA / TEST_CONTENT_{key}\n公式阅读夹具：$x^2 + y^2$。",
            epistemic_basis="source_derived" if sources else "personal_judgment",
            source_links=sources,
        ).knowledge_object
        items[key] = item.id
    # Existing directed relations remain cyclic and multi-parent; no fake tree.
    service.add_relation(items["A"], items["B"], relation_type="requires",
                         description="NON_REAL_TEST_DATA")
    service.add_relation(items["B"], items["C"], relation_type="derived_from",
                         description="NON_REAL_TEST_DATA")
    service.add_relation(items["C"], items["A"], relation_type="relates_to",
                         description="NON_REAL_TEST_DATA")
    service.add_relation(items["LONG"], items["B"], relation_type="supports",
                         description="NON_REAL_TEST_DATA")
    for index in range(large_count):
        service.create(kind="concept", title=f"TEST_NODE_SCALE_{index:04d}",
                       content="NON_REAL_TEST_DATA", epistemic_basis="personal_judgment")
    return {"document_id": document.id, "page_ids": [page.id for page in pages],
            "knowledge_ids": items, "pdf_path": str(pdf_path)}
