"""Durable, editable AI review documents in the existing conversation sidecar."""

from __future__ import annotations

import json
import sqlite3
from uuid import uuid4

from src.ai.provider import CompletionProvider

from . import chat
from .chat_citations import visible_answer
from .chat_store import ChatStore
from .models import Draft, StudyError, now
from .storage import StudyStore, encode

MAX_CONTENT = 200_000
REVIEW_REQUEST = """请根据本次选中的材料生成一份可供我直接编辑和保存的学习复盘稿。
沿用课程考试范围约束，主动汇总所选材料的关键公式，逐项说明符号含义、单位与适用条件，
不需要我另行提醒整理公式。说明哪些题目条件触发相应方法、关键步骤及每一步为什么成立；
串联同题小问、概念、公式和知识点之间的联系，讲清易错点与适用边界。
按知识与方法重新组织并独立核对，不按原解析逐段复述。
默认重点是可迁移的公式、方法主线和必要边界：用紧凑公式表、方法联系与易错点组织，
不要逐小问重写完整解答，也不重复题干数据、原解析与结果表；仅保留核查确有必要的少量代入。
每条公式的取值范围须紧接该式明确写出，不能借用其他公式的限制代替说明。
若某量按定义是非负大小而其输入可正可负，先由定义检查符号，再说明关系在何范围成立。
边界以本题所需前提与变形合法性为主，不展开题外工况和额外案例；必要扩展时逐量核验，
不得把不同情况的方向、大小或变化趋势合并为同一结论。
易错点说清正确关系即可，不为凑栏目添加错误数值示例；不猜测出题者为何给出或合并某数据，
不得用相关性代替因果，未由题干、定义或可靠依据支持的因果说明应省略。
公式不仅用于代入的主式要正确，表格、说明、易错提醒和边界讨论中的公式也要逐一核验。
先定义各量的对象与计量约定，区分整体与分量、输入与输出、带方向的代数量与非负的大小量。
改变表示或基准时，从完整定义式写出代换后的完整等价式，再核对求和、平均、比例与变换系数；
不能用“只去掉或加上某系数”代替推导，也不能只凭量纲相同就认为系数正确。
说明每个实际使用的符号及单位；单位转换要显式一致，尤其注意公式隐含的基本单位与显示单位。
每条公式须限定变量取值范围与成立条件。若从本题条件扩展到零值、负值或其他工况，
重新检查变量定义、符号约定和适用条件，不把本题范围内成立的表达直接宣称为任意情形成立；
涉及大小与有符号量时给出明确约定。无助于本题理解且未经核实的题外扩展应省略。
本题采用的范围不等于公式或模型的极限；分别判断定义式、恒等关系和含除法的变形式的适用域，
不能把本题未讨论的边界直接宣布为模型不覆盖或公式失效。
解释错误算式的含义时，也必须从各量定义和完整关系验证它究竟代表什么；
不得把漏项、漏系数或量纲相同的表达未经推导就等同于另一对象。无必要时只保留正确关系与条件。
同一符号不得在不同公式中默换单位；量方程与代入数值的计量单位要说明一致，中间新引入的符号也须定义。
文字中的加减和变化方向也要先从完整平衡式解出当前所求量，再结合保持不变的条件判断；
区分正向过程中的扣除与反向求解时的还原，不能沿用同一句增减口令。
比值变化须明确分子与分母并重新计算，不能把有单位的项直接加减于无量纲结果。
提交前核对公式表、文字解释、算式和结论是否互相一致；用定义或独立关系复核关键系数，
以允许的边界值检验外推断言。发现可纠正的错误就在本次正文内纠正，不把已知错误留作人工任务。
这些核验在同一次生成内完成，只呈现清楚、紧凑的复盘正文，不输出自检清单或冗长的核验流水。
只核对本次上下文中实际存在的题目已知、材料参考说法、历史AI记录与用户注释；发现问题时说明理由，不能照抄为真。
记录为空或缺失时，不得声称已核对该类记录，也不编造其数值、结论或一致性；没有实质问题无需另设来源核对栏目。
个人问题、错因或理解障碍只在真实作答、我的明确反馈或相关对话有证据时记录；没有证据时
不推断我已经作答、答错或掌握，不编造个人评语，也不列出等待填写的空栏目。
只使用本次选中的学习条目，不带入同文件同页的其他题目。
直接返回完整、清楚的Markdown复盘正文与数学公式，不返回表单、JSON或开发诊断。
这只是AI候选稿；不修改原材料、不自动记入掌握状态或长期记忆。"""
# Shared mathematical review and colour policies are supplied once by chat.POLICY.


class AIReviewService:
    """Append AI candidates and saved human versions without replacing user history."""

    def __init__(self, store: StudyStore) -> None:
        self.store = store
        self.chats = ChatStore(store)
        with self.chats.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ai_review_documents(
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id));
                CREATE TABLE IF NOT EXISTS ai_review_versions(
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    number INTEGER NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(document_id, number),
                    FOREIGN KEY(document_id) REFERENCES ai_review_documents(id));
            """)

    def _items(self) -> dict[str, Draft]:
        return {item.id: item for item in [*self.store.drafts(), *self.store.items()]}

    @staticmethod
    def _readable_text(item: Draft) -> bool:
        pending = [item.question]
        while pending:
            question = pending.pop()
            if question.content.strip():
                return True
            pending.extend(question.children)
        return False

    def _selected(self, item_ids: list[str]) -> list[Draft]:
        if not item_ids or len(item_ids) > 100 or len(item_ids) != len(set(item_ids)):
            raise StudyError("请选择 1 至 100 份不重复的学习材料。")
        available = self._items()
        if not set(item_ids) <= available.keys():
            raise StudyError("所选学习材料已删除或不存在，请重新选择。")
        selected = [available[identity] for identity in item_ids]
        verified = set()
        for item in selected:
            if item.source_id not in verified:
                self.store.read_source(self.store.source(item.source_id))
                verified.add(item.source_id)
            self.store.referenced_pages(item.source_id, item.page_ids)
            if not self._readable_text(item):
                raise StudyError("所选材料没有可读文字，请先完成导入或文字识别。")
        return selected

    def options(self) -> list[dict]:
        """Return imported readable drafts and saved items, without confirming any answer."""
        saved = {item.id for item in self.store.items()}
        values, sources = [], {}
        for item in self._items().values():
            try:
                if item.source_id not in sources:
                    sources[item.source_id] = None
                    source = self.store.source(item.source_id)
                    self.store.read_source(source)
                    sources[item.source_id] = source
                source = sources[item.source_id]
                if source is None or not self._readable_text(item):
                    continue
                self.store.referenced_pages(item.source_id, item.page_ids)
            except StudyError:
                continue
            values.append(
                {
                    "id": item.id,
                    "source_id": item.source_id,
                    "filename": source.name,
                    "number": item.question.number,
                    "preview": item.question.content[:160],
                    "subject": self.chats.course_scope(item.subject, item.source_id)["subject"],
                    "saved": item.id in saved,
                }
            )
        return values

    def _selection_context(self, items: list[Draft]) -> list[dict]:
        attempts = self.store.attempt_records()
        previous = self.chats.conversations("question")
        saved = {item.id for item in self.store.items()}
        selected = []
        for item in items:
            source = self.store.source(item.source_id)
            unit_ids = {unit.id for unit in item.question.units()}
            historical = []
            for conversation in previous:
                original = conversation["context"].get("item", {})
                if (
                    original.get("id") != item.id
                    or original.get("question") != item.to_dict()["question"]
                ):
                    continue
                messages = self.chats.messages(conversation["id"])
                chat._require_live_memory_evidence(self.chats, messages)
                historical.append(
                    {
                        "conversation_id": conversation["id"],
                        "messages": [
                            {
                                key: message[key]
                                for key in ("role", "content", "status", "created_at")
                            }
                            for message in messages
                            if message["role"] == "user" or message["status"] == "complete"
                        ],
                    }
                )
            selected.append(
                {
                    "item": item.to_dict(),
                    "source": {"id": source.id, "name": source.name, "sha256": source.sha256},
                    "confirmed_material": item.id in saved,
                    "historical_question_dialogues": historical,
                    "actual_attempts": [
                    attempt for attempt in attempts
                    if attempt.get("identity_valid") is True
                    and attempt.get("source") == "user_attempt"
                    and attempt.get("item_id") == item.id
                    and attempt.get("unit_id") in unit_ids
                    and any(isinstance(attempt.get(field), str) and attempt[field].strip()
                            for field in ("answer", "note"))
                    ],
                }
            )
        return selected

    def generate(
        self,
        item_ids: list[str],
        provider: CompletionProvider,
        *,
        requirements: str = "",
    ) -> dict:
        """Generate a new candidate with at most one explicit provider call."""
        requirements_limit = 16000 - len(REVIEW_REQUEST) - len("\n\n我本次的具体要求：\n")
        if len(requirements) > requirements_limit:
            raise StudyError(f"本次要求过长，请控制在 {requirements_limit} 字符以内。")
        items = self._selected(item_ids)
        selected = self._selection_context(items)
        subjects = {
            self.chats.course_scope(item.subject, item.source_id)["subject"] for item in items
        }
        subject = next(iter(subjects)) if len(subjects) == 1 else ""
        source_ids = list(dict.fromkeys(item.source_id for item in items))
        conversation_id = self.chats.open_review(subject, source_ids, {"review_items": selected})
        identity, timestamp = uuid4().hex, now()
        document = {
            "id": identity,
            "conversation_id": conversation_id,
            "item_ids": list(item_ids),
            "source_ids": source_ids,
            "subject": subject,
            "status": "generating",
            "raw_ai": "",
            "working": "",
            "initial_working": "",
            "working_revision": 0,
            "requirements": requirements,
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_saved_version_id": None,
            "error": "",
        }
        with self.chats.connect() as db:
            db.execute(
                "INSERT INTO ai_review_documents VALUES (?,?,?)",
                (identity, conversation_id, encode(document)),
            )
        request = REVIEW_REQUEST + (
            "\n\n我本次的具体要求：\n" + requirements if requirements else ""
        )
        try:
            chat.send(
                self.chats, conversation_id, request, uuid4().hex, provider, allow_remote=True
            )
        except StudyError as exc:
            with self.chats.connect() as db:
                current = self._load(db, identity)
                current.update(status="failed", error=str(exc), updated_at=now())
                self._write(db, current)
            raise
        return self.document(identity)

    @staticmethod
    def _load(db, identity: str) -> dict:
        row = db.execute(
            "SELECT payload FROM ai_review_documents WHERE id=?", (identity,)
        ).fetchone()
        if row is None:
            raise StudyError("复盘稿不存在。")
        return json.loads(row[0])

    @staticmethod
    def _write(db, document: dict) -> None:
        db.execute(
            "UPDATE ai_review_documents SET payload=? WHERE id=?",
            (encode(document), document["id"]),
        )

    def document(self, identity: str) -> dict:
        """Read a durable working draft; recover a locally saved result without calling a model."""
        with self.chats.connect() as db:
            document = self._load(db, identity)
        if document["status"] == "generating":
            messages = self.chats.messages(document["conversation_id"])
            answers = [message for message in messages if message["role"] == "assistant"]
            if answers:
                answer = answers[-1]
                raw = answer["metadata"].get("raw_response", "")
                evidence = [
                    ref for message in messages for ref in message["metadata"].get("evidence", [])
                ]
                with self.chats.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    document = self._load(db, identity)
                    if document["status"] == "generating":
                        initial = (
                            visible_answer(answer["content"], evidence)
                            if answer["status"] == "complete" else ""
                        )
                        if not document["working_revision"] and answer["status"] == "complete":
                            document["working"] = initial
                        document.update(
                            initial_working=initial,
                            raw_ai=raw,
                            status="ready" if answer["status"] == "complete" else "failed",
                            error=answer["content"] if answer["status"] == "error" else "",
                            updated_at=now(),
                        )
                        self._write(db, document)
        return document

    def documents(self, item_id: str | None = None, *, include_deleted: bool = False) -> list[dict]:
        with self.chats.connect() as db:
            rows = db.execute(
                "SELECT id,payload FROM ai_review_documents ORDER BY rowid DESC"
            ).fetchall()
        values = [
            self.document(row[0])
            for row in rows
            if item_id is None or item_id in json.loads(row[1])["item_ids"]
        ]
        return [value for value in values if include_deleted or not value.get("deleted_at")]

    def remember_view(self, identity: str, version_id: str | None = None) -> None:
        """Remember the local return destination without creating a saved version or an AI call."""
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = self._load(db, identity)
            if document.get("deleted_at"):
                raise StudyError("这份复盘已删除，请先恢复。")
            if version_id:
                row = db.execute("SELECT document_id FROM ai_review_versions WHERE id=?",
                                 (version_id,)).fetchone()
                if row is None or row[0] != identity:
                    raise StudyError("保存版本与这份复盘不匹配。")
            document.update(last_opened_at=now(), viewed_version_id=version_id)
            self._write(db, document)

    def resume_document(self, item_id: str | None = None) -> dict | None:
        """Recover the last viewed draft/version after navigation or service restart."""
        values = self.documents(item_id)
        live = {option["id"] for option in self.options()}
        values = [value for value in values if set(value["item_ids"]) <= live]
        return max(values, key=lambda value: (value.get("last_opened_at", ""),
                                             value["created_at"]), default=None)

    def delete_document(self, identity: str, *, confirmed: bool) -> None:
        """Soft-delete one manuscript, retaining its versions and provenance for recovery."""
        if not confirmed:
            raise StudyError("请确认删除所选复盘，删除后可恢复。")
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = self._load(db, identity)
            document["deleted_at"] = now()
            self._write(db, document)

    def restore_document(self, identity: str) -> None:
        """Restore only the selected manuscript; source-material deletion remains independent."""
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = self._load(db, identity)
            document.pop("deleted_at", None)
            self._write(db, document)

    @staticmethod
    def _content(content: str) -> None:
        if not isinstance(content, str) or len(content) > MAX_CONTENT:
            raise StudyError("复盘稿过长，请控制在 200000 字符以内。")

    def update_working(
        self,
        identity: str,
        content: str,
        *,
        expected_revision: int | None = None,
    ) -> dict:
        """Persist edits locally, refusing stale editor updates instead of overwriting them."""
        self._content(content)
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = self._load(db, identity)
            if expected_revision is not None and document["working_revision"] != expected_revision:
                raise StudyError("这份稿件已在其他页面更新；请保留当前输入并创建恢复副本。")
            if document["working"] != content:
                document.update(
                    working=content, working_revision=document["working_revision"] + 1,
                    updated_at=now(),
                )
                self._write(db, document)
        with self.chats.connect() as check:
            committed = self._load(check, identity)
        if (committed["working"] != content
                or committed["working_revision"] != document["working_revision"]):
            raise StudyError("复盘草稿保存后的正文或修订号不一致，请保留当前输入并重新核对。")
        return committed

    def fork_working(self, identity: str, content: str) -> dict:
        """Keep a conflicting edit as a separate recoverable draft, with no AI request."""
        self._content(content)
        original = self.document(identity)
        timestamp = now()
        document = {
            **original,
            "id": uuid4().hex,
            "working": content,
            "working_revision": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
            "recovery_of": identity,
            "last_saved_version_id": None,
        }
        document.pop("viewed_version_id", None)
        document.pop("deleted_at", None)
        with self.chats.connect() as db:
            db.execute(
                "INSERT INTO ai_review_documents VALUES (?,?,?)",
                (document["id"], document["conversation_id"], encode(document)),
            )
        return document

    def save_version(
        self, identity: str, *, expected_revision: int | None = None,
        expected_content: str | None = None,
    ) -> dict:
        """Save the exact acknowledged body once and independently verify the committed version."""
        with self.chats.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = self._load(db, identity)
            if document.get("deleted_at"):
                raise StudyError("这份复盘已删除，请先恢复。")
            if ((expected_revision is not None
                 and document["working_revision"] != expected_revision)
                    or (expected_content is not None
                        and document["working"] != expected_content)):
                raise StudyError("待保存的复盘正文已变化，请保留当前输入并重新核对。")
            if not document["working"].strip():
                raise StudyError("复盘稿为空，请先生成或填写内容。")
            previous = db.execute(
                "SELECT * FROM ai_review_versions WHERE id=? AND document_id=?",
                (document.get("last_saved_version_id"), identity),
            ).fetchone()
            if previous is not None and previous["content"] == document["working"]:
                version = dict(previous)
            else:
                number = db.execute(
                    "SELECT COALESCE(MAX(number),0)+1 FROM ai_review_versions WHERE document_id=?",
                    (identity,),
                ).fetchone()[0]
                version = {
                    "id": uuid4().hex,
                    "document_id": identity,
                    "number": number,
                    "content": document["working"],
                    "created_at": now(),
                }
                db.execute("INSERT INTO ai_review_versions VALUES (?,?,?,?,?)",
                           tuple(version.values()))
                document.update(last_saved_version_id=version["id"], updated_at=now())
                self._write(db, document)
        try:
            with sqlite3.connect(self.chats.db.as_uri() + "?mode=ro", uri=True) as check:
                check.row_factory = sqlite3.Row
                check.execute("PRAGMA query_only=ON")
                check.execute("BEGIN")
                committed_document = self._load(check, identity)
                committed_version = check.execute(
                    "SELECT * FROM ai_review_versions WHERE id=? AND document_id=?",
                    (version["id"], identity),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StudyError("复盘保存后的独立核对失败，输入已保留，请重新打开核对。") from exc
        if (committed_version is None or dict(committed_version) != version
                or committed_document["working"] != version["content"]
                or committed_document["working_revision"] != document["working_revision"]
                or committed_document.get("last_saved_version_id") != version["id"]):
            raise StudyError("复盘保存后的正文或版本不一致，输入已保留，请重新打开核对。")
        return dict(committed_version)

    def versions(self, identity: str) -> list[dict]:
        with self.chats.connect() as db:
            self._load(db, identity)
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,document_id,number,content,created_at FROM ai_review_versions "
                    "WHERE document_id=? ORDER BY number DESC",
                    (identity,),
                )
            ]

    def version(self, identity: str) -> dict:
        with self.chats.connect() as db:
            row = db.execute("SELECT * FROM ai_review_versions WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise StudyError("保存的复盘版本不存在。")
        return dict(row)

    def memory_candidate(self, version_id: str) -> str:
        """Use the selected saved human version through the existing memory confirmation path."""
        version = self.version(version_id)
        document = self.document(version["document_id"])
        if any(self.store.is_source_deleted(identity) for identity in document["source_ids"]):
            raise StudyError("来源材料已删除，请先恢复后再保存到记忆候选。")
        archive = self.chats.archive(document["conversation_id"], saved_review_version=version)
        return self.chats.add_candidate(archive, version["content"])
