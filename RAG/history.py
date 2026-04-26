"""
历史会话记录：将多轮对话持久化到本地 JSON，支持创建会话、追加消息、列出与删除会话。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 与 config_data.PROJECT_ROOT 一致：仓库根目录下的 chat_history.json（不 import config，避免拉起 API/embedding）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HISTORY_FILE = _PROJECT_ROOT / "chat_history.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _default_history_path() -> Path:
    return _DEFAULT_HISTORY_FILE


class ChatHistoryStore:
    """基于单个 JSON 文件的会话存储（适合个人/小规模使用）。"""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path is not None else _default_history_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {"sessions": []}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {"sessions": []}
            return
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or "sessions" not in raw:
            self._data = {"sessions": []}
            return
        self._data = raw

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def _sessions(self) -> list[dict[str, Any]]:
        return self._data.setdefault("sessions", [])

    def _find_index(self, session_id: str) -> int:
        for i, s in enumerate(self._sessions()):
            if s.get("id") == session_id:
                return i
        return -1

    def create_session(self, title: str | None = None) -> str:
        """新建会话，返回 session_id。新会话排在列表最前。"""
        sid = str(uuid.uuid4())
        now = _now_iso()
        session: dict[str, Any] = {
            "id": sid,
            "title": (title or "新对话").strip() or "新对话",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._sessions().insert(0, session)
        self._save()
        return sid

    def _maybe_set_title_from_first_user(self, session: dict[str, Any], content: str) -> None:
        if session.get("title") not in (None, "", "新对话"):
            return
        first = (content or "").strip().split("\n")[0].strip()
        if not first:
            return
        session["title"] = first[:80] + ("…" if len(first) > 80 else "")

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息；role 建议为 user / assistant。"""
        idx = self._find_index(session_id)
        if idx < 0:
            raise ValueError(f"会话不存在: {session_id}")
        session = self._sessions()[idx]
        msg = {"role": role, "content": content, "ts": _now_iso()}
        session.setdefault("messages", []).append(msg)
        session["updated_at"] = msg["ts"]
        if role == "user":
            self._maybe_set_title_from_first_user(session, content)
        self._save()

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        """追加一轮问答（先 user 后 assistant）。"""
        self.append_message(session_id, "user", question)
        self.append_message(session_id, "assistant", answer)

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        """返回会话摘要列表（含 id、title、时间、消息条数），按更新时间新到旧。"""
        out: list[dict[str, Any]] = []
        for s in self._sessions()[:limit]:
            msgs = s.get("messages") or []
            out.append(
                {
                    "id": s.get("id"),
                    "title": s.get("title", "新对话"),
                    "created_at": s.get("created_at"),
                    "updated_at": s.get("updated_at"),
                    "message_count": len(msgs),
                }
            )
        return out

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """返回完整会话（含 messages），不存在则 None。"""
        idx = self._find_index(session_id)
        if idx < 0:
            return None
        return json.loads(json.dumps(self._sessions()[idx], ensure_ascii=False))

    def delete_session(self, session_id: str) -> bool:
        idx = self._find_index(session_id)
        if idx < 0:
            return False
        del self._sessions()[idx]
        self._save()
        return True

    def rename_session(self, session_id: str, title: str) -> bool:
        idx = self._find_index(session_id)
        if idx < 0:
            return False
        self._sessions()[idx]["title"] = (title or "新对话").strip() or "新对话"
        self._sessions()[idx]["updated_at"] = _now_iso()
        self._save()
        return True
