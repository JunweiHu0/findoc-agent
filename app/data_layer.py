"""Custom Chainlit DataLayer bridging to FinDoc backend SQLite storage.

Provides left-sidebar conversation history by implementing Chainlit's
BaseDataLayer. Threads = conversations. Steps = messages.

Wire in: set this as Chainlit's data layer in chainlit_app.py via
`cl.data._data_layer = FinDocDataLayer()`.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chainlit.data.base import BaseDataLayer
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

from backend import storage


def _conv_exists(conv_id: str) -> bool:
    return storage.get_conversation(conv_id) is not None


def _resolve_page_path(doc_id: str, page_num: int) -> str | None:
    from agent.config import PAGES_DIR
    candidate = PAGES_DIR / doc_id / f"p{page_num:03d}.png"
    return str(candidate) if candidate.exists() else None


class FinDocDataLayer(BaseDataLayer):
    """Data layer that persists threads/steps to the FinDoc backend SQLite."""

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        return PersistedUser(
            id=identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
            identifier=identifier,
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        return PersistedUser(
            id=user.identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
            identifier=user.identifier,
        )

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        convs = storage.list_conversations()
        threads: list[ThreadDict] = []
        for c in convs:
            threads.append(ThreadDict(
                id=c["id"],
                createdAt=datetime.fromtimestamp(c["created_at"], tz=timezone.utc).isoformat(),
                name=c["title"] or c["id"],
                userId=filters.userId,
                userIdentifier=filters.userId,
                tags=None,
                metadata=None,
                steps=[],
                elements=None,
            ))
        return PaginatedResponse(
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
            data=threads,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        conv = storage.get_conversation(thread_id)
        if not conv:
            return None

        steps: list[StepDict] = []
        elements: list[ElementDict] = []

        for m in conv.get("messages", []):
            step_id = m["id"]
            created = datetime.fromtimestamp(m["created_at"], tz=timezone.utc).isoformat()
            step_type = "user_message" if m["role"] == "user" else "assistant_message"

            step: StepDict = {
                "id": step_id,
                "name": "User" if m["role"] == "user" else "Assistant",
                "type": step_type,  # type: ignore
                "threadId": thread_id,
                "parentId": None,
                "streaming": False,
                "waitForAnswer": False,
                "isError": False,
                "metadata": {},
                "tags": None,
                "input": m["content"] if m["role"] == "user" else "",
                "output": m["content"] if m["role"] == "assistant" else "",
                "createdAt": created,
                "start": None,
                "end": None,
                "generation": None,
                "showInput": "json",
                "defaultOpen": False,
                "autoCollapse": False,
                "language": None,
                "icon": None,
            }
            steps.append(step)

            # Build image elements from citation pages
            for page in m.get("pages", []):
                doc_id = page.get("doc_id", "")
                page_num = page.get("page_num", 0)
                path = _resolve_page_path(doc_id, page_num)
                if path:
                    elem: ElementDict = {
                        "id": str(uuid.uuid4()),
                        "threadId": thread_id,
                        "type": "image",
                        "chainlitKey": None,
                        "path": path,
                        "url": None,
                        "objectKey": None,
                        "name": f"{doc_id} p.{page_num}",
                        "display": "inline",
                        "size": "medium",
                        "language": None,
                        "page": None,
                        "props": None,
                        "autoPlay": None,
                        "playerConfig": None,
                        "forId": step_id,
                        "mime": "image/png",
                    }
                    elements.append(elem)

        return ThreadDict(
            id=conv["id"],
            createdAt=datetime.fromtimestamp(conv["created_at"], tz=timezone.utc).isoformat(),
            name=conv["title"],
            userId=None,
            userIdentifier=None,
            tags=None,
            metadata=None,
            steps=steps,
            elements=elements if elements else None,
        )

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        if name is not None:
            storage.update_conversation_title(thread_id, name)

    async def delete_thread(self, thread_id: str):
        storage.delete_conversation(thread_id)

    async def create_step(self, step_dict: StepDict):
        """No-op. Backend /api/v1/query is the single writer for conversations
        and messages — keeps titles + citations + pages aligned in one place."""
        return

    async def create_element(self, element: "Element"):
        pass

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass

    async def delete_feedback(self, feedback_id: str) -> bool:
        return False

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return ""

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        return None

    async def set_step_favorite(self, step_id: str, favorite: bool) -> bool:
        return False

    async def get_favorite_steps(self, thread_id: str) -> List[StepDict]:
        return []

    async def delete_step(self, step_id: str):
        pass

    async def update_step(self, step_dict: StepDict):
        pass

    async def build_debug_url(self, thread_id: str) -> str:
        return ""

    async def get_thread_author(self, thread_id: str) -> str:
        return ""

    async def close(self):
        pass