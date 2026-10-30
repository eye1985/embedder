"""The conversation endpoints: listing, reading, renaming, deletion.

Writing is not here -- a turn is stored by the request that answered it, on
the chat path. These are the requests around that, every one scoped to the
current user, so another user's conversation is reported missing rather than
refused.
"""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.config import CURRENT_USER_ID
from api.schemas import (
    ConversationDetail,
    ConversationOut,
    DeleteConversationResult,
    RenameResult,
)
from api.services.conversations import list_conversations, read_conversation
from db.pool import connection
from db.tables.chat import delete_conversation, rename_conversation

router = APIRouter()


class Rename(BaseModel):
    title: str


@router.get("/conversations", response_model=list[ConversationOut])
def conversations():
    """The user's conversations, most recently used first."""
    with connection() as conn:
        return list_conversations(conn, CURRENT_USER_ID)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def conversation(conversation_id: uuid.UUID):
    """One conversation's transcript, with each answer's sources resolved."""
    with connection() as conn:
        found = read_conversation(conn, CURRENT_USER_ID, conversation_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return found


@router.patch("/conversations/{conversation_id}", response_model=RenameResult)
def rename(conversation_id: uuid.UUID, body: Rename):
    """Rename a conversation.

    The name starts as the question that opened the conversation, which is a
    description of the first turn and not always of the conversation it became.
    """
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="A title cannot be empty.")

    with connection() as conn:
        if not rename_conversation(conn, CURRENT_USER_ID, conversation_id, title):
            raise HTTPException(status_code=404, detail="No such conversation.")
    return {"conversation_id": conversation_id, "title": title}


@router.delete(
    "/conversations/{conversation_id}", response_model=DeleteConversationResult
)
def remove(conversation_id: uuid.UUID):
    """Delete a conversation and its messages.

    No document is touched. The citations inside a conversation are plain ids
    rather than references, so deleting the conversation that holds them is
    not a reason to disturb what they pointed at.
    """
    with connection() as conn:
        deleted = delete_conversation(conn, CURRENT_USER_ID, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return {"conversation_id": conversation_id, "deleted": True}
