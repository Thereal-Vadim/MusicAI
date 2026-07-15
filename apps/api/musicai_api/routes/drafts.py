"""Draft API routes."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from musicai_api.db.models import Draft
from musicai_api.db.session import get_session
from tab_schema.models import EditRecord, TabDocument, TabNote

router = APIRouter(prefix="/v1/drafts", tags=["drafts"])


class DraftResponse(BaseModel):
    id: str
    job_id: str
    document: TabDocument


class NotePatch(BaseModel):
    string: int | None = None
    fret: int | None = None
    pitch: str | None = None


@router.get("/{draft_id}", response_model=DraftResponse)
async def get_draft(draft_id: str, session: AsyncSession = Depends(get_session)) -> DraftResponse:
    draft = await session.get(Draft, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    document = TabDocument.model_validate_json(draft.document_json)
    return DraftResponse(id=draft.id, job_id=draft.job_id, document=document)


@router.patch("/{draft_id}/notes/{note_id}", response_model=DraftResponse)
async def patch_note(
    draft_id: str,
    note_id: str,
    patch: NotePatch,
    session: AsyncSession = Depends(get_session),
) -> DraftResponse:
    draft = await session.get(Draft, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    document = TabDocument.model_validate_json(draft.document_json)
    target: TabNote | None = None
    for note in document.all_notes():
        if note.id == note_id:
            target = note
            break
    if not target:
        raise HTTPException(status_code=404, detail="Note not found")

    timestamp = datetime.now(timezone.utc).isoformat()
    for field_name, value in patch.model_dump(exclude_none=True).items():
        old_value = getattr(target, field_name)
        setattr(target, field_name, value)
        document.edit_history.append(
            EditRecord(
                timestamp=timestamp,
                note_id=note_id,
                field=field_name,
                old_value=old_value,
                new_value=value,
            )
        )
        if field_name in {"string", "fret"}:
            target.flags = [f for f in target.flags if f != "manual_edit"]
            target.flags.append("manual_edit")

    draft.document_json = document.model_dump_json()
    await session.commit()
    return DraftResponse(id=draft.id, job_id=draft.job_id, document=document)


@router.get("/by-job/{job_id}", response_model=DraftResponse)
async def get_draft_by_job(job_id: str, session: AsyncSession = Depends(get_session)) -> DraftResponse:
    draft = await session.scalar(select(Draft).where(Draft.job_id == job_id))
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    document = TabDocument.model_validate_json(draft.document_json)
    return DraftResponse(id=draft.id, job_id=draft.job_id, document=document)
