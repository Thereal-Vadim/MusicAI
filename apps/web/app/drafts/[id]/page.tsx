"use client";

import { useCallback, useEffect, useState } from "react";
import { TabEditor } from "@/components/tab/TabEditor";
import { API_BASE, type TabDocument } from "@/lib/tab-format";

export default function DraftPage({ params }: { params: Promise<{ id: string }> }) {
  const [draftId, setDraftId] = useState<string | null>(null);
  const [document, setDocument] = useState<TabDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    params.then((p) => setDraftId(p.id));
  }, [params]);

  useEffect(() => {
    if (!draftId) return;
    fetch(`${API_BASE}/v1/drafts/${draftId}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => setDocument(data.document))
      .catch((err) => setError(err.message));
  }, [draftId]);

  const onEditNote = useCallback(
    async (noteId: string, patch: { string?: number; fret?: number }) => {
      if (!draftId) return;
      const res = await fetch(`${API_BASE}/v1/drafts/${draftId}/notes/${noteId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setDocument(data.document);
    },
    [draftId]
  );

  if (error) return <main><p style={{ color: "var(--danger)" }}>{error}</p></main>;
  if (!document) return <main><p className="muted">Загрузка draft…</p></main>;

  return (
    <main>
      <h1>Draft табулатуры</h1>
      <p className="muted">
        {document.meta.key} {document.meta.mode} · {document.meta.bpm.toFixed(0)} BPM · confidence{" "}
        {(document.meta.overall_confidence * 100).toFixed(0)}%
      </p>
      <div style={{ marginTop: "1.5rem" }}>
        <TabEditor document={document} onEditNote={onEditNote} />
      </div>
    </main>
  );
}
