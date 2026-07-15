"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { StemAudioPanel } from "@/components/job/StemPreviewPanel";
import { TabEditor } from "@/components/tab/TabEditor";
import { API_BASE, type TabDocument } from "@/lib/tab-format";
import "./draft-page.css";

const AlphaTabViewer = dynamic(
  () => import("@/components/tab/AlphaTabViewer").then((m) => m.AlphaTabViewer),
  { ssr: false, loading: () => <p className="muted">Загрузка табулатуры…</p> }
);

type ViewMode = "tab" | "conflicts" | "audio";

export default function DraftPage({ params }: { params: Promise<{ id: string }> }) {
  const [draftId, setDraftId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [document, setDocument] = useState<TabDocument | null>(null);
  const [alphaTex, setAlphaTex] = useState<string>("");
  const [view, setView] = useState<ViewMode>("tab");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    params.then((p) => setDraftId(p.id));
  }, [params]);

  useEffect(() => {
    if (!draftId) return;
    Promise.all([
      fetch(`${API_BASE}/v1/drafts/${draftId}`).then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      }),
      fetch(`${API_BASE}/v1/drafts/${draftId}/alphatex`).then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.text();
      }),
    ])
      .then(([draft, tex]) => {
        setDocument(draft.document);
        setJobId(draft.job_id);
        setAlphaTex(tex);
      })
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
      const texRes = await fetch(`${API_BASE}/v1/drafts/${draftId}/alphatex`);
      if (texRes.ok) setAlphaTex(await texRes.text());
    },
    [draftId]
  );

  if (error) {
    return (
      <div className="draft-page">
        <main>
          <p style={{ color: "var(--danger)" }}>{error}</p>
        </main>
      </div>
    );
  }

  if (!document || !draftId) {
    return (
      <div className="draft-page">
        <main>
          <p className="muted">Загрузка draft…</p>
        </main>
      </div>
    );
  }

  const title = document.meta.title ?? "Draft";
  const artist = document.meta.artist ?? "MusicAI";
  const trackName = document.tracks[0]?.name ?? "Guitar";
  const guitarPartLabel =
    document.meta.guitar_part === "solo"
      ? "соло-гитара"
      : document.meta.guitar_part === "rhythm"
        ? "ритм-гитара"
        : null;
  const quality = document.meta.quality;

  return (
    <div className="draft-page">
      <main>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
          <div>
            <h1 style={{ marginBottom: 0 }}>{title}</h1>
            <p className="muted" style={{ marginTop: "0.25rem" }}>
              {artist} · {document.meta.key} {document.meta.mode} · {document.meta.bpm.toFixed(0)} BPM ·
              средняя уверенность {(document.meta.overall_confidence * 100).toFixed(0)}%
              {guitarPartLabel ? <> · {guitarPartLabel}</> : null}
            </p>
            {quality ? (
              <p className="muted" style={{ marginTop: "0.35rem", fontSize: "0.9rem" }}>
                ≥95%: {(quality.high_confidence_pct * 100).toFixed(0)}% · исправлено судьёй:{" "}
                {(quality.snapped_pct * 100).toFixed(0)}%
                {quality.reference_url ? (
                  <>
                    {" "}
                    · Songsterr:{" "}
                    {quality.reference_match_pct != null
                      ? `${(quality.reference_match_pct * 100).toFixed(0)}% совпадение`
                      : "—"}
                    {quality.reference_mismatch_count
                      ? ` (${quality.reference_mismatch_count} расхождений)`
                      : ""}
                  </>
                ) : null}
              </p>
            ) : null}
          </div>
          <div className="toolbar">
            <button
              type="button"
              className={view === "tab" ? "active" : undefined}
              onClick={() => setView("tab")}
            >
              Табулатура
            </button>
            <button
              type="button"
              className={view === "conflicts" ? "active" : undefined}
              onClick={() => setView("conflicts")}
            >
              Конфликты
            </button>
            <button
              type="button"
              className={view === "audio" ? "active" : undefined}
              onClick={() => setView("audio")}
            >
              Аудио
            </button>
            <a
              className="download"
              href={`${API_BASE}/v1/drafts/${draftId}/gp5`}
              download={`${title.replace(/\s+/g, "_")}.gp5`}
            >
              Download .gp5
            </a>
          </div>
        </div>

        <div style={{ marginTop: "1.25rem" }}>
          {view === "tab" ? (
            <AlphaTabViewer
              draftId={draftId}
              tex={alphaTex}
              gp5Url={`${API_BASE}/v1/drafts/${draftId}/gp5`}
              trackName={trackName}
            />
          ) : view === "conflicts" ? (
            <div className="editor-panel">
              <TabEditor document={document} onEditNote={onEditNote} />
            </div>
          ) : jobId ? (
            <StemAudioPanel jobId={jobId} variant="page" />
          ) : (
            <p className="muted">Job ID не найден — аудио недоступно.</p>
          )}
        </div>
      </main>
    </div>
  );
}
