"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE, PIPELINE_STAGES } from "@/lib/tab-format";

type SourceMode = "upload" | "youtube";

export function UploadForm() {
  const router = useRouter();
  const [mode, setMode] = useState<SourceMode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState("queued");

  async function pollJob(id: string) {
    const interval = setInterval(async () => {
      const res = await fetch(`${API_BASE}/v1/jobs/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      setStage(data.stage);
      if (data.status === "done" && data.draft_id) {
        clearInterval(interval);
        router.push(`/drafts/${data.draft_id}`);
      }
      if (data.status === "failed") {
        clearInterval(interval);
        setError(data.stage_detail || "Job failed");
        setLoading(false);
      }
    }, 1500);
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (mode === "upload") {
        if (!file) throw new Error("Выберите файл");
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(`${API_BASE}/v1/jobs`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setJobId(data.job.id);
        pollJob(data.job.id);
      } else {
        if (!rightsConfirmed) throw new Error("Подтвердите права на источник");
        const res = await fetch(`${API_BASE}/v1/jobs/youtube`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ youtube_url: youtubeUrl, rights_confirmed: true }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setJobId(data.job.id);
        pollJob(data.job.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setLoading(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="card">
      <div className="field">
        <label>Источник</label>
        <select value={mode} onChange={(e) => setMode(e.target.value as SourceMode)}>
          <option value="upload">Upload MP3/WAV</option>
          <option value="youtube">YouTube URL</option>
        </select>
      </div>

      {mode === "upload" ? (
        <div className="field">
          <label>Аудиофайл</label>
          <input type="file" accept=".mp3,.wav,audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
      ) : (
        <>
          <div className="field">
            <label>YouTube URL</label>
            <input value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)} placeholder="https://youtube.com/watch?v=..." />
          </div>
          <div className="field">
            <label>
              <input type="checkbox" checked={rightsConfirmed} onChange={(e) => setRightsConfirmed(e.target.checked)} />{" "}
              Подтверждаю права на использование источника (local-only ingest)
            </label>
          </div>
        </>
      )}

      <button type="submit" disabled={loading}>
        {loading ? "Обработка..." : "Создать таб"}
      </button>
      {error && <p style={{ color: "var(--danger)", marginTop: "1rem" }}>{error}</p>}

      {jobId && (
        <div className="stage-list">
          {PIPELINE_STAGES.map((s) => {
            const idx = PIPELINE_STAGES.indexOf(stage as (typeof PIPELINE_STAGES)[number]);
            const currentIdx = PIPELINE_STAGES.indexOf(s);
            const cls =
              stage === s ? "stage-item active" : currentIdx < idx || stage === "done" ? "stage-item done" : "stage-item pending";
            return (
              <div key={s} className={cls}>
                <span>{s}</span>
                <span>{stage === s ? "…" : currentIdx < idx || stage === "done" ? "✓" : ""}</span>
              </div>
            );
          })}
        </div>
      )}
    </form>
  );
}
