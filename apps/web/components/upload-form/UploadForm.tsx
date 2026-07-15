"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { JobLivePanel } from "@/components/job/JobLivePanel";
import { API_BASE, type GuitarPart } from "@/lib/tab-format";

type SourceMode = "upload" | "youtube";

async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("API не отвечает. Запустите сервер на http://localhost:8000");
    }
    throw new Error("Не удалось связаться с API (http://localhost:8000)");
  } finally {
    window.clearTimeout(timeout);
  }
}

export function UploadForm() {
  const router = useRouter();
  const [mode, setMode] = useState<SourceMode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [guitarPart, setGuitarPart] = useState<GuitarPart>("combined");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [draftId, setDraftId] = useState<string | null>(null);

  const resetForm = useCallback(() => {
    setSubmitting(false);
    setProcessing(false);
    setError(null);
    setJobId(null);
    setDraftId(null);
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (processing) return;

    setSubmitting(true);
    setError(null);
    setDraftId(null);
    setJobId(null);

    try {
      if (mode === "upload") {
        if (!file) throw new Error("Выберите файл");
        const form = new FormData();
        form.append("file", file);
        form.append("guitar_part", guitarPart);
        const res = await apiFetch(`${API_BASE}/v1/jobs`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setJobId(data.job.id);
        setProcessing(true);
      } else {
        if (!youtubeUrl.trim()) throw new Error("Введите YouTube URL");
        if (!rightsConfirmed) throw new Error("Подтвердите права на источник");
        const res = await apiFetch(`${API_BASE}/v1/jobs/youtube`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            youtube_url: youtubeUrl,
            guitar_part: guitarPart,
            rights_confirmed: true,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setJobId(data.job.id);
        setProcessing(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setProcessing(false);
    } finally {
      setSubmitting(false);
    }
  }

  const handleJobDone = useCallback((id: string) => {
    setDraftId(id);
    setProcessing(false);
  }, []);

  const handleJobFailed = useCallback((message: string) => {
    setError(message);
    setProcessing(false);
  }, []);

  const formLocked = processing && !draftId;

  return (
    <>
      <form onSubmit={onSubmit} className="card">
        <div className="field">
          <label>Источник</label>
          <select value={mode} onChange={(e) => setMode(e.target.value as SourceMode)} disabled={formLocked}>
            <option value="upload">Upload MP3/WAV</option>
            <option value="youtube">YouTube URL</option>
          </select>
        </div>

        <div className="field">
          <label>Какую гитару транскрибировать</label>
          <select
            value={guitarPart}
            onChange={(e) => setGuitarPart(e.target.value as GuitarPart)}
            disabled={formLocked}
          >
            <option value="combined">Все гитары (Demucs stem)</option>
            <option value="solo">Соло-гитара</option>
            <option value="rhythm">Ритм-гитара (Ensemble → CASA demix)</option>
          </select>
          <p className="muted" style={{ marginTop: "0.35rem", fontSize: "0.875rem" }}>
            Coarse: RoFormer (vocals) → Demucs (bass/drums/guitar) или fallback Demucs.
            Demix: Wave-U-Net → CASA. Настройка в pipeline.yaml.
          </p>
        </div>

        {mode === "upload" ? (
          <div className="field">
            <label>Аудиофайл</label>
            <input
              type="file"
              accept=".mp3,.wav,audio/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              disabled={formLocked}
            />
          </div>
        ) : (
          <>
            <div className="field">
              <label>YouTube URL</label>
              <input
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                placeholder="https://youtube.com/watch?v=..."
                disabled={formLocked}
              />
            </div>
            <div className="field">
              <label>
                <input
                  type="checkbox"
                  checked={rightsConfirmed}
                  onChange={(e) => setRightsConfirmed(e.target.checked)}
                  disabled={formLocked}
                />{" "}
                Подтверждаю права на использование источника (local-only ingest)
              </label>
            </div>
          </>
        )}

        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="submit" disabled={submitting || formLocked}>
            {submitting ? "Отправка…" : processing ? "Обработка…" : draftId ? "Готово" : "Создать таб"}
          </button>
          {jobId ? (
            <button type="button" className="secondary" onClick={resetForm} disabled={submitting}>
              Новый таб
            </button>
          ) : null}
        </div>
        {error && <p style={{ color: "var(--danger)", marginTop: "1rem" }}>{error}</p>}
      </form>

      {jobId ? (
        <JobLivePanel jobId={jobId} onDone={handleJobDone} onFailed={handleJobFailed} />
      ) : null}

      {draftId ? (
        <div style={{ marginTop: "1rem" }}>
          <button type="button" onClick={() => router.push(`/drafts/${draftId}`)}>
            Открыть результат
          </button>
        </div>
      ) : null}
    </>
  );
}
