"use client";

import { useEffect, useState } from "react";
import { API_BASE, type JobStemsInfo, type JobStatus } from "@/lib/tab-format";

export function useJobStems(jobId: string, poll = false) {
  const [stems, setStems] = useState<JobStemsInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function load() {
      try {
        const res = await fetch(`${API_BASE}/v1/jobs/${jobId}/stems`);
        if (res.ok) {
          const data: JobStemsInfo = await res.json();
          if (!cancelled) {
            setStems(data);
            setError(null);
            setLoading(false);
          }
          return true;
        }

        if (!cancelled) {
          if (res.status === 404 && poll) {
            setError(null);
          } else {
            const jobRes = await fetch(`${API_BASE}/v1/jobs/${jobId}`);
            if (jobRes.ok) {
              const job: JobStatus = await jobRes.json();
              if (job.status === "failed") {
                setError(`Job завершился с ошибкой: ${job.stage_detail || "unknown"}`);
              } else if (
                job.stage === "ingest" ||
                job.stage === "separate" ||
                job.stage === "guitar_demix" ||
                job.stage === "demix_validate" ||
                job.stage === "audio_cleanup" ||
                job.stage === "timbre_classify"
              ) {
                setError(null);
              } else {
                setError("Файлы stems не найдены на сервере для этого job.");
              }
            } else if (res.status === 404) {
              setError("Stems ещё не готовы или job не содержит аудиофайлов.");
            } else {
              setError(`API вернул ошибку ${res.status}`);
            }
          }
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setError("API не отвечает. Запустите сервер: ./scripts/start-api.sh");
          setLoading(false);
        }
      }
      return false;
    }

    setLoading(true);
    setError(null);
    void load().then((ready) => {
      if (cancelled || ready || !poll) return;
      timer = window.setInterval(async () => {
        const ok = await load();
        if (ok && timer) window.clearInterval(timer);
      }, 2000);
    });

    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [jobId, poll]);

  return { stems, loading, error };
}

interface StemAudioPanelProps {
  jobId: string;
  poll?: boolean;
  variant?: "inline" | "page";
}

export function StemAudioPanel({ jobId, poll = false, variant = "page" }: StemAudioPanelProps) {
  const { stems, loading, error } = useJobStems(jobId, poll);
  const rootClass = variant === "inline" ? "stem-preview" : "stem-audio-page";

  if (loading && !stems && !error) {
    return (
      <div className={rootClass}>
        <p className="muted">{poll ? "Ожидание изолированной гитары…" : "Загрузка аудио…"}</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={rootClass}>
        <p style={{ color: "var(--danger, #e03131)", margin: 0 }}>{error}</p>
        <p className="muted" style={{ marginTop: "0.5rem", fontSize: "0.9rem" }}>
          Если job завершился успешно, но аудио нет — возможно ingest/separate не сохранили файлы (пустой
          input.wav или ошибка YouTube). Создайте новый job и проверьте логи на этапе separate.
        </p>
      </div>
    );
  }

  if (!stems?.items.length) {
    return (
      <div className={rootClass}>
        <p className="muted">Stems недоступны. Создайте новый job — файлы появятся после этапа separate.</p>
      </div>
    );
  }

  return (
    <div className={rootClass}>
      {variant === "page" ? (
        <div className="stem-audio-page-header">
          <h2 style={{ margin: 0, fontSize: "1.1rem" }}>Прослушать и скачать изоляцию</h2>
          <p className="muted" style={{ margin: "0.35rem 0 0" }}>
            Сравните исходный микс, Demucs stem и финальный вариант — так проще заметить артефакты алгоритма.
          </p>
        </div>
      ) : (
        <div className="stem-preview-header">
          <strong>Прослушать изоляцию</strong>
          <span className="muted">Сравните варианты и проверьте артефакты</span>
        </div>
      )}

      <div className="stem-preview-list">
        {stems.items.map((item) => {
          const audioUrl = `${API_BASE}/v1/jobs/${jobId}/stems/audio/${item.id}`;
          return (
            <div key={item.id} className="stem-preview-item">
              <div className="stem-preview-item-head">
                <label htmlFor={`stem-audio-${jobId}-${item.id}`}>{item.label}</label>
                <a className="stem-download-link" href={audioUrl} download={item.filename}>
                  Скачать WAV
                </a>
              </div>
              <audio
                id={`stem-audio-${jobId}-${item.id}`}
                controls
                preload="metadata"
                src={audioUrl}
              />
              <span className="muted stem-preview-filename">{item.filename}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** @deprecated Use StemAudioPanel */
export function StemPreviewPanel(props: StemAudioPanelProps) {
  return <StemAudioPanel {...props} variant="inline" />;
}
