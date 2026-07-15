"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { StemPreviewPanel } from "@/components/job/StemPreviewPanel";
import { API_BASE, type JobStatus, type LiveLogLine } from "@/lib/tab-format";
import "./job-live.css";

function formatDuration(sec: number | null | undefined): string {
  if (sec == null || Number.isNaN(sec)) return "0:00";
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

interface JobLivePanelProps {
  jobId: string;
  onDone?: (draftId: string) => void;
  onFailed?: (message: string) => void;
}

export function JobLivePanel({ jobId, onDone, onFailed }: JobLivePanelProps) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [logs, setLogs] = useState<LiveLogLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const logOffsetRef = useRef(0);
  const doneRef = useRef(false);
  const failedRef = useRef(false);
  const onDoneRef = useRef(onDone);
  const onFailedRef = useRef(onFailed);

  useEffect(() => {
    onDoneRef.current = onDone;
    onFailedRef.current = onFailed;
  }, [onDone, onFailed]);

  useEffect(() => {
    let cancelled = false;
    doneRef.current = false;
    failedRef.current = false;
    logOffsetRef.current = 0;
    setLogs([]);
    setError(null);

    async function pollStatus() {
      const res = await fetch(`${API_BASE}/v1/jobs/${jobId}`);
      if (!res.ok || cancelled) return;
      const data: JobStatus = await res.json();
      setJob(data);
      if (data.status === "failed") {
        const message = data.stage_detail || "Job failed";
        setError(message);
        if (onFailedRef.current && !failedRef.current) {
          failedRef.current = true;
          onFailedRef.current(message);
        }
      }
      if (data.status === "done" && data.draft_id && onDoneRef.current && !doneRef.current) {
        doneRef.current = true;
        onDoneRef.current(data.draft_id);
      }
    }

    async function pollLogs() {
      const res = await fetch(`${API_BASE}/v1/jobs/${jobId}/logs?from_line=${logOffsetRef.current}`);
      if (!res.ok || cancelled) return;
      const data = await res.json();
      if (data.lines?.length) {
        setLogs((prev) => [...prev, ...data.lines]);
        logOffsetRef.current = data.total_lines;
      }
    }

    pollStatus();
    pollLogs();
    const statusTimer = window.setInterval(pollStatus, 1000);
    const logTimer = window.setInterval(pollLogs, 600);

    return () => {
      cancelled = true;
      window.clearInterval(statusTimer);
      window.clearInterval(logTimer);
    };
  }, [jobId]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  const progress = job?.progress_pct ?? 0;
  const isDone = job?.status === "done";
  const isFailed = job?.status === "failed";
  const separateDone =
    isDone ||
    (job?.stages ?? []).some((s) =>
      ["separate", "guitar_demix", "demix_validate", "audio_cleanup", "timbre_classify"].includes(s.name) &&
      s.duration_sec != null
    ) ||
    ["transcribe", "vision", "fusion", "judge", "draft", "done"].includes(job?.stage ?? "");

  return (
    <div className="job-live">
      <div className="job-live-header">
        <div>
          <strong>Job {jobId.slice(0, 8)}…</strong>
          <span className="muted" style={{ marginLeft: "0.75rem" }}>
            {isDone ? "Завершено" : isFailed ? "Ошибка" : job?.stage ?? "queued"}
          </span>
        </div>
        <div className="job-live-timer">
          <span>{progress}%</span>
          <span className="muted">·</span>
          <span>{formatDuration(job?.elapsed_sec)}</span>
        </div>
      </div>

      <div className="job-progress-track">
        <div className="job-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      {job?.stage_detail ? <p className="muted job-live-detail">{job.stage_detail}</p> : null}

      {separateDone && !isFailed ? <StemPreviewPanel jobId={jobId} poll={!isDone} /> : null}

      <div className="stage-list job-stage-list">
        {(job?.stages ?? []).map((stage) => {
          const currentIdx = (job?.stages ?? []).findIndex((s) => s.name === job?.stage);
          const stageIdx = (job?.stages ?? []).findIndex((s) => s.name === stage.name);
          const cls =
            job?.stage === stage.name
              ? "stage-item active"
              : stageIdx < currentIdx || isDone
                ? "stage-item done"
                : "stage-item pending";
          return (
            <div key={stage.name} className={cls}>
              <span>
                {stage.name}{" "}
                <span className="stage-pct">{stage.target_pct}%</span>
              </span>
              <span>
                {stage.duration_sec != null ? `${stage.duration_sec.toFixed(1)}s` : job?.stage === stage.name ? "…" : ""}
              </span>
            </div>
          );
        })}
      </div>

      <div className="job-log-panel" ref={logRef}>
        {logs.length === 0 ? <p className="muted">Ожидание логов…</p> : null}
        {logs.map((line, idx) => (
          <div key={`${line.ts}-${idx}`} className={`job-log-line level-${(line.level ?? "INFO").toLowerCase()}`}>
            <span className="job-log-ts">{line.ts ? line.ts.slice(11, 19) : "--:--:--"}</span>
            <span className="job-log-stage">[{line.stage ?? "-"}]</span>
            <span className="job-log-msg">{line.message}</span>
          </div>
        ))}
      </div>

      {isDone && job?.elapsed_sec != null ? (
        <div className="job-complete-banner">
          Обработка завершена за <strong>{formatDuration(job.elapsed_sec)}</strong>
          {job.draft_id ? (
            <>
              {" "}
              —{" "}
              <Link href={`/drafts/${job.draft_id}`}>Открыть таб</Link>
            </>
          ) : null}
        </div>
      ) : null}

      {error ? <p className="job-error">{error}</p> : null}
    </div>
  );
}
