"use client";

import {
  BENCHMARK_STATUS_COLORS,
  BENCHMARK_STATUS_LABELS,
  type BenchmarkComparison,
  type NoteMatchStatus,
} from "@/lib/benchmark-format";
import "./benchmark-panel.css";

interface BenchmarkComparePanelProps {
  comparison: BenchmarkComparison;
}

function pct(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

export function BenchmarkComparePanel({ comparison }: BenchmarkComparePanelProps) {
  const m = comparison.metrics;
  const alignments = m.alignments ?? [];

  const statusCounts = alignments.reduce<Record<string, number>>((acc, row) => {
    acc[row.status] = (acc[row.status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="benchmark-panel">
      <div className="benchmark-metrics">
        <div className="benchmark-metric">
          <span className="benchmark-metric-label">Pitch F1</span>
          <strong>{pct(m.pitch_f1)}</strong>
        </div>
        <div className="benchmark-metric">
          <span className="benchmark-metric-label">Pitch accuracy</span>
          <strong>{pct(m.pitch_accuracy)}</strong>
        </div>
        <div className="benchmark-metric">
          <span className="benchmark-metric-label">Fret accuracy</span>
          <strong>{pct(m.fret_accuracy)}</strong>
        </div>
        <div className="benchmark-metric">
          <span className="benchmark-metric-label">String accuracy</span>
          <strong>{pct(m.string_accuracy)}</strong>
        </div>
        <div className="benchmark-metric">
          <span className="benchmark-metric-label">Timing accuracy</span>
          <strong>{pct(m.timing_accuracy)}</strong>
        </div>
        <div className="benchmark-metric benchmark-metric--overall">
          <span className="benchmark-metric-label">Overall F1</span>
          <strong>{pct(m.overall_f1)}</strong>
        </div>
      </div>

      <div className="benchmark-legend">
        {(Object.keys(BENCHMARK_STATUS_LABELS) as NoteMatchStatus[]).map((status) => (
          <span key={status} className="benchmark-legend-item">
            <span
              className="benchmark-legend-swatch"
              style={{ background: BENCHMARK_STATUS_COLORS[status] }}
            />
            {BENCHMARK_STATUS_LABELS[status]}
            {statusCounts[status] ? ` (${statusCounts[status]})` : ""}
          </span>
        ))}
      </div>

      {comparison.reference_url ? (
        <p className="benchmark-ref-link muted">
          Эталон:{" "}
          <a href={comparison.reference_url} target="_blank" rel="noreferrer">
            Songsterr reference
          </a>
        </p>
      ) : null}

      <div className="benchmark-table-wrap">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Статус</th>
              <th>Эталон (ms)</th>
              <th>Предсказание (ms)</th>
              <th>Струна</th>
              <th>Лад</th>
              <th>Δt (ms)</th>
            </tr>
          </thead>
          <tbody>
            {alignments.map((row, idx) => (
              <tr key={`${row.status}-${idx}`}>
                <td>
                  <span
                    className="benchmark-status-dot"
                    style={{ background: BENCHMARK_STATUS_COLORS[row.status] }}
                  />
                  {BENCHMARK_STATUS_LABELS[row.status]}
                </td>
                <td>{row.ref_start_ms?.toFixed(0) ?? "—"}</td>
                <td>{row.pred_start_ms?.toFixed(0) ?? "—"}</td>
                <td>
                  {row.ref_string ?? "—"}
                  {row.pred_string != null && row.ref_string !== row.pred_string
                    ? ` → ${row.pred_string}`
                    : ""}
                </td>
                <td>
                  {row.ref_fret ?? "—"}
                  {row.pred_fret != null && row.ref_fret !== row.pred_fret ? ` → ${row.pred_fret}` : ""}
                </td>
                <td>{row.timing_delta_ms?.toFixed(0) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
