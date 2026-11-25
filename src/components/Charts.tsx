import React, { useState } from 'react';

import type { ExecutionMetrics } from '../lib/mlMath';

export interface MetricTileProps {
  label: string;
  value: string;
  unit?: string;
  detail?: string;
}

export const MetricTile: React.FC<MetricTileProps> = ({
  label,
  value,
  unit,
  detail,
}) => (
  <div className="bg-white border border-slate-200 rounded-lg px-4 py-3">
    <div className="text-[11px] font-medium tracking-wider uppercase text-slate-500">
      {label}
    </div>
    <div className="mt-1 flex items-baseline gap-1.5">
      <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
        {value}
      </span>
      {unit && (
        <span className="text-xs font-mono uppercase text-slate-400">
          {unit}
        </span>
      )}
    </div>
    {detail && (
      <div className="mt-1 text-xs font-mono tabular-nums text-slate-500">
        {detail}
      </div>
    )}
  </div>
);

function formatDurationMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 10) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatCompactLoss(v: number): string {
  if (Math.abs(v) >= 1000) return v.toExponential(2);
  if (Math.abs(v) > 0 && Math.abs(v) < 0.001) return v.toExponential(2);
  return v.toFixed(4);
}

export const ExecutionPerformancePanel: React.FC<{
  execution: ExecutionMetrics;
  iterationLabel?: string;
  lossLabel?: string;
  rateLabel?: string;
}> = ({
  execution,
  iterationLabel = 'Iterations',
  lossLabel = 'Objective Loss',
  rateLabel = 'Convergence Rate',
}) => {
  const history = execution.lossHistory;
  const hasCurve = Boolean(history && history.length >= 2);

  let sparkPath = '';
  if (hasCurve && history) {
    const w = 220;
    const h = 40;
    const minV = Math.min(...history);
    const maxV = Math.max(...history);
    const span = maxV - minV || 1;
    sparkPath = history
      .map((val, idx) => {
        const x = (idx / (history.length - 1)) * (w - 4) + 2;
        const y = h - 4 - ((val - minV) / span) * (h - 8);
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }

  const hasInitialAndFinal =
    execution.initialLoss !== undefined && execution.finalLoss !== undefined;

  return (
    <section
      aria-label="Execution and convergence telemetry"
      className="bg-white border border-slate-200 rounded-lg px-4 py-3.5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700">
            Execution Performance
          </h3>
          <span className="text-xs font-mono text-slate-500">
            {execution.solver}
          </span>
        </div>
        <span
          className={`text-xs font-mono font-medium ${
            execution.converged ? 'text-emerald-700' : 'text-amber-700'
          }`}
        >
          {execution.converged ? '● Converged' : '▲ Iteration Limit'}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-4 items-center">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500">
            Training Time
          </div>
          <div className="mt-0.5 text-base font-mono font-semibold tabular-nums text-slate-900">
            {formatDurationMs(execution.trainingTimeMs)}
          </div>
        </div>

        <div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500">
            {iterationLabel}
          </div>
          <div className="mt-0.5 text-base font-mono font-semibold tabular-nums text-slate-900">
            {execution.iterations.toLocaleString()}
          </div>
        </div>

        <div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500">
            {lossLabel}
          </div>
          <div className="mt-0.5 text-sm font-mono font-semibold tabular-nums text-slate-900">
            {hasInitialAndFinal
              ? `${formatCompactLoss(execution.initialLoss!)} → ${formatCompactLoss(
                  execution.finalLoss!
                )}`
              : execution.finalLoss !== undefined
              ? formatCompactLoss(execution.finalLoss)
              : '—'}
          </div>
        </div>

        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-slate-500">
              {rateLabel}
            </div>
            <div className="mt-0.5 text-sm font-mono font-semibold tabular-nums text-slate-900">
              {execution.lossReductionPct !== undefined
                ? `−${execution.lossReductionPct.toFixed(1)}%`
                : execution.convergenceRate !== undefined
                ? execution.convergenceRate.toExponential(2)
                : '—'}
              {execution.convergenceRate !== undefined &&
                execution.lossReductionPct !== undefined && (
                  <span className="ml-1.5 text-[11px] font-normal text-slate-500">
                    ({execution.convergenceRate >= 0 ? '−' : '+'}
                    {Math.abs(execution.convergenceRate).toFixed(4)}/step)
                  </span>
                )}
            </div>
          </div>

          {hasCurve && (
            <svg
              viewBox="0 0 220 40"
              className="w-24 h-8 shrink-0 hidden sm:block"
              aria-label="Loss convergence curve"
            >
              <path
                d={sparkPath}
                fill="none"
                stroke="#2563EB"
                strokeWidth="1.75"
              />
            </svg>
          )}
        </div>
      </div>

      {execution.note && (
        <div className="mt-2.5 pt-2 border-t border-slate-100 text-xs font-mono text-slate-500">
          {execution.note}
        </div>
      )}
    </section>
  );
};

export interface ScatterPoint {
  x: number;
  y: number;
  group?: string;
  label?: string;
}

export const ScatterPlot: React.FC<{
  points: ScatterPoint[];
  xLabel: string;
  yLabel: string;
  title: string;
  showDiagonal?: boolean;
  groupColors?: Record<string, string>;
  height?: number;
}> = ({
  points,
  xLabel,
  yLabel,
  title,
  showDiagonal = false,
  groupColors = { default: '#2563EB' },
  height = 280,
}) => {
  const [hoverPt, setHoverPt] = useState<ScatterPoint | null>(null);
  const w = 520;
  const h = height;
  const pad = { top: 20, right: 20, bottom: 42, left: 62 };

  if (points.length === 0) {
    return (
      <div className="border border-slate-200 bg-white rounded-lg p-4 text-xs text-slate-500">
        No points available.
      </div>
    );
  }

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (showDiagonal) {
    minX = Math.min(minX, minY);
    minY = minX;
    maxX = Math.max(maxX, maxY);
    maxY = maxX;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  minX -= spanX * 0.04;
  maxX += spanX * 0.04;
  minY -= spanY * 0.04;
  maxY += spanY * 0.04;

  const toSx = (x: number) =>
    pad.left + ((x - minX) / (maxX - minX)) * (w - pad.left - pad.right);
  const toSy = (y: number) =>
    h - pad.bottom - ((y - minY) / (maxY - minY)) * (h - pad.top - pad.bottom);

  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const formatTick = (v: number) =>
    Math.abs(v) >= 1000
      ? `${(v / 1000).toFixed(1)}k`
      : Math.abs(v) < 10
      ? v.toFixed(1)
      : Math.round(v).toString();

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
        <div className="text-xs font-mono tabular-nums text-slate-500 h-4">
          {hoverPt
            ? `${hoverPt.label ? `${hoverPt.label} · ` : ''}(${hoverPt.x.toFixed(1)}, ${hoverPt.y.toFixed(1)})`
            : `${points.length} pts`}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full h-auto select-none cursor-crosshair"
      >
        {ticks.map((t) => {
          const xVal = minX + t * (maxX - minX);
          const yVal = minY + t * (maxY - minY);
          const sx = toSx(xVal);
          const sy = toSy(yVal);
          return (
            <g key={t}>
              <line
                x1={pad.left}
                y1={sy}
                x2={w - pad.right}
                y2={sy}
                stroke="#E2E8F0"
                strokeWidth="1"
              />
              <line
                x1={sx}
                y1={pad.top}
                x2={sx}
                y2={h - pad.bottom}
                stroke="#F1F5F9"
                strokeWidth="1"
              />
              <text
                x={pad.left - 8}
                y={sy + 3}
                textAnchor="end"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {formatTick(yVal)}
              </text>
              <text
                x={sx}
                y={h - pad.bottom + 14}
                textAnchor="middle"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {formatTick(xVal)}
              </text>
            </g>
          );
        })}

        {showDiagonal && (
          <line
            x1={toSx(minX)}
            y1={toSy(minX)}
            x2={toSx(maxX)}
            y2={toSy(maxX)}
            stroke="#94A3B8"
            strokeDasharray="4 4"
            strokeWidth="1.25"
          />
        )}

        {points.map((p, idx) => {
          const color = groupColors[p.group ?? 'default'] ?? '#2563EB';
          return (
            <circle
              key={idx}
              cx={toSx(p.x)}
              cy={toSy(p.y)}
              r={hoverPt === p ? 4.5 : 3}
              fill={color}
              fillOpacity={0.72}
              stroke="#FFFFFF"
              strokeWidth={0.6}
              onMouseEnter={() => setHoverPt(p)}
              onMouseLeave={() => setHoverPt(null)}
            />
          );
        })}

        <text
          x={(pad.left + w - pad.right) / 2}
          y={h - 6}
          textAnchor="middle"
          className="fill-slate-600 text-[11px] font-medium"
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={(pad.top + h - pad.bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${(pad.top + h - pad.bottom) / 2})`}
          className="fill-slate-600 text-[11px] font-medium"
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
};

export interface LineSeries {
  name: string;
  color: string;
  data: { x: number; y: number }[];
  dashed?: boolean;
}

export const MultiLineChart: React.FC<{
  series: LineSeries[];
  title: string;
  xLabel: string;
  yLabel: string;
  yDomain?: [number, number];
  height?: number;
}> = ({ series, title, xLabel, yLabel, yDomain, height = 240 }) => {
  const w = 520;
  const h = height;
  const pad = { top: 20, right: 20, bottom: 40, left: 54 };

  const allPts = series.flatMap((s) => s.data);
  if (allPts.length === 0) return null;

  let minX = Infinity,
    maxX = -Infinity,
    minY = yDomain ? yDomain[0] : Infinity,
    maxY = yDomain ? yDomain[1] : -Infinity;

  for (const p of allPts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (!yDomain) {
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }
  if (maxX === minX) maxX = minX + 1;
  if (maxY === minY) maxY = minY + 1;

  const toSx = (x: number) =>
    pad.left + ((x - minX) / (maxX - minX)) * (w - pad.left - pad.right);
  const toSy = (y: number) =>
    h - pad.bottom - ((y - minY) / (maxY - minY)) * (h - pad.top - pad.bottom);

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
        <div className="flex items-center gap-4 text-xs text-slate-600">
          {series.map((s) => (
            <span key={s.name} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-0.5"
                style={{ backgroundColor: s.color }}
              />
              {s.name}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto select-none">
        {ticks.map((t) => {
          const yVal = minY + t * (maxY - minY);
          const xVal = minX + t * (maxX - minX);
          const sy = toSy(yVal);
          const sx = toSx(xVal);
          return (
            <g key={t}>
              <line
                x1={pad.left}
                y1={sy}
                x2={w - pad.right}
                y2={sy}
                stroke="#E2E8F0"
                strokeWidth="1"
              />
              <text
                x={pad.left - 8}
                y={sy + 3}
                textAnchor="end"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {Math.abs(maxY) <= 1.1 ? yVal.toFixed(2) : Math.round(yVal)}
              </text>
              <text
                x={sx}
                y={h - pad.bottom + 14}
                textAnchor="middle"
                className="fill-slate-400 text-[10px] font-mono"
              >
                {Math.round(xVal)}
              </text>
            </g>
          );
        })}

        {series.map((s) => {
          if (s.data.length === 0) return null;
          // Downsample slightly if > 300 pts for crisp SVG rendering
          const step = Math.max(1, Math.floor(s.data.length / 260));
          const pts = s.data.filter((_, idx) => idx % step === 0 || idx === s.data.length - 1);
          const d = pts
            .map(
              (p, i) =>
                `${i === 0 ? 'M' : 'L'} ${toSx(p.x).toFixed(1)} ${toSy(p.y).toFixed(1)}`
            )
            .join(' ');
          return (
            <path
              key={s.name}
              d={d}
              fill="none"
              stroke={s.color}
              strokeWidth={1.75}
              strokeDasharray={s.dashed ? '4 3' : undefined}
            />
          );
        })}

        <text
          x={(pad.left + w - pad.right) / 2}
          y={h - 6}
          textAnchor="middle"
          className="fill-slate-600 text-[11px] font-medium"
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={(pad.top + h - pad.bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${(pad.top + h - pad.bottom) / 2})`}
          className="fill-slate-600 text-[11px] font-medium"
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
};

export const ConfusionMatrixCard: React.FC<{
  matrix: [[number, number], [number, number]];
  labels: [string, string];
  title?: string;
}> = ({ matrix, labels, title = 'Confusion Matrix' }) => {
  const [[tn, fp], [fn, tp]] = matrix;
  const total = tn + fp + fn + tp || 1;
  const cells = [
    { row: 0, col: 0, val: tn, tag: 'TN', good: true },
    { row: 0, col: 1, val: fp, tag: 'FP', good: false },
    { row: 1, col: 0, val: fn, tag: 'FN', good: false },
    { row: 1, col: 1, val: tp, tag: 'TP', good: true },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
        <span className="text-xs font-mono tabular-nums text-slate-500">
          n = {total}
        </span>
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center text-xs">
        <div />
        <div className="text-center font-medium text-slate-500 pb-1">
          Pred: {labels[0]}
        </div>
        <div className="text-center font-medium text-slate-500 pb-1">
          Pred: {labels[1]}
        </div>

        <div className="font-medium text-slate-500 pr-2 text-right">
          True: {labels[0]}
        </div>
        {cells.slice(0, 2).map((c) => (
          <div
            key={c.tag}
            className={`rounded-md border px-3 py-3 text-center ${
              c.good
                ? 'bg-blue-50/70 border-blue-200 text-slate-900'
                : 'bg-slate-50 border-slate-200 text-slate-700'
            }`}
          >
            <div className="text-lg font-mono font-semibold tabular-nums">
              {c.val}
            </div>
            <div className="text-[11px] font-mono text-slate-500">
              {c.tag} · {((c.val / total) * 100).toFixed(1)}%
            </div>
          </div>
        ))}

        <div className="font-medium text-slate-500 pr-2 text-right">
          True: {labels[1]}
        </div>
        {cells.slice(2, 4).map((c) => (
          <div
            key={c.tag}
            className={`rounded-md border px-3 py-3 text-center ${
              c.good
                ? 'bg-blue-50/70 border-blue-200 text-slate-900'
                : 'bg-slate-50 border-slate-200 text-slate-700'
            }`}
          >
            <div className="text-lg font-mono font-semibold tabular-nums">
              {c.val}
            </div>
            <div className="text-[11px] font-mono text-slate-500">
              {c.tag} · {((c.val / total) * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const CoefficientBarChart: React.FC<{
  items: { name: string; value: number }[];
  title: string;
  subtitle?: string;
}> = ({ items, title, subtitle }) => {
  const sorted = [...items].sort((a, b) => b.value - a.value);
  const maxAbs = Math.max(
    1e-6,
    ...sorted.map((it) => Math.abs(it.value))
  );

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
        {subtitle && (
          <span className="text-xs text-slate-500">{subtitle}</span>
        )}
      </div>
      <div className="space-y-2">
        {sorted.map((item) => {
          const pct = (Math.abs(item.value) / maxAbs) * 50;
          const pos = item.value >= 0;
          return (
            <div
              key={item.name}
              className="grid grid-cols-[140px_1fr_72px] items-center gap-3 text-xs"
            >
              <span className="truncate font-mono text-slate-700" title={item.name}>
                {item.name}
              </span>
              <div className="relative h-4 bg-slate-100 rounded-sm overflow-hidden">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-300" />
                <div
                  className={`absolute top-0.5 bottom-0.5 rounded-[1px] ${
                    pos ? 'bg-rose-500' : 'bg-blue-600'
                  }`}
                  style={{
                    left: pos ? '50%' : `${50 - pct}%`,
                    width: `${pct}%`,
                  }}
                />
              </div>
              <span className="text-right font-mono tabular-nums text-slate-700">
                {item.value >= 0 ? '+' : ''}
                {item.value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
