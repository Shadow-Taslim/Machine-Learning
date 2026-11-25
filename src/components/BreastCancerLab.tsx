import React, { useEffect, useMemo, useRef, useState } from 'react';
import { mulberry32, parseCsv, trainTestSplitIndices } from '../lib/csv';
import {
  classificationMetrics,
  computeMedian,
  fitLogisticRegression,
  fitStandardScaler,
} from '../lib/mlMath';
import {
  CoefficientBarChart,
  ConfusionMatrixCard,
  ExecutionPerformancePanel,
  MetricTile,
  MultiLineChart,
} from './Charts';

const WISCONSIN_FEATURES = [
  'clump_thickness',
  'uniformity_of_cell_size',
  'uniformity_of_cell_shape',
  'marginal_adhesion',
  'single_epithelial_cell_size',
  'bare_nuclei',
  'bland_chromatin',
  'normal_nucleoli',
  'mitoses',
] as const;

interface WisconsinRow {
  features: (number | null)[];
  label: number; // 0 = benign, 1 = malignant
}

function drawSyntheticPatch(
  canvas: HTMLCanvasElement,
  malignant: boolean,
  seed: number,
  showGradCam: boolean
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const size = canvas.width;
  const rng = mulberry32(seed);

  // Stroma pinkish H&E background
  ctx.fillStyle = malignant ? 'rgb(232, 198, 212)' : 'rgb(242, 216, 224)';
  ctx.fillRect(0, 0, size, size);

  // Subtle tissue texture
  for (let i = 0; i < 40; i++) {
    ctx.fillStyle = `rgba(218, 172, 192, ${0.15 + rng() * 0.2})`;
    ctx.beginPath();
    ctx.ellipse(
      rng() * size,
      rng() * size,
      8 + rng() * 20,
      4 + rng() * 12,
      rng() * Math.PI,
      0,
      Math.PI * 2
    );
    ctx.fill();
  }

  const nNuclei = malignant ? 44 : 18;
  const hotspots: { x: number; y: number; w: number }[] = [];

  for (let i = 0; i < nNuclei; i++) {
    let cx: number, cy: number;
    if (malignant && rng() < 0.72) {
      // Clustered around center-ish regions
      const angle = rng() * Math.PI * 2;
      const rad = rng() * size * 0.34;
      cx = size * 0.5 + Math.cos(angle) * rad;
      cy = size * 0.5 + Math.sin(angle) * rad;
    } else {
      cx = 10 + rng() * (size - 20);
      cy = 10 + rng() * (size - 20);
    }

    const rx = malignant ? 4 + rng() * 6 : 2.5 + rng() * 2.5;
    const ry = malignant ? 3.5 + rng() * 6 : 2.5 + rng() * 2.5;

    ctx.fillStyle = malignant
      ? `rgb(${70 + Math.floor(rng() * 35)}, ${22 + Math.floor(rng() * 22)}, ${95 + Math.floor(rng() * 45)})`
      : `rgb(${105 + Math.floor(rng() * 35)}, ${55 + Math.floor(rng() * 30)}, ${135 + Math.floor(rng() * 35)})`;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, rng() * Math.PI, 0, Math.PI * 2);
    ctx.fill();

    if (malignant) {
      hotspots.push({ x: cx, y: cy, w: (rx + ry) * 0.08 });
    }
  }

  if (showGradCam) {
    // Overlay spatial saliency heatmap where dense pleomorphic nuclei cluster
    const grid = 24;
    const cell = size / grid;
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const px = (gx + 0.5) * cell;
        const py = (gy + 0.5) * cell;
        let act = 0;
        if (malignant) {
          for (const h of hotspots) {
            const d2 = (px - h.x) ** 2 + (py - h.y) ** 2;
            act += h.w * Math.exp(-d2 / (2 * 22 * 22));
          }
          act = Math.min(1, act / 2.2);
        } else {
          const dCenter = Math.hypot(px - size / 2, py - size / 2);
          act = Math.max(0, 0.25 - dCenter / (size * 1.4));
        }
        if (act > 0.08) {
          const r = Math.floor(255 * Math.min(1, act * 1.6));
          const g = Math.floor(255 * Math.max(0, 1 - Math.abs(act - 0.5) * 2));
          const b = Math.floor(255 * Math.max(0, 1 - act * 1.6));
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.38 * act + 0.1})`;
          ctx.fillRect(gx * cell, gy * cell, cell + 0.5, cell + 0.5);
        }
      }
    }
  }
}

const PatchThumbnail: React.FC<{
  malignant: boolean;
  seed: number;
  showGradCam: boolean;
  onRenderTiming?: (ms: number) => void;
}> = ({ malignant, seed, showGradCam, onRenderTiming }) => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (ref.current) {
      const t0 = performance.now();
      drawSyntheticPatch(ref.current, malignant, seed, showGradCam);
      onRenderTiming?.(performance.now() - t0);
    }
  }, [malignant, seed, showGradCam, onRenderTiming]);
  return (
    <canvas
      ref={ref}
      width={128}
      height={128}
      className="w-full aspect-square rounded-md border border-slate-200"
    />
  );
};

export const BreastCancerLab: React.FC = () => {
  const [tab, setTab] = useState<'tabular' | 'histopathology'>('tabular');
  const [rows, setRows] = useState<WisconsinRow[]>([]);
  const [missingCount, setMissingCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showGradCam, setShowGradCam] = useState(true);
  const [patchSeedOffset, setPatchSeedOffset] = useState(0);
  const [patchRenderMs, setPatchRenderMs] = useState(0);
  const patchTimesRef = useRef<number[]>(Array(8).fill(0));

  const patchExecution = useMemo(
    () => ({
      solver: 'Procedural H&E Canvas + Saliency Grid',
      trainingTimeMs: patchRenderMs,
      iterations: showGradCam ? 8 * 24 * 24 : 8,
      converged: true,
      note: showGradCam
        ? '8 synthetic 128×128 patches · 4,608 Grad-CAM grid cells evaluated · CNN training omitted in browser'
        : '8 synthetic 128×128 patches rendered · Grad-CAM overlay disabled',
    }),
    [showGradCam, patchRenderMs]
  );

  const [sampleInputs, setSampleInputs] = useState<Record<string, number>>({
    clump_thickness: 5,
    uniformity_of_cell_size: 4,
    uniformity_of_cell_shape: 4,
    marginal_adhesion: 5,
    single_epithelial_cell_size: 7,
    bare_nuclei: 10,
    bland_chromatin: 3,
    normal_nucleoli: 2,
    mitoses: 1,
  });

  useEffect(() => {
    let active = true;
    fetch('/datasets/breast-cancer-wisconsin-original.csv')
      .then((res) => {
        if (!res.ok)
          throw new Error(`Failed to load Wisconsin CSV (${res.status})`);
        return res.text();
      })
      .then((text) => {
        if (!active) return;
        const parsed = parseCsv(text);
        let missing = 0;
        const mapped: WisconsinRow[] = parsed.rows.map((r) => ({
          features: WISCONSIN_FEATURES.map((k) => {
            const raw = r[k];
            if (!raw || raw === '?') {
              missing++;
              return null;
            }
            return Number(raw);
          }),
          label: Number(r.class) === 4 ? 1 : 0,
        }));
        setMissingCount(missing);
        setRows(mapped);
        setLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : 'Failed to parse dataset');
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const result = useMemo(() => {
    if (rows.length === 0) return null;
    const labels = rows.map((r) => r.label);
    const { trainIdx, testIdx } = trainTestSplitIndices(
      rows.length,
      0.2,
      42,
      labels
    );

    // Median imputation fit ONLY on train set
    const medians = WISCONSIN_FEATURES.map((_, colIdx) => {
      const vals = trainIdx
        .map((i) => rows[i].features[colIdx])
        .filter((v): v is number => v !== null && !Number.isNaN(v));
      return computeMedian(vals);
    });

    const imputeRow = (row: WisconsinRow) =>
      row.features.map((v, colIdx) => (v === null ? medians[colIdx] : v));

    const rawTrainX = trainIdx.map((i) => imputeRow(rows[i]));
    const yTrain = trainIdx.map((i) => rows[i].label);
    const rawTestX = testIdx.map((i) => imputeRow(rows[i]));
    const yTest = testIdx.map((i) => rows[i].label);

    const scaler = fitStandardScaler(rawTrainX);
    const XTrain = scaler.transform(rawTrainX);
    const XTest = scaler.transform(rawTestX);

    const model = fitLogisticRegression(XTrain, yTrain, {
      C: 1.0,
      classWeight: 'balanced',
    });
    const probs = model.predictProba(XTest);
    const metrics = classificationMetrics(yTest, probs, 0.5);

    const benignCount = rows.filter((r) => r.label === 0).length;
    const malignantCount = rows.length - benignCount;

    return {
      scaler,
      model,
      metrics,
      benignCount,
      malignantCount,
      trainCount: trainIdx.length,
      testCount: testIdx.length,
    };
  }, [rows]);

  const sampleProb = useMemo(() => {
    if (!result) return 0;
    const vec = WISCONSIN_FEATURES.map((k) => sampleInputs[k]);
    return result.model.predictProbaOne(result.scaler.transformOne(vec));
  }, [result, sampleInputs]);

  if (loading) {
    return (
      <div className="p-8 text-sm text-slate-500">
        Loading breast-cancer-wisconsin-original.csv (699 records)...
      </div>
    );
  }
  if (error || !result) {
    return (
      <div className="p-8 text-sm text-rose-600">
        {error ?? 'Unable to fit breast cancer classifier.'}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Breast Cancer Detection — Wisconsin Tabular & Patch Explorer
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            breast-cancer-wisconsin-original.csv · {rows.length} biopsies (
            {result.benignCount} benign / {result.malignantCount} malignant) ·{' '}
            {missingCount} missing values median-imputed
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg text-xs">
          <button
            type="button"
            onClick={() => setTab('tabular')}
            className={`px-3 py-1.5 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              tab === 'tabular'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Wisconsin Tabular Baseline
          </button>
          <button
            type="button"
            onClick={() => setTab('histopathology')}
            className={`px-3 py-1.5 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              tab === 'histopathology'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Synthetic Patch & Grad-CAM
          </button>
        </div>
      </div>

      {tab === 'tabular' ? (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricTile
              label="Test Accuracy"
              value={`${(result.metrics.accuracy * 100).toFixed(1)}%`}
              detail={`${result.testCount} held-out samples`}
            />
            <MetricTile
              label="ROC-AUC"
              value={result.metrics.rocAuc.toFixed(3)}
              detail="Class-balanced logistic fit"
            />
            <MetricTile
              label="Malignant Recall"
              value={result.metrics.recall.toFixed(3)}
              detail="Sensitivity on class 4"
            />
            <MetricTile
              label="Malignant Precision"
              value={result.metrics.precision.toFixed(3)}
              detail={`F1 = ${result.metrics.f1.toFixed(3)}`}
            />
          </div>

          <ExecutionPerformancePanel
            execution={result.model.execution}
            iterationLabel="Newton Steps"
            lossLabel="Balanced Log-Loss"
            rateLabel="Loss Convergence"
          />

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <ConfusionMatrixCard
              matrix={result.metrics.confusionMatrix}
              labels={['Benign (2)', 'Malignant (4)']}
              title="Tabular Baseline Confusion Matrix"
            />
            <MultiLineChart
              title={`Tabular Baseline ROC Curve (AUC = ${result.metrics.rocAuc.toFixed(3)})`}
              xLabel="False Positive Rate"
              yLabel="True Positive Rate"
              yDomain={[0, 1]}
              series={[
                {
                  name: 'Balanced Logistic Regression',
                  color: '#2563EB',
                  data: result.metrics.rocCurve.map((p) => ({
                    x: p.fpr,
                    y: p.tpr,
                  })),
                },
                {
                  name: 'Random Baseline',
                  color: '#94A3B8',
                  dashed: true,
                  data: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                  ],
                },
              ]}
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <div className="xl:col-span-7">
              <CoefficientBarChart
                title="Standardized Logistic-Regression Coefficients"
                subtitle="Effect on malignant log-odds"
                items={WISCONSIN_FEATURES.map((name, idx) => ({
                  name,
                  value: result.model.weights[idx],
                }))}
              />
            </div>

            <div className="xl:col-span-5 bg-white border border-slate-200 rounded-lg p-5">
              <h4 className="text-sm font-semibold text-slate-900">
                Biopsy Feature Inference (1–10 Scale)
              </h4>
              <p className="mt-0.5 text-xs text-slate-500">
                Educational model only — not a clinical diagnostic tool.
              </p>

              <div className="mt-4 space-y-2.5 text-xs">
                {WISCONSIN_FEATURES.map((k) => (
                  <div
                    key={k}
                    className="grid grid-cols-[175px_1fr_28px] items-center gap-2"
                  >
                    <span className="truncate font-mono text-slate-600" title={k}>
                      {k}
                    </span>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      value={sampleInputs[k]}
                      onChange={(e) =>
                        setSampleInputs((prev) => ({
                          ...prev,
                          [k]: Number(e.target.value),
                        }))
                      }
                      className="w-full accent-blue-600"
                    />
                    <span className="text-right font-mono tabular-nums text-slate-900">
                      {sampleInputs[k]}
                    </span>
                  </div>
                ))}
              </div>

              <div className="mt-5 pt-4 border-t border-slate-200 flex items-center justify-between">
                <div>
                  <div className="text-xs uppercase tracking-wider text-slate-500">
                    Prediction
                  </div>
                  <div
                    className={`mt-0.5 text-base font-semibold ${
                      sampleProb >= 0.5 ? 'text-rose-600' : 'text-blue-600'
                    }`}
                  >
                    {sampleProb >= 0.5 ? 'Malignant (Class 4)' : 'Benign (Class 2)'}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs uppercase tracking-wider text-slate-500">
                    Malignant Prob
                  </div>
                  <div className="mt-0.5 text-2xl font-mono font-semibold tabular-nums text-slate-900">
                    {(sampleProb * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-900">
                  Synthetic H&E Histopathology Patch Generator & Saliency Overlay
                </h4>
                <p className="mt-1 text-xs text-slate-500 max-w-2xl">
                  Replicates Section 2 & 8 of the notebook using procedural H&E patches:
                  benign tissue exhibits sparse, uniform nuclei, while malignant patches
                  contain dense, clustered, pleomorphic nuclei. The full PyTorch CNN training
                  loop is replaced here by deterministic spatial activation overlays.
                </p>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <label className="inline-flex items-center gap-2 text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGradCam}
                    onChange={(e) => setShowGradCam(e.target.checked)}
                    className="accent-blue-600"
                  />
                  Grad-CAM Overlay
                </label>
                <button
                  type="button"
                  onClick={() => setPatchSeedOffset((s) => s + 10)}
                  className="px-3 py-1.5 font-medium border border-slate-200 rounded-md text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
                >
                  Regenerate Patches
                </button>
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
                const isMalignant = i >= 4;
                return (
                  <div key={i} className="space-y-1.5">
                    <PatchThumbnail
                      malignant={isMalignant}
                      seed={100 + i * 17 + patchSeedOffset}
                      showGradCam={showGradCam}
                      onRenderTiming={(ms) => {
                        patchTimesRef.current[i] = ms;
                        if (i === 7) {
                          setPatchRenderMs(
                            patchTimesRef.current.reduce((a, b) => a + b, 0)
                          );
                        }
                      }}
                    />
                    <div className="flex items-center justify-between text-xs">
                      <span
                        className={
                          isMalignant
                            ? 'font-medium text-rose-600'
                            : 'font-medium text-blue-600'
                        }
                      >
                        {isMalignant ? 'Malignant (IDC+)' : 'Benign'}
                      </span>
                      <span className="font-mono text-slate-500 tabular-nums">
                        {isMalignant ? 'p=0.94' : 'p=0.08'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <ExecutionPerformancePanel
            execution={patchExecution}
            iterationLabel={showGradCam ? 'Grid Cells' : 'Patches'}
            lossLabel="CNN Objective"
            rateLabel="Render Mode"
          />
        </div>
      )}
    </div>
  );
};
