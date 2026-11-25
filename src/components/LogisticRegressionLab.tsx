import React, { useEffect, useMemo, useState } from 'react';
import { parseCsv, trainTestSplitIndices } from '../lib/csv';
import {
  classificationMetrics,
  fitLogisticRegression,
  fitStandardScaler,
} from '../lib/mlMath';
import {
  CoefficientBarChart,
  ConfusionMatrixCard,
  MetricTile,
  MultiLineChart,
} from './Charts';

const FEATURE_KEYS = [
  'age',
  'sex',
  'cp',
  'trestbps',
  'chol',
  'fbs',
  'restecg',
  'thalach',
  'exang',
  'oldpeak',
  'slope',
  'ca',
  'thal',
] as const;

const DEFAULT_PATIENT: Record<(typeof FEATURE_KEYS)[number], number> = {
  age: 63,
  sex: 1,
  cp: 3,
  trestbps: 145,
  chol: 233,
  fbs: 1,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 2.3,
  slope: 0,
  ca: 0,
  thal: 1,
};

export const LogisticRegressionLab: React.FC = () => {
  const [rows, setRows] = useState<{ x: number[]; y: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [regC, setRegC] = useState(1.0);
  const [threshold, setThreshold] = useState(0.5);
  const [patient, setPatient] = useState({ ...DEFAULT_PATIENT });

  useEffect(() => {
    let active = true;
    fetch('/datasets/heart_disease_data.csv')
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load heart_disease_data.csv (${res.status})`);
        return res.text();
      })
      .then((text) => {
        if (!active) return;
        const parsed = parseCsv(text);
        const mapped = parsed.rows.map((r) => ({
          x: FEATURE_KEYS.map((k) => Number(r[k])),
          y: Number(r.target),
        }));
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
    const labels = rows.map((r) => r.y);
    const { trainIdx, testIdx } = trainTestSplitIndices(
      rows.length,
      0.2,
      42,
      labels
    );
    const rawTrainX = trainIdx.map((i) => rows[i].x);
    const yTrain = trainIdx.map((i) => rows[i].y);
    const rawTestX = testIdx.map((i) => rows[i].x);
    const yTest = testIdx.map((i) => rows[i].y);

    const scaler = fitStandardScaler(rawTrainX);
    const XTrain = scaler.transform(rawTrainX);
    const XTest = scaler.transform(rawTestX);

    const model = fitLogisticRegression(XTrain, yTrain, { C: regC });
    const probs = model.predictProba(XTest);
    const metrics = classificationMetrics(yTest, probs, threshold);

    return {
      scaler,
      model,
      metrics,
      trainCount: trainIdx.length,
      testCount: testIdx.length,
    };
  }, [rows, regC, threshold]);

  const patientProb = useMemo(() => {
    if (!result) return 0;
    const vec = FEATURE_KEYS.map((k) => patient[k]);
    return result.model.predictProbaOne(result.scaler.transformOne(vec));
  }, [result, patient]);

  if (loading) {
    return (
      <div className="p-8 text-sm text-slate-500">
        Loading heart_disease_data.csv (303 records)...
      </div>
    );
  }
  if (error || !result) {
    return (
      <div className="p-8 text-sm text-rose-600">
        {error ?? 'Unable to train logistic regression pipeline.'}
      </div>
    );
  }

  const predClass = patientProb >= threshold ? 1 : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Logistic Regression — Heart Disease Classification
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            heart_disease_data.csv · {rows.length} patients · {result.trainCount} train /{' '}
            {result.testCount} test stratified · StandardScaler + L2 Logistic Regression
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center gap-2 text-slate-600">
            Inverse Reg (C = {regC.toFixed(2)})
            <input
              type="range"
              min={0.05}
              max={5}
              step={0.05}
              value={regC}
              onChange={(e) => setRegC(Number(e.target.value))}
              className="w-24 accent-blue-600"
            />
          </label>
          <label className="flex items-center gap-2 text-slate-600">
            Threshold ({threshold.toFixed(2)})
            <input
              type="range"
              min={0.2}
              max={0.8}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-24 accent-blue-600"
            />
          </label>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          label="Test Accuracy"
          value={`${(result.metrics.accuracy * 100).toFixed(1)}%`}
          detail={`Threshold = ${threshold.toFixed(2)}`}
        />
        <MetricTile
          label="ROC-AUC"
          value={result.metrics.rocAuc.toFixed(3)}
          detail="Discrimination area"
        />
        <MetricTile
          label="Precision"
          value={result.metrics.precision.toFixed(3)}
          detail="Positive predictive value"
        />
        <MetricTile
          label="Recall / F1"
          value={result.metrics.recall.toFixed(3)}
          detail={`F1 = ${result.metrics.f1.toFixed(3)}`}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ConfusionMatrixCard
          matrix={result.metrics.confusionMatrix}
          labels={['No Disease (0)', 'Heart Disease (1)']}
          title="Test Set Confusion Matrix"
        />
        <MultiLineChart
          title={`ROC Curve (AUC = ${result.metrics.rocAuc.toFixed(3)})`}
          xLabel="False Positive Rate"
          yLabel="True Positive Rate"
          yDomain={[0, 1]}
          series={[
            {
              name: 'Logistic Regression',
              color: '#2563EB',
              data: result.metrics.rocCurve.map((p) => ({ x: p.fpr, y: p.tpr })),
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
        <div className="xl:col-span-6">
          <CoefficientBarChart
            title="Standardized Feature Weights"
            subtitle="Log-odds impact per 1 SD"
            items={FEATURE_KEYS.map((k, idx) => ({
              name: k,
              value: result.model.weights[idx],
            }))}
          />
        </div>

        <div className="xl:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h4 className="text-sm font-semibold text-slate-900">
                Patient Risk Inference
              </h4>
              <p className="text-xs text-slate-500">
                Initialized to notebook sample patient (age 63, cp=3, trestbps=145).
              </p>
            </div>
            <button
              type="button"
              onClick={() => setPatient({ ...DEFAULT_PATIENT })}
              className="px-3 py-1.5 text-xs font-medium border border-slate-200 rounded-md text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              Reset Sample
            </button>
          </div>

          <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            {FEATURE_KEYS.map((k) => (
              <label key={k} className="block">
                <span className="font-mono text-slate-600">{k}</span>
                <input
                  type="number"
                  step={k === 'oldpeak' ? 0.1 : 1}
                  value={patient[k]}
                  onChange={(e) =>
                    setPatient((prev) => ({
                      ...prev,
                      [k]: Number(e.target.value),
                    }))
                  }
                  className="mt-1 w-full border border-slate-200 rounded-md px-2.5 py-1.5 font-mono text-slate-900"
                />
              </label>
            ))}
          </div>

          <div className="mt-5 pt-4 border-t border-slate-200 flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500">
                Predicted Class
              </div>
              <div
                className={`mt-0.5 text-base font-semibold ${
                  predClass === 1 ? 'text-rose-600' : 'text-emerald-700'
                }`}
              >
                {predClass === 1
                  ? 'Class 1 — Heart Disease Present'
                  : 'Class 0 — No Heart Disease'}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs uppercase tracking-wider text-slate-500">
                Estimated Probability
              </div>
              <div className="mt-0.5 text-2xl font-mono font-semibold tabular-nums text-slate-900">
                {(patientProb * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
