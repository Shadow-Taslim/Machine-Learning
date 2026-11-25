import React, { useEffect, useMemo, useState } from 'react';
import { parseCsv, trainTestSplitIndices } from '../lib/csv';
import { fitLinearRegression, regressionMetrics } from '../lib/mlMath';
import {
  CoefficientBarChart,
  ExecutionPerformancePanel,
  MetricTile,
  ScatterPlot,
} from './Charts';

interface InsuranceRow {
  age: number;
  sex: number; // 0 male, 1 female
  bmi: number;
  children: number;
  smoker: number; // 0 no, 1 yes
  region: string;
  charges: number;
}

const FEATURE_NAMES = [
  'age',
  'sex (female=1)',
  'bmi',
  'children',
  'smoker (yes=1)',
  'region_northwest',
  'region_southeast',
  'region_southwest',
];

function encodeRow(r: InsuranceRow): number[] {
  return [
    r.age,
    r.sex,
    r.bmi,
    r.children,
    r.smoker,
    r.region === 'northwest' ? 1 : 0,
    r.region === 'southeast' ? 1 : 0,
    r.region === 'southwest' ? 1 : 0,
  ];
}

export const LinearRegressionLab: React.FC = () => {
  const [rows, setRows] = useState<InsuranceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testSize, setTestSize] = useState(0.2);
  const [seed, setSeed] = useState(42);

  // Interactive predictor inputs
  const [age, setAge] = useState(34);
  const [sex, setSex] = useState<'male' | 'female'>('female');
  const [bmi, setBmi] = useState(28.4);
  const [children, setChildren] = useState(1);
  const [smoker, setSmoker] = useState<'no' | 'yes'>('no');
  const [region, setRegion] = useState('southeast');

  useEffect(() => {
    let active = true;
    fetch('/datasets/insurance.csv')
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load insurance.csv (${res.status})`);
        return res.text();
      })
      .then((text) => {
        if (!active) return;
        const parsed = parseCsv(text);
        const mapped: InsuranceRow[] = parsed.rows.map((r) => ({
          age: Number(r.age),
          sex: r.sex === 'female' ? 1 : 0,
          bmi: Number(r.bmi),
          children: Number(r.children),
          smoker: r.smoker === 'yes' ? 1 : 0,
          region: r.region,
          charges: Number(r.charges),
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
    const { trainIdx, testIdx } = trainTestSplitIndices(rows.length, testSize, seed);
    const XTrain = trainIdx.map((i) => encodeRow(rows[i]));
    const yTrain = trainIdx.map((i) => rows[i].charges);
    const XTest = testIdx.map((i) => encodeRow(rows[i]));
    const yTest = testIdx.map((i) => rows[i].charges);

    const model = fitLinearRegression(XTrain, yTrain);
    const yPred = model.predict(XTest);
    const metrics = regressionMetrics(yTest, yPred);

    return {
      model,
      metrics,
      trainCount: trainIdx.length,
      testCount: testIdx.length,
      yTest,
      yPred,
    };
  }, [rows, testSize, seed]);

  const predictedCharge = useMemo(() => {
    if (!result) return 0;
    return result.model.predictOne(
      encodeRow({
        age,
        sex: sex === 'female' ? 1 : 0,
        bmi,
        children,
        smoker: smoker === 'yes' ? 1 : 0,
        region,
        charges: 0,
      })
    );
  }, [result, age, sex, bmi, children, smoker, region]);

  if (loading) {
    return (
      <div className="p-8 text-sm text-slate-500">
        Loading insurance.csv (1,338 records)...
      </div>
    );
  }
  if (error || !result) {
    return (
      <div className="p-8 text-sm text-rose-600">
        {error ?? 'Unable to fit linear regression model.'}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Linear Regression — Medical Insurance Charges
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            insurance.csv · {rows.length.toLocaleString()} records ·{' '}
            {result.trainCount} train / {result.testCount} test · Ordinary Least Squares
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center gap-2 text-slate-600">
            Test split
            <select
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
              className="border border-slate-200 bg-white rounded-md px-2.5 py-1.5 font-mono text-slate-800"
            >
              <option value={0.15}>15%</option>
              <option value={0.2}>20%</option>
              <option value={0.3}>30%</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-slate-600">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 42)}
              className="w-20 border border-slate-200 bg-white rounded-md px-2.5 py-1.5 font-mono text-slate-800"
            />
          </label>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          label="R² Score"
          value={result.metrics.r2.toFixed(4)}
          detail="Explained variance"
        />
        <MetricTile
          label="RMSE"
          value={`$${Math.round(result.metrics.rmse).toLocaleString()}`}
          unit="USD"
          detail="Root mean squared error"
        />
        <MetricTile
          label="MAE"
          value={`$${Math.round(result.metrics.mae).toLocaleString()}`}
          unit="USD"
          detail="Mean absolute error"
        />
        <MetricTile
          label="Intercept (β₀)"
          value={`$${Math.round(result.model.intercept).toLocaleString()}`}
          detail="OLS baseline term"
        />
      </div>

      <ExecutionPerformancePanel
        execution={result.model.execution}
        iterationLabel="Matrix Pivots"
        lossLabel="Train MSE"
        rateLabel="Residual ‖Ax−b‖₂"
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ScatterPlot
          title="Actual vs Predicted Charges (Test Set)"
          xLabel="Actual Charges ($)"
          yLabel="Predicted Charges ($)"
          showDiagonal
          points={result.yTest.map((y, idx) => ({
            x: y,
            y: result.yPred[idx],
            label: `Test #${idx + 1}`,
          }))}
        />
        <ScatterPlot
          title="Age vs Charges by Smoker Status"
          xLabel="Age (years)"
          yLabel="Annual Charges ($)"
          groupColors={{ smoker: '#E11D48', nonsmoker: '#2563EB' }}
          points={rows.slice(0, 450).map((r) => ({
            x: r.age,
            y: r.charges,
            group: r.smoker ? 'smoker' : 'nonsmoker',
            label: r.smoker ? 'Smoker' : 'Non-smoker',
          }))}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-7">
          <CoefficientBarChart
            title="Fitted OLS Coefficients"
            subtitle="Dollar impact per unit increase"
            items={FEATURE_NAMES.map((name, idx) => ({
              name,
              value: result.model.weights[idx],
            }))}
          />
        </div>

        <div className="xl:col-span-5 bg-white border border-slate-200 rounded-lg p-5">
          <h4 className="text-sm font-semibold text-slate-900">
            Interactive Charge Predictor
          </h4>
          <p className="mt-0.5 text-xs text-slate-500">
            Adjust profile parameters to evaluate the fitted OLS equation live.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
            <label className="block">
              <span className="text-slate-600">Age ({age} yrs)</span>
              <input
                type="range"
                min={18}
                max={64}
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
                className="mt-1 w-full accent-blue-600"
              />
            </label>
            <label className="block">
              <span className="text-slate-600">BMI ({bmi.toFixed(1)})</span>
              <input
                type="range"
                min={16}
                max={50}
                step={0.1}
                value={bmi}
                onChange={(e) => setBmi(Number(e.target.value))}
                className="mt-1 w-full accent-blue-600"
              />
            </label>
            <label className="block">
              <span className="text-slate-600">Children</span>
              <select
                value={children}
                onChange={(e) => setChildren(Number(e.target.value))}
                className="mt-1 w-full border border-slate-200 rounded-md px-2.5 py-1.5 font-mono"
              >
                {[0, 1, 2, 3, 4, 5].map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-slate-600">Sex</span>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value as 'male' | 'female')}
                className="mt-1 w-full border border-slate-200 rounded-md px-2.5 py-1.5"
              >
                <option value="female">female</option>
                <option value="male">male</option>
              </select>
            </label>
            <label className="block">
              <span className="text-slate-600">Smoker</span>
              <select
                value={smoker}
                onChange={(e) => setSmoker(e.target.value as 'no' | 'yes')}
                className="mt-1 w-full border border-slate-200 rounded-md px-2.5 py-1.5"
              >
                <option value="no">no</option>
                <option value="yes">yes</option>
              </select>
            </label>
            <label className="block">
              <span className="text-slate-600">Region</span>
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="mt-1 w-full border border-slate-200 rounded-md px-2.5 py-1.5"
              >
                <option value="northeast">northeast</option>
                <option value="northwest">northwest</option>
                <option value="southeast">southeast</option>
                <option value="southwest">southwest</option>
              </select>
            </label>
          </div>

          <div className="mt-5 pt-4 border-t border-slate-200 flex items-baseline justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">
              Predicted Annual Charges
            </span>
            <span className="text-2xl font-mono font-semibold tabular-nums text-blue-600">
              ${Math.max(0, predictedCharge).toLocaleString(undefined, {
                maximumFractionDigits: 2,
              })}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
