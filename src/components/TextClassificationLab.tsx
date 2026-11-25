import React, { useMemo, useState } from 'react';
import { classificationMetrics } from '../lib/mlMath';
import {
  ReviewSample,
  SAMPLE_YELP_REVIEWS,
  trainTfidfLogistic,
} from '../lib/nlp';
import {
  CoefficientBarChart,
  ConfusionMatrixCard,
  ExecutionPerformancePanel,
  MetricTile,
} from './Charts';

export const TextClassificationLab: React.FC = () => {
  const [reviews, setReviews] = useState<ReviewSample[]>(SAMPLE_YELP_REVIEWS);
  const [regC, setRegC] = useState(2.0);
  const [testSize, setTestSize] = useState(0.25);
  const [inputText, setInputText] = useState(
    'The wood-fired pizza was crisp and flavorful, and our server was super attentive.'
  );
  const [newLabel, setNewLabel] = useState<0 | 1>(1);

  const trained = useMemo(
    () => trainTfidfLogistic(reviews, { C: regC, testSize, seed: 42 }),
    [reviews, regC, testSize]
  );

  const metrics = useMemo(
    () =>
      classificationMetrics(
        trained.testSamples.map((s) => s.label),
        trained.testProbs,
        0.5
      ),
    [trained]
  );

  const topFeatures = useMemo(() => {
    const pairs = trained.vocab.map((word, idx) => ({
      name: word,
      value: trained.weights[idx],
    }));
    const sorted = [...pairs].sort((a, b) => b.value - a.value);
    return [...sorted.slice(0, 7), ...sorted.slice(-7)];
  }, [trained]);

  const livePrediction = useMemo(
    () => trained.predictOne(inputText),
    [trained, inputText]
  );

  const handleAddSample = () => {
    if (!inputText.trim()) return;
    setReviews((prev) => [
      ...prev,
      { id: prev.length + 1, label: newLabel, text: inputText.trim() },
    ]);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Text Classification — Review Sentiment (TF-IDF + Logistic Regression)
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            {reviews.length} curated reviews · {trained.trainSamples.length} train /{' '}
            {trained.testSamples.length} test · 1–2 n-gram TF-IDF ({trained.vocab.length} terms)
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center gap-2 text-slate-600">
            Inverse Reg (C = {regC.toFixed(1)})
            <input
              type="range"
              min={0.2}
              max={6}
              step={0.2}
              value={regC}
              onChange={(e) => setRegC(Number(e.target.value))}
              className="w-24 accent-blue-600"
            />
          </label>
          <label className="flex items-center gap-2 text-slate-600">
            Test split
            <select
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
              className="border border-slate-200 bg-white rounded-md px-2.5 py-1.5 font-mono"
            >
              <option value={0.2}>20%</option>
              <option value={0.25}>25%</option>
              <option value={0.35}>35%</option>
            </select>
          </label>
        </div>
      </div>

      <div className="bg-slate-100 border border-slate-200 rounded-lg px-4 py-3 text-xs text-slate-600">
        Repository note: <span className="font-mono">train.csv</span> and{' '}
        <span className="font-mono">test.csv</span> in the imported repo are Git LFS
        pointers (411 MB / 27 MB) and DistilBERT weights require PyTorch/GPU. This module
        runs the exact TF-IDF (1–2 n-gram) + Logistic Regression baseline on a bundled
        sample corpus so training and live inference work directly in the browser.
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          label="Test Accuracy"
          value={`${(metrics.accuracy * 100).toFixed(1)}%`}
          detail={`${trained.testSamples.length} held-out reviews`}
        />
        <MetricTile
          label="ROC-AUC"
          value={metrics.rocAuc.toFixed(3)}
          detail="TF-IDF baseline"
        />
        <MetricTile
          label="F1 Score"
          value={metrics.f1.toFixed(3)}
          detail={`Precision ${metrics.precision.toFixed(2)} · Recall ${metrics.recall.toFixed(2)}`}
        />
        <MetricTile
          label="Vocabulary Size"
          value={trained.vocab.length.toString()}
          unit="1-2 grams"
          detail="Stop-word filtered"
        />
      </div>

      <ExecutionPerformancePanel
        execution={trained.execution}
        iterationLabel="GD Iterations"
        lossLabel="L2 Log-Loss (Init → Final)"
        rateLabel="Loss Convergence"
      />

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-5 space-y-6">
          <ConfusionMatrixCard
            matrix={metrics.confusionMatrix}
            labels={['Negative (0)', 'Positive (1)']}
            title="Test Set Confusion Matrix"
          />

          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
            <h4 className="text-sm font-semibold text-slate-900">
              Try Your Own Review
            </h4>
            <textarea
              rows={3}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="w-full border border-slate-200 rounded-md p-2.5 text-sm text-slate-900"
              placeholder="Write a restaurant or product review..."
            />

            <div className="flex items-center justify-between pt-2 border-t border-slate-200">
              <div>
                <div className="text-[11px] uppercase tracking-wider text-slate-500">
                  Predicted Sentiment
                </div>
                <div
                  className={`mt-0.5 text-base font-semibold ${
                    livePrediction.label === 1
                      ? 'text-emerald-700'
                      : 'text-rose-600'
                  }`}
                >
                  {livePrediction.label === 1
                    ? 'Positive Review'
                    : 'Negative Review'}
                </div>
              </div>
              <div className="text-right">
                <div className="text-[11px] uppercase tracking-wider text-slate-500">
                  Positive Confidence
                </div>
                <div className="mt-0.5 text-2xl font-mono font-semibold tabular-nums text-slate-900">
                  {(livePrediction.probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {livePrediction.topTokens.length > 0 && (
              <div className="pt-2 border-t border-slate-100">
                <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
                  Matched N-gram Contributions
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono">
                  {livePrediction.topTokens.map((t) => (
                    <span
                      key={t.token}
                      className={
                        t.weight >= 0 ? 'text-emerald-700' : 'text-rose-600'
                      }
                    >
                      {t.token} ({t.weight >= 0 ? '+' : ''}
                      {t.weight.toFixed(2)})
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="pt-3 border-t border-slate-100 flex items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-slate-500">Add to corpus as:</span>
                <select
                  value={newLabel}
                  onChange={(e) => setNewLabel(Number(e.target.value) as 0 | 1)}
                  className="border border-slate-200 rounded px-2 py-1"
                >
                  <option value={1}>Positive</option>
                  <option value={0}>Negative</option>
                </select>
              </div>
              <button
                type="button"
                onClick={handleAddSample}
                className="px-3 py-1.5 font-medium bg-slate-900 text-white rounded-md hover:bg-slate-800 whitespace-nowrap shrink-0"
              >
                Add & Retrain
              </button>
            </div>
          </div>
        </div>

        <div className="xl:col-span-7">
          <CoefficientBarChart
            title="Top Weighted Positive & Negative N-grams"
            subtitle="TF-IDF Logistic Regression weights"
            items={topFeatures}
          />
        </div>
      </div>
    </div>
  );
};
