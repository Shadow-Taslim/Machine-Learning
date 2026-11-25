import React, { useEffect, useMemo, useRef, useState } from 'react';
import { trainTestSplitIndices } from '../lib/csv';
import {
  classificationMetrics,
  fitStandardScaler,
  roc50Score,
} from '../lib/mlMath';
import {
  buildProteinSpectrumFeatures,
  fitSVM,
  generateClassification2D,
  generateMoons2D,
  generateScopFamilySamples,
  SCOP_FAMILIES,
} from '../lib/svm';
import { MetricTile, MultiLineChart } from './Charts';

export const SvmLab: React.FC = () => {
  const [tab, setTab] = useState<'svm2d' | 'protein'>('svm2d');

  // 2D SVM state
  const [datasetType, setDatasetType] = useState<'linear' | 'moons'>('moons');
  const [kernel, setKernel] = useState<'linear' | 'rbf'>('rbf');
  const [C, setC] = useState(1.0);
  const [gamma, setGamma] = useState(1.2);
  const [noise, setNoise] = useState(0.15);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Protein Homology state
  const [familyId, setFamilyId] = useState(SCOP_FAMILIES[0].id);
  const [customSeq, setCustomSeq] = useState(
    'MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG'
  );

  const svm2dResult = useMemo(() => {
    const rawPts =
      datasetType === 'linear'
        ? generateClassification2D(200, 1.5, Math.min(0.2, noise * 0.35), 42)
        : generateMoons2D(200, noise, 42);

    const { trainIdx, testIdx } = trainTestSplitIndices(
      rawPts.length,
      0.2,
      42,
      rawPts.map((p) => p.y)
    );
    const scaler = fitStandardScaler(trainIdx.map((i) => rawPts[i].x));
    const XTrain = scaler.transform(trainIdx.map((i) => rawPts[i].x));
    const yTrain = trainIdx.map((i) => rawPts[i].y);
    const XTest = scaler.transform(testIdx.map((i) => rawPts[i].x));
    const yTest = testIdx.map((i) => rawPts[i].y);

    const model = fitSVM(XTrain, yTrain, {
      kernel,
      C,
      gamma: kernel === 'rbf' ? gamma : undefined,
      seed: 42,
    });

    const trainAcc =
      XTrain.filter((x, i) => model.predict(x) === yTrain[i]).length /
      XTrain.length;
    const testAcc =
      XTest.filter((x, i) => model.predict(x) === yTest[i]).length /
      XTest.length;

    return { model, XTrain, yTrain, XTest, yTest, trainAcc, testAcc };
  }, [datasetType, kernel, C, gamma, noise]);

  useEffect(() => {
    if (tab !== 'svm2d' || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const allX = [...svm2dResult.XTrain, ...svm2dResult.XTest];
    let minX = -2.8,
      maxX = 2.8,
      minY = -2.8,
      maxY = 2.8;
    for (const p of allX) {
      minX = Math.min(minX, p[0] - 0.5);
      maxX = Math.max(maxX, p[0] + 0.5);
      minY = Math.min(minY, p[1] - 0.5);
      maxY = Math.max(maxY, p[1] + 0.5);
    }

    const img = ctx.createImageData(w, h);
    const step = 3; // evaluate 3x3 blocks for instant smoothness
    for (let py = 0; py < h; py += step) {
      const yCoord = maxY - (py / h) * (maxY - minY);
      for (let px = 0; px < w; px += step) {
        const xCoord = minX + (px / w) * (maxX - minX);
        const val = svm2dResult.model.decisionFunction([xCoord, yCoord]);
        const isBoundary = Math.abs(val) < 0.08;
        const isMargin = Math.abs(Math.abs(val) - 1.0) < 0.06;

        let r = 248,
          g = 250,
          b = 252;
        if (isBoundary) {
          r = 15;
          g = 23;
          b = 42;
        } else if (isMargin) {
          r = 100;
          g = 116;
          b = 139;
        } else if (val >= 0) {
          const t = Math.min(1, val / 1.5);
          r = Math.round(252 - t * 14);
          g = Math.round(238 - t * 38);
          b = Math.round(240 - t * 34);
        } else {
          const t = Math.min(1, -val / 1.5);
          r = Math.round(239 - t * 30);
          g = Math.round(246 - t * 20);
          b = Math.round(255);
        }

        for (let dy = 0; dy < step && py + dy < h; dy++) {
          for (let dx = 0; dx < step && px + dx < w; dx++) {
            const idx = ((py + dy) * w + (px + dx)) * 4;
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    const toPx = (x: number) => ((x - minX) / (maxX - minX)) * w;
    const toPy = (y: number) => ((maxY - y) / (maxY - minY)) * h;

    const svSet = new Set(svm2dResult.model.supportIndices);

    // Draw train points
    svm2dResult.XTrain.forEach((pt, idx) => {
      const cx = toPx(pt[0]);
      const cy = toPy(pt[1]);
      const isSV = svSet.has(idx);
      ctx.beginPath();
      ctx.arc(cx, cy, isSV ? 5.5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = svm2dResult.yTrain[idx] === 1 ? '#E11D48' : '#2563EB';
      ctx.fill();
      ctx.lineWidth = isSV ? 2 : 1;
      ctx.strokeStyle = isSV ? '#0F172A' : '#FFFFFF';
      ctx.stroke();
    });

    // Draw test points as diamonds
    svm2dResult.XTest.forEach((pt, idx) => {
      const cx = toPx(pt[0]);
      const cy = toPy(pt[1]);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = svm2dResult.yTest[idx] === 1 ? '#FB7185' : '#60A5FA';
      ctx.strokeStyle = '#0F172A';
      ctx.lineWidth = 1;
      ctx.fillRect(-3.5, -3.5, 7, 7);
      ctx.strokeRect(-3.5, -3.5, 7, 7);
      ctx.restore();
    });
  }, [tab, svm2dResult]);

  // Protein Homology Benchmark computation
  const proteinResult = useMemo(() => {
    const samples = generateScopFamilySamples(familyId, 42);
    const train = samples.filter((s) => s.split === 'train');
    const test = samples.filter((s) => s.split === 'test');
    const yTrain = train.map((s) => s.label);
    const yTest = test.map((s) => s.label);

    const { XTrain, XTest, vocab } = buildProteinSpectrumFeatures(
      train.map((s) => s.sequence),
      test.map((s) => s.sequence)
    );

    const linearSvm = fitSVM(XTrain, yTrain, { kernel: 'linear', C: 1.0 });
    const rbfSvm = fitSVM(XTrain, yTrain, { kernel: 'rbf', C: 2.0 });

    const linearScores = XTest.map((x) => linearSvm.decisionFunction(x));
    const rbfScores = XTest.map((x) => rbfSvm.decisionFunction(x));

    const linearMetrics = classificationMetrics(yTest, linearScores, 0);
    const rbfMetrics = classificationMetrics(yTest, rbfScores, 0);

    const linearRoc50 = roc50Score(yTest, linearScores, 50);
    const rbfRoc50 = roc50Score(yTest, rbfScores, 50);

    const ranked = test
      .map((s, idx) => ({
        id: s.id,
        label: s.label,
        score: rbfScores[idx],
        seqPreview: s.sequence.slice(0, 36) + '...',
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 12);

    // Custom sequence scoring helper
    const scoreCustom = (seq: string) => {
      const { XTest: xCustom } = buildProteinSpectrumFeatures(
        train.map((s) => s.sequence),
        [seq]
      );
      return {
        linear: linearSvm.decisionFunction(xCustom[0]),
        rbf: rbfSvm.decisionFunction(xCustom[0]),
      };
    };

    return {
      vocabSize: vocab.length,
      trainCount: train.length,
      testCount: test.length,
      linearMetrics,
      rbfMetrics,
      linearRoc50,
      rbfRoc50,
      ranked,
      scoreCustom,
    };
  }, [familyId]);

  const customScore = useMemo(
    () => proteinResult.scoreCustom(customSeq),
    [proteinResult, customSeq]
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Support Vector Machines — 2D Kernels & Protein Homology
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Platt SMO solver · Linear & RBF kernels · Character 2–3mer L2 spectrum features
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg text-xs">
          <button
            type="button"
            onClick={() => setTab('svm2d')}
            className={`px-3 py-1.5 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              tab === 'svm2d'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            2D Decision Boundary
          </button>
          <button
            type="button"
            onClick={() => setTab('protein')}
            className={`px-3 py-1.5 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              tab === 'protein'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Protein Remote Homology
          </button>
        </div>
      </div>

      {tab === 'svm2d' ? (
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          <div className="xl:col-span-4 space-y-4">
            <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4 text-xs">
              <h4 className="text-sm font-semibold text-slate-900">
                Dataset & Kernel Parameters
              </h4>

              <div>
                <span className="block text-slate-600 mb-1.5">Dataset Topology</span>
                <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 rounded-md">
                  <button
                    type="button"
                    onClick={() => {
                      setDatasetType('linear');
                      setKernel('linear');
                    }}
                    className={`py-1.5 rounded font-medium ${
                      datasetType === 'linear'
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600'
                    }`}
                  >
                    Linear Clusters
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setDatasetType('moons');
                      setKernel('rbf');
                    }}
                    className={`py-1.5 rounded font-medium ${
                      datasetType === 'moons'
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600'
                    }`}
                  >
                    Two Moons
                  </button>
                </div>
              </div>

              <div>
                <span className="block text-slate-600 mb-1.5">SVM Kernel</span>
                <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 rounded-md">
                  <button
                    type="button"
                    onClick={() => setKernel('linear')}
                    className={`py-1.5 rounded font-medium ${
                      kernel === 'linear'
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600'
                    }`}
                  >
                    Linear
                  </button>
                  <button
                    type="button"
                    onClick={() => setKernel('rbf')}
                    className={`py-1.5 rounded font-medium ${
                      kernel === 'rbf'
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600'
                    }`}
                  >
                    RBF (Gaussian)
                  </button>
                </div>
              </div>

              <label className="block">
                <div className="flex justify-between text-slate-600">
                  <span>Regularization (C)</span>
                  <span className="font-mono">{C.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={10}
                  step={0.1}
                  value={C}
                  onChange={(e) => setC(Number(e.target.value))}
                  className="mt-1 w-full accent-blue-600"
                />
              </label>

              {kernel === 'rbf' && (
                <label className="block">
                  <div className="flex justify-between text-slate-600">
                    <span>RBF Gamma (γ)</span>
                    <span className="font-mono">{gamma.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={5}
                    step={0.1}
                    value={gamma}
                    onChange={(e) => setGamma(Number(e.target.value))}
                    className="mt-1 w-full accent-blue-600"
                  />
                </label>
              )}

              <label className="block">
                <div className="flex justify-between text-slate-600">
                  <span>Sample Noise</span>
                  <span className="font-mono">{noise.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.45}
                  step={0.02}
                  value={noise}
                  onChange={(e) => setNoise(Number(e.target.value))}
                  className="mt-1 w-full accent-blue-600"
                />
              </label>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <MetricTile
                label="Train Accuracy"
                value={`${(svm2dResult.trainAcc * 100).toFixed(1)}%`}
              />
              <MetricTile
                label="Test Accuracy"
                value={`${(svm2dResult.testAcc * 100).toFixed(1)}%`}
              />
              <MetricTile
                label="Support Vectors"
                value={svm2dResult.model.supportVectors.length.toString()}
                detail={`of ${svm2dResult.XTrain.length} train pts`}
              />
              <MetricTile
                label="Bias (b)"
                value={svm2dResult.model.b.toFixed(3)}
              />
            </div>
          </div>

          <div className="xl:col-span-8 bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h4 className="text-sm font-semibold text-slate-900">
                Decision Boundary & Margin Contours
              </h4>
              <div className="text-xs text-slate-500">
                Dark outline = Support Vector · Diamond = Held-out Test Point
              </div>
            </div>
            <canvas
              ref={canvasRef}
              width={640}
              height={440}
              className="w-full h-auto rounded-md border border-slate-200"
            />
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-slate-100 border border-slate-200 rounded-lg px-4 py-3 text-xs text-slate-600 flex flex-wrap items-center justify-between gap-3">
            <span>
              Offline SCOP-style benchmark replica: runs real 2–3mer L2-normalized
              character spectrum features and Platt SMO SVMs on representative curated
              protein families (45 train / 45 test per family) without external HTTP calls.
            </span>
            <select
              value={familyId}
              onChange={(e) => setFamilyId(e.target.value)}
              className="border border-slate-300 bg-white rounded-md px-2.5 py-1.5 text-xs font-medium text-slate-900"
            >
              {SCOP_FAMILIES.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricTile
              label="RBF SVM ROC-AUC"
              value={proteinResult.rbfMetrics.rocAuc.toFixed(3)}
              detail={`Linear: ${proteinResult.linearMetrics.rocAuc.toFixed(3)}`}
            />
            <MetricTile
              label="RBF SVM ROC50"
              value={proteinResult.rbfRoc50.toFixed(3)}
              detail={`Linear: ${proteinResult.linearRoc50.toFixed(3)}`}
            />
            <MetricTile
              label="RBF SVM PR-AUC"
              value={proteinResult.rbfMetrics.prAuc.toFixed(3)}
              detail={`Linear: ${proteinResult.linearMetrics.prAuc.toFixed(3)}`}
            />
            <MetricTile
              label="Spectrum Vocabulary"
              value={proteinResult.vocabSize.toString()}
              unit="2-3mers"
              detail="use_idf=False, norm=l2"
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <MultiLineChart
              title="ROC Curves (Linear vs RBF Spectrum SVM)"
              xLabel="False Positive Rate"
              yLabel="True Positive Rate"
              yDomain={[0, 1]}
              series={[
                {
                  name: `RBF SVM (${proteinResult.rbfMetrics.rocAuc.toFixed(3)})`,
                  color: '#2563EB',
                  data: proteinResult.rbfMetrics.rocCurve.map((p) => ({
                    x: p.fpr,
                    y: p.tpr,
                  })),
                },
                {
                  name: `Linear SVM (${proteinResult.linearMetrics.rocAuc.toFixed(3)})`,
                  color: '#D97706',
                  data: proteinResult.linearMetrics.rocCurve.map((p) => ({
                    x: p.fpr,
                    y: p.tpr,
                  })),
                },
              ]}
            />
            <MultiLineChart
              title="Precision-Recall Curves"
              xLabel="Recall"
              yLabel="Precision"
              yDomain={[0, 1]}
              series={[
                {
                  name: `RBF SVM (${proteinResult.rbfMetrics.prAuc.toFixed(3)})`,
                  color: '#2563EB',
                  data: proteinResult.rbfMetrics.prCurve.map((p) => ({
                    x: p.recall,
                    y: p.precision,
                  })),
                },
                {
                  name: `Linear SVM (${proteinResult.linearMetrics.prAuc.toFixed(3)})`,
                  color: '#D97706',
                  data: proteinResult.linearMetrics.prCurve.map((p) => ({
                    x: p.recall,
                    y: p.precision,
                  })),
                },
              ]}
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <div className="xl:col-span-7 bg-white border border-slate-200 rounded-lg p-5 overflow-x-auto">
              <h4 className="text-sm font-semibold text-slate-900 mb-3">
                Top-Ranked Test Sequences (RBF SVM Decision Score)
              </h4>
              <table className="w-full text-xs text-left">
                <thead className="border-b border-slate-200 text-slate-500 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-2 pr-3">ID</th>
                    <th className="py-2 pr-3">Homolog</th>
                    <th className="py-2 pr-3">Score</th>
                    <th className="py-2">Sequence Preview</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                  {proteinResult.ranked.map((r) => (
                    <tr key={r.id}>
                      <td className="py-2 pr-3 text-slate-900">{r.id}</td>
                      <td
                        className={`py-2 pr-3 font-sans font-medium ${
                          r.label === 1 ? 'text-emerald-700' : 'text-slate-400'
                        }`}
                      >
                        {r.label === 1 ? 'Positive' : 'Negative'}
                      </td>
                      <td className="py-2 pr-3 text-slate-800">
                        {r.score >= 0 ? '+' : ''}
                        {r.score.toFixed(3)}
                      </td>
                      <td className="py-2 text-slate-500">{r.seqPreview}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="xl:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
              <h4 className="text-sm font-semibold text-slate-900">
                Score Custom Protein Sequence
              </h4>
              <textarea
                rows={4}
                value={customSeq}
                onChange={(e) => setCustomSeq(e.target.value)}
                className="w-full border border-slate-200 rounded-md p-2.5 text-xs font-mono text-slate-800"
                placeholder="Paste amino acid sequence (e.g. MVLSEGEWQL...)"
              />
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-200">
                <div>
                  <div className="text-[11px] uppercase tracking-wider text-slate-500">
                    Linear SVM Score
                  </div>
                  <div className="mt-1 text-lg font-mono font-semibold tabular-nums text-slate-900">
                    {customScore.linear >= 0 ? '+' : ''}
                    {customScore.linear.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-wider text-slate-500">
                    RBF SVM Score
                  </div>
                  <div className="mt-1 text-lg font-mono font-semibold tabular-nums text-blue-600">
                    {customScore.rbf >= 0 ? '+' : ''}
                    {customScore.rbf.toFixed(3)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
