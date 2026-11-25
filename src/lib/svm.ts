import { mulberry32 } from './csv';

export interface Point2D {
  x: number[];
  y: number; // 0 or 1
}

function randNormal(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function generateClassification2D(
  nSamples = 200,
  classSep = 1.5,
  flipY = 0.05,
  seed = 42
): Point2D[] {
  const rng = mulberry32(seed);
  const pts: Point2D[] = [];
  for (let i = 0; i < nSamples; i++) {
    const baseLabel = i < nSamples / 2 ? 0 : 1;
    const shift = baseLabel === 0 ? -classSep : classSep;
    const x1 = shift * 0.8 + randNormal(rng) * 0.85;
    const x2 = shift * 0.55 + randNormal(rng) * 0.9;
    const y = rng() < flipY ? 1 - baseLabel : baseLabel;
    pts.push({ x: [x1, x2], y });
  }
  return pts;
}

export function generateMoons2D(
  nSamples = 200,
  noise = 0.15,
  seed = 42
): Point2D[] {
  const rng = mulberry32(seed);
  const pts: Point2D[] = [];
  const half = Math.floor(nSamples / 2);
  for (let i = 0; i < half; i++) {
    const theta = (Math.PI * i) / half;
    pts.push({
      x: [
        Math.cos(theta) + randNormal(rng) * noise,
        Math.sin(theta) + randNormal(rng) * noise,
      ],
      y: 0,
    });
  }
  for (let i = 0; i < nSamples - half; i++) {
    const theta = (Math.PI * i) / (nSamples - half);
    pts.push({
      x: [
        1 - Math.cos(theta) + randNormal(rng) * noise,
        1 - Math.sin(theta) - 0.5 + randNormal(rng) * noise,
      ],
      y: 1,
    });
  }
  return pts;
}

import type { ExecutionMetrics } from './mlMath';

export interface SVMModel {
  kernel: 'linear' | 'rbf';
  gamma: number;
  C: number;
  b: number;
  supportVectors: number[][];
  supportLabels: number[]; // -1 or +1
  supportAlphas: number[];
  supportIndices: number[];
  decisionFunction: (x: number[]) => number;
  predict: (x: number[]) => number; // 0 or 1
  execution: ExecutionMetrics;
}

function kernelVal(
  a: number[],
  b: number[],
  kernel: 'linear' | 'rbf',
  gamma: number
): number {
  if (kernel === 'linear') {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }
  let sqDist = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sqDist += d * d;
  }
  return Math.exp(-gamma * sqDist);
}

/**
 * Platt Sequential Minimal Optimization (SMO) SVM solver for binary classification.
 */
export function fitSVM(
  X: number[][],
  y01: number[],
  options: {
    kernel: 'linear' | 'rbf';
    C: number;
    gamma?: number;
    tol?: number;
    maxPasses?: number;
    seed?: number;
  }
): SVMModel {
  const t0 = performance.now();
  const n = X.length;
  const y = y01.map((v) => (v === 1 ? 1 : -1));
  const C = options.C;
  const tol = options.tol ?? 1e-3;
  const maxPasses = options.maxPasses ?? 25;
  const rng = mulberry32(options.seed ?? 42);

  // Compute default 'scale' gamma = 1 / (n_features * X.var()) if not specified
  let gamma = options.gamma ?? 1.0;
  if (options.gamma === undefined && n > 0) {
    const d = X[0].length;
    let meanAll = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        meanAll += X[i][j];
        count++;
      }
    }
    meanAll /= count || 1;
    let varAll = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        const diff = X[i][j] - meanAll;
        varAll += diff * diff;
      }
    }
    varAll /= count || 1;
    gamma = 1 / Math.max(1e-6, d * varAll);
  }

  // Precompute kernel matrix
  const K: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const kv = kernelVal(X[i], X[j], options.kernel, gamma);
      K[i][j] = kv;
      K[j][i] = kv;
    }
  }

  const alphas = Array(n).fill(0);
  let b = 0;
  let passes = 0;
  let totalIters = 0;
  let alphaUpdates = 0;
  const maxTotalIters = 1200;
  const lossHistory: number[] = [];

  const computePrimalHingeLoss = () => {
    let hingeSum = 0;
    let normSq = 0;
    for (let i = 0; i < n; i++) {
      let fx = b;
      for (let k = 0; k < n; k++) {
        if (alphas[k] > 0) fx += alphas[k] * y[k] * K[k][i];
      }
      hingeSum += Math.max(0, 1 - y[i] * fx);
      if (alphas[i] > 0) {
        for (let k = 0; k < n; k++) {
          if (alphas[k] > 0) normSq += alphas[i] * alphas[k] * y[i] * y[k] * K[i][k];
        }
      }
    }
    return (0.5 * normSq + C * hingeSum) / (n || 1);
  };

  lossHistory.push(computePrimalHingeLoss());

  while (passes < maxPasses && totalIters < maxTotalIters) {
    let numChanged = 0;
    totalIters++;

    for (let i = 0; i < n; i++) {
      let Ei = b - y[i];
      for (let k = 0; k < n; k++) {
        if (alphas[k] > 0) Ei += alphas[k] * y[k] * K[k][i];
      }

      if (
        (y[i] * Ei < -tol && alphas[i] < C) ||
        (y[i] * Ei > tol && alphas[i] > 0)
      ) {
        let j = Math.floor(rng() * (n - 1));
        if (j >= i) j++;

        let Ej = b - y[j];
        for (let k = 0; k < n; k++) {
          if (alphas[k] > 0) Ej += alphas[k] * y[k] * K[k][j];
        }

        const aiOld = alphas[i];
        const ajOld = alphas[j];

        let L = 0;
        let H = C;
        if (y[i] !== y[j]) {
          L = Math.max(0, ajOld - aiOld);
          H = Math.min(C, C + ajOld - aiOld);
        } else {
          L = Math.max(0, aiOld + ajOld - C);
          H = Math.min(C, aiOld + ajOld);
        }
        if (L === H) continue;

        const eta = 2 * K[i][j] - K[i][i] - K[j][j];
        if (eta >= 0) continue;

        let ajNew = ajOld - (y[j] * (Ei - Ej)) / eta;
        if (ajNew > H) ajNew = H;
        else if (ajNew < L) ajNew = L;

        if (Math.abs(ajNew - ajOld) < 1e-5) continue;

        const aiNew = aiOld + y[i] * y[j] * (ajOld - ajNew);
        alphas[i] = aiNew;
        alphas[j] = ajNew;

        const b1 =
          b -
          Ei -
          y[i] * (aiNew - aiOld) * K[i][i] -
          y[j] * (ajNew - ajOld) * K[i][j];
        const b2 =
          b -
          Ej -
          y[i] * (aiNew - aiOld) * K[i][j] -
          y[j] * (ajNew - ajOld) * K[j][j];

        if (aiNew > 0 && aiNew < C) b = b1;
        else if (ajNew > 0 && ajNew < C) b = b2;
        else b = (b1 + b2) / 2;

        numChanged++;
        alphaUpdates++;
      }
    }

    if (totalIters % 10 === 0 || numChanged === 0) {
      lossHistory.push(computePrimalHingeLoss());
    }

    if (numChanged === 0) passes++;
    else passes = 0;
  }

  const supportIndices: number[] = [];
  const supportVectors: number[][] = [];
  const supportLabels: number[] = [];
  const supportAlphas: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alphas[i] > 1e-5) {
      supportIndices.push(i);
      supportVectors.push(X[i]);
      supportLabels.push(y[i]);
      supportAlphas.push(alphas[i]);
    }
  }

  const initialLoss = lossHistory[0];
  const finalLoss = lossHistory[lossHistory.length - 1];
  const lossReductionPct =
    initialLoss > 0 ? ((initialLoss - finalLoss) / initialLoss) * 100 : 0;
  const trainingTimeMs = performance.now() - t0;

  const decisionFunction = (pt: number[]) => {
    let sum = b;
    for (let k = 0; k < supportVectors.length; k++) {
      sum +=
        supportAlphas[k] *
        supportLabels[k] *
        kernelVal(supportVectors[k], pt, options.kernel, gamma);
    }
    return sum;
  };

  return {
    kernel: options.kernel,
    gamma,
    C,
    b,
    supportVectors,
    supportLabels,
    supportAlphas,
    supportIndices,
    decisionFunction,
    predict: (pt: number[]) => (decisionFunction(pt) >= 0 ? 1 : 0),
    execution: {
      solver: `Platt SMO (${options.kernel.toUpperCase()} kernel)`,
      trainingTimeMs,
      iterations: totalIters,
      converged: passes >= maxPasses,
      initialLoss,
      finalLoss,
      lossReductionPct,
      convergenceRate:
        totalIters > 0 ? (initialLoss - finalLoss) / totalIters : 0,
      lossHistory,
      note: `${alphaUpdates} α-pair updates · ${supportVectors.length}/${n} support vectors (${((supportVectors.length / (n || 1)) * 100).toFixed(0)}%)`,
    },
  };
}

export interface ProteinSample {
  id: string;
  sequence: string;
  label: number; // 1 = positive remote homolog, 0 = negative
  split: 'train' | 'test';
  family: string;
}

const AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY';

function buildSequence(
  rng: () => number,
  length: number,
  motifs: string[],
  motifCount: number
): string {
  const chars: string[] = [];
  for (let i = 0; i < length; i++) {
    chars.push(AMINO_ACIDS[Math.floor(rng() * AMINO_ACIDS.length)]);
  }
  for (let m = 0; m < motifCount; m++) {
    const motif = motifs[Math.floor(rng() * motifs.length)];
    const pos = Math.floor(rng() * Math.max(1, length - motif.length));
    for (let k = 0; k < motif.length; k++) {
      chars[pos + k] = motif[k];
    }
  }
  return chars.join('');
}

export const SCOP_FAMILIES = [
  {
    id: '1.1.1.1.1',
    name: 'Globins (a.1.1.1)',
    motifs: ['FGH', 'WAK', 'HGK', 'LVR', 'GHH', 'DLD'],
  },
  {
    id: '2.1.1.1.1',
    name: 'Immunoglobulin V-set (b.1.1.1)',
    motifs: ['WYQ', 'YYC', 'FGQ', 'GTK', 'VTV', 'SCQ'],
  },
  {
    id: '3.2.1.1.1',
    name: 'Rossmann-fold Dehydrogenase (c.2.1.1)',
    motifs: ['GAG', 'GIG', 'VLV', 'DVV', 'GYK', 'TAA'],
  },
];

export function generateScopFamilySamples(
  familyId: string,
  seed = 42
): ProteinSample[] {
  const family =
    SCOP_FAMILIES.find((f) => f.id === familyId) ?? SCOP_FAMILIES[0];
  const rng = mulberry32(seed + familyId.length * 13);
  const samples: ProteinSample[] = [];

  // 45 train (12 pos, 33 neg) + 45 test (10 pos, 35 neg) to reflect realistic imbalance
  for (let i = 0; i < 45; i++) {
    const isPos = i < 12;
    const seq = buildSequence(
      rng,
      64 + Math.floor(rng() * 30),
      family.motifs,
      isPos ? 4 + Math.floor(rng() * 3) : Math.floor(rng() * 2)
    );
    samples.push({
      id: `d1${family.id.slice(0, 3).replace(/\./g, '')}t${i + 1}`,
      sequence: seq,
      label: isPos ? 1 : 0,
      split: 'train',
      family: family.name,
    });
  }
  for (let i = 0; i < 45; i++) {
    const isPos = i < 10;
    const seq = buildSequence(
      rng,
      64 + Math.floor(rng() * 30),
      family.motifs,
      isPos ? 3 + Math.floor(rng() * 3) : Math.floor(rng() * 2)
    );
    samples.push({
      id: `d1${family.id.slice(0, 3).replace(/\./g, '')}e${i + 1}`,
      sequence: seq,
      label: isPos ? 1 : 0,
      split: 'test',
      family: family.name,
    });
  }
  return samples;
}

/**
 * Character 2-3gram spectrum representation without IDF, L2-normalized,
 * matching TfidfVectorizer(analyzer='char', ngram_range=(2,3), use_idf=False, norm='l2').
 */
export function buildProteinSpectrumFeatures(
  trainSeqs: string[],
  testSeqs: string[],
  maxVocab = 180
): { XTrain: number[][]; XTest: number[][]; vocab: string[] } {
  const freqMap = new Map<string, number>();
  const extractNgrams = (seq: string) => {
    const s = seq.toUpperCase().replace(/[^A-Z]/g, '');
    const counts = new Map<string, number>();
    for (let n = 2; n <= 3; n++) {
      for (let i = 0; i <= s.length - n; i++) {
        const gram = s.slice(i, i + n);
        counts.set(gram, (counts.get(gram) ?? 0) + 1);
      }
    }
    return counts;
  };

  for (const seq of trainSeqs) {
    const counts = extractNgrams(seq);
    for (const [g, c] of counts.entries()) {
      freqMap.set(g, (freqMap.get(g) ?? 0) + c);
    }
  }

  const vocab = Array.from(freqMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxVocab)
    .map(([g]) => g);
  const vocabIdx = new Map(vocab.map((g, idx) => [g, idx]));

  const vectorize = (seq: string) => {
    const vec = Array(vocab.length).fill(0);
    const counts = extractNgrams(seq);
    let sqSum = 0;
    for (const [g, c] of counts.entries()) {
      const idx = vocabIdx.get(g);
      if (idx !== undefined) {
        vec[idx] = c;
        sqSum += c * c;
      }
    }
    const norm = Math.sqrt(sqSum) || 1;
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    return vec;
  };

  return {
    XTrain: trainSeqs.map(vectorize),
    XTest: testSeqs.map(vectorize),
    vocab,
  };
}
