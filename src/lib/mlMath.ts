export interface LinearRegressionResult {
  weights: number[];
  intercept: number;
  predict: (X: number[][]) => number[];
  predictOne: (x: number[]) => number;
}

export function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const val = Math.abs(M[r][col]);
      if (val > maxVal) {
        maxVal = val;
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
    }
    const diag = Math.abs(M[col][col]) < 1e-12 ? 1e-12 : M[col][col];
    for (let j = col; j <= n; j++) {
      M[col][j] /= diag;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = col; j <= n; j++) {
        M[r][j] -= factor * M[col][j];
      }
    }
  }
  return M.map((row) => row[n]);
}

export function fitLinearRegression(
  X: number[][],
  y: number[],
  ridge = 1e-8
): LinearRegressionResult {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const dim = d + 1; // bias + features

  const XtX: number[][] = Array.from({ length: dim }, () => Array(dim).fill(0));
  const Xty: number[] = Array(dim).fill(0);

  for (let i = 0; i < n; i++) {
    const row = X[i];
    const yi = y[i];
    Xty[0] += yi;
    XtX[0][0] += 1;
    for (let j = 0; j < d; j++) {
      const xj = row[j];
      XtX[0][j + 1] += xj;
      XtX[j + 1][0] += xj;
      Xty[j + 1] += xj * yi;
      for (let k = j; k < d; k++) {
        const prod = xj * row[k];
        XtX[j + 1][k + 1] += prod;
        if (k !== j) XtX[k + 1][j + 1] += prod;
      }
    }
  }

  for (let j = 1; j < dim; j++) {
    XtX[j][j] += ridge;
  }

  const sol = solveLinearSystem(XtX, Xty);
  const intercept = sol[0];
  const weights = sol.slice(1);

  const predictOne = (x: number[]) =>
    weights.reduce((acc, w, idx) => acc + w * x[idx], intercept);

  return {
    weights,
    intercept,
    predictOne,
    predict: (inputs: number[][]) => inputs.map(predictOne),
  };
}

export function regressionMetrics(yTrue: number[], yPred: number[]) {
  const n = yTrue.length;
  let sumSqErr = 0;
  let sumAbsErr = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i];
    sumSqErr += err * err;
    sumAbsErr += Math.abs(err);
    sumY += yTrue[i];
  }
  const meanY = sumY / n;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const diff = yTrue[i] - meanY;
    ssTot += diff * diff;
  }
  const mse = sumSqErr / n;
  const rmse = Math.sqrt(mse);
  const mae = sumAbsErr / n;
  const r2 = ssTot === 0 ? 0 : 1 - sumSqErr / ssTot;
  return { mse, rmse, mae, r2 };
}

export interface Scaler {
  means: number[];
  stds: number[];
  transform: (X: number[][]) => number[][];
  transformOne: (x: number[]) => number[];
}

export function fitStandardScaler(X: number[][]): Scaler {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const means = Array(d).fill(0);
  const stds = Array(d).fill(1);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) means[j] += X[i][j];
  }
  for (let j = 0; j < d; j++) means[j] /= n || 1;

  const vars = Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const diff = X[i][j] - means[j];
      vars[j] += diff * diff;
    }
  }
  for (let j = 0; j < d; j++) {
    const s = Math.sqrt(vars[j] / (n || 1));
    stds[j] = s < 1e-12 ? 1 : s;
  }

  const transformOne = (x: number[]) =>
    x.map((v, j) => (v - means[j]) / stds[j]);

  return {
    means,
    stds,
    transformOne,
    transform: (inputs: number[][]) => inputs.map(transformOne),
  };
}

export function computeMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

export interface LogisticRegressionResult {
  weights: number[];
  intercept: number;
  predictProba: (X: number[][]) => number[];
  predictProbaOne: (x: number[]) => number;
  predict: (X: number[][], threshold?: number) => number[];
}

export function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

/**
 * L2-regularized Logistic Regression solved via damped Newton-Raphson (IRLS)
 * for small/medium feature counts, falling back to Adam/GD for high dimensions.
 */
export function fitLogisticRegression(
  X: number[][],
  y: number[],
  options: {
    C?: number;
    maxIter?: number;
    classWeight?: 'balanced' | null;
  } = {}
): LogisticRegressionResult {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const C = options.C ?? 1.0;
  const lambda = 1.0 / Math.max(1e-6, C);
  const maxIter = options.maxIter ?? 40;

  let posCount = 0;
  for (let i = 0; i < n; i++) if (y[i] === 1) posCount++;
  const negCount = n - posCount;
  const wPos =
    options.classWeight === 'balanced' && posCount > 0 ? n / (2 * posCount) : 1;
  const wNeg =
    options.classWeight === 'balanced' && negCount > 0 ? n / (2 * negCount) : 1;
  const sampleWeights = y.map((yi) => (yi === 1 ? wPos : wNeg));

  if (d <= 64) {
    // Exact IRLS / Newton-Raphson with L2 regularization
    const dim = d + 1;
    let beta = Array(dim).fill(0);

    for (let iter = 0; iter < maxIter; iter++) {
      const grad = Array(dim).fill(0);
      const H: number[][] = Array.from({ length: dim }, () =>
        Array(dim).fill(0)
      );

      for (let i = 0; i < n; i++) {
        const xi = X[i];
        let z = beta[0];
        for (let j = 0; j < d; j++) z += beta[j + 1] * xi[j];
        const p = sigmoid(z);
        const sw = sampleWeights[i];
        const diff = sw * (p - y[i]);
        const w = Math.max(1e-7, sw * p * (1 - p));

        grad[0] += diff;
        H[0][0] += w;
        for (let j = 0; j < d; j++) {
          const xj = xi[j];
          grad[j + 1] += diff * xj;
          H[0][j + 1] += w * xj;
          H[j + 1][0] += w * xj;
          for (let k = j; k < d; k++) {
            const val = w * xj * xi[k];
            H[j + 1][k + 1] += val;
            if (k !== j) H[k + 1][j + 1] += val;
          }
        }
      }

      H[0][0] += 1e-8;
      for (let j = 1; j < dim; j++) {
        grad[j] += lambda * beta[j];
        H[j][j] += lambda;
      }

      const step = solveLinearSystem(H, grad);
      let maxStep = 0;
      for (let j = 0; j < dim; j++) {
        beta[j] -= step[j];
        maxStep = Math.max(maxStep, Math.abs(step[j]));
      }
      if (maxStep < 1e-6) break;
    }

    const intercept = beta[0];
    const weights = beta.slice(1);
    const predictProbaOne = (x: number[]) =>
      sigmoid(weights.reduce((acc, w, idx) => acc + w * x[idx], intercept));
    return {
      weights,
      intercept,
      predictProbaOne,
      predictProba: (inputs) => inputs.map(predictProbaOne),
      predict: (inputs, threshold = 0.5) =>
        inputs.map((x) => (predictProbaOne(x) >= threshold ? 1 : 0)),
    };
  }

  // High-dimensional sparse/dense fallback via momentum GD
  const weights = Array(d).fill(0);
  let intercept = 0;
  const lr = 0.25;
  const iters = Math.max(250, maxIter);
  for (let iter = 0; iter < iters; iter++) {
    const gradW = Array(d).fill(0);
    let gradB = 0;
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      let z = intercept;
      for (let j = 0; j < d; j++) z += weights[j] * xi[j];
      const p = sigmoid(z);
      const err = sampleWeights[i] * (p - y[i]);
      gradB += err;
      for (let j = 0; j < d; j++) {
        if (xi[j] !== 0) gradW[j] += err * xi[j];
      }
    }
    intercept -= (lr * gradB) / n;
    for (let j = 0; j < d; j++) {
      weights[j] -= (lr * (gradW[j] + lambda * weights[j])) / n;
    }
  }

  const predictProbaOne = (x: number[]) =>
    sigmoid(weights.reduce((acc, w, idx) => acc + w * x[idx], intercept));
  return {
    weights,
    intercept,
    predictProbaOne,
    predictProba: (inputs) => inputs.map(predictProbaOne),
    predict: (inputs, threshold = 0.5) =>
      inputs.map((x) => (predictProbaOne(x) >= threshold ? 1 : 0)),
  };
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: [[number, number], [number, number]]; // [[tn, fp], [fn, tp]]
  rocCurve: { fpr: number; tpr: number }[];
  rocAuc: number;
  prCurve: { recall: number; precision: number }[];
  prAuc: number;
}

export function trapezoidalAuc(xs: number[], ys: number[]): number {
  let area = 0;
  for (let i = 1; i < xs.length; i++) {
    area += ((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])) / 2;
  }
  return Math.abs(area);
}

export function classificationMetrics(
  yTrue: number[],
  yScore: number[],
  threshold = 0.5
): ClassificationMetrics {
  const n = yTrue.length;
  let tn = 0,
    fp = 0,
    fn = 0,
    tp = 0;
  for (let i = 0; i < n; i++) {
    const pred = yScore[i] >= threshold ? 1 : 0;
    const actual = yTrue[i];
    if (actual === 0 && pred === 0) tn++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 1 && pred === 0) fn++;
    else tp++;
  }
  const accuracy = (tp + tn) / (n || 1);
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 =
    precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);

  // ROC & PR Curves
  const pairs = yTrue
    .map((y, i) => ({ y, s: yScore[i] }))
    .sort((a, b) => b.s - a.s);
  const totalPos = pairs.filter((p) => p.y === 1).length || 1;
  const totalNeg = pairs.filter((p) => p.y === 0).length || 1;

  const rocCurve: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  const prCurve: { recall: number; precision: number }[] = [
    { recall: 0, precision: 1 },
  ];

  let cumTp = 0;
  let cumFp = 0;
  let prevRecall = 0;
  let prAuc = 0;

  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].y === 1) cumTp++;
    else cumFp++;
    // Only emit point at threshold change or end
    if (i === pairs.length - 1 || pairs[i + 1].s !== pairs[i].s) {
      const fpr = cumFp / totalNeg;
      const tpr = cumTp / totalPos;
      const rec = tpr;
      const prec = cumTp / (cumTp + cumFp);
      rocCurve.push({ fpr, tpr });
      prCurve.push({ recall: rec, precision: prec });
      prAuc += (rec - prevRecall) * prec;
      prevRecall = rec;
    }
  }

  const rocAuc = trapezoidalAuc(
    rocCurve.map((p) => p.fpr),
    rocCurve.map((p) => p.tpr)
  );

  return {
    accuracy,
    precision,
    recall,
    f1,
    confusionMatrix: [
      [tn, fp],
      [fn, tp],
    ],
    rocCurve,
    rocAuc,
    prCurve,
    prAuc,
  };
}

export function roc50Score(
  yTrue: number[],
  scores: number[],
  maxFalsePositives = 50
): number {
  const nPos = yTrue.reduce((a, b) => a + (b === 1 ? 1 : 0), 0);
  const nNeg = yTrue.length - nPos;
  if (nPos === 0 || nNeg === 0) return 0;

  const order = yTrue
    .map((y, i) => ({ y, s: scores[i] }))
    .sort((a, b) => b.s - a.s);
  const cutoff = Math.min(maxFalsePositives, nNeg);

  const fpCurve = [0];
  const tprCurve = [0];
  let tp = 0;
  let fp = 0;

  for (let i = 0; i < order.length; i++) {
    if (order[i].y === 1) tp++;
    else fp++;
    if (fp <= cutoff) {
      fpCurve.push(fp);
      tprCurve.push(tp / nPos);
    } else {
      const prevFp = fpCurve[fpCurve.length - 1];
      if (prevFp < cutoff) {
        const prevTpr = tprCurve[tprCurve.length - 1];
        const nextTpr = tp / nPos;
        const frac = (cutoff - prevFp) / (fp - prevFp);
        fpCurve.push(cutoff);
        tprCurve.push(prevTpr + frac * (nextTpr - prevTpr));
      }
      break;
    }
  }

  const normFp = fpCurve.map((f) => f / cutoff);
  return trapezoidalAuc(normFp, tprCurve);
}
