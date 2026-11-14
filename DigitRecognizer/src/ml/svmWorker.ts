import {
  BinaryClassifier,
  ClassEvaluationMetric,
  HydratedSvmModel,
  PcaBasis,
  SvmKernelType,
  WorkerTrainRequest,
} from './svmTypes';

const MNIST_DIM = 784;
const PCA_COMPONENTS = 50;

function postProgress(jobId: number, percent: number, log?: string): void {
  self.postMessage({ type: 'progress', jobId, percent, log });
}

function orthonormalizeCols(mat: Float64Array, rows: number, cols: number): void {
  for (let c = 0; c < cols; c++) {
    const cOff = c * rows;
    for (let prev = 0; prev < c; prev++) {
      const pOff = prev * rows;
      let dot = 0;
      for (let r = 0; r < rows; r++) dot += mat[cOff + r] * mat[pOff + r];
      for (let r = 0; r < rows; r++) mat[cOff + r] -= dot * mat[pOff + r];
    }
    let norm = 0;
    for (let r = 0; r < rows; r++) norm += mat[cOff + r] * mat[cOff + r];
    norm = Math.sqrt(norm) || 1;
    const inv = 1.0 / norm;
    for (let r = 0; r < rows; r++) mat[cOff + r] *= inv;
  }
}

function fitAndTransformPca(
  trainImages: Float32Array,
  trainCount: number,
  testImages: Float32Array,
  testCount: number,
  jobId: number
): { pca: PcaBasis; trainProjected: Float32Array; testProjected: Float32Array } {
  postProgress(jobId, 6, `Computing pixel mean across ${trainCount} samples...`);

  const mean = new Float32Array(MNIST_DIM);
  for (let i = 0; i < trainCount; i++) {
    const off = i * MNIST_DIM;
    for (let j = 0; j < MNIST_DIM; j++) mean[j] += trainImages[off + j];
  }
  const invN = 1.0 / trainCount;
  for (let j = 0; j < MNIST_DIM; j++) mean[j] *= invN;

  const fitCount = Math.min(trainCount, 1500);
  const stride = Math.max(1, Math.floor(trainCount / fitCount));
  const centeredFit = new Float32Array(fitCount * MNIST_DIM);
  for (let i = 0; i < fitCount; i++) {
    const srcOff = ((i * stride) % trainCount) * MNIST_DIM;
    const dstOff = i * MNIST_DIM;
    for (let j = 0; j < MNIST_DIM; j++) centeredFit[dstOff + j] = trainImages[srcOff + j] - mean[j];
  }

  postProgress(jobId, 11, `Fitting ${PCA_COMPONENTS}-component PCA via subspace iteration...`);

  const q = new Float64Array(MNIST_DIM * PCA_COMPONENTS);
  let seed = 1337;
  for (let i = 0; i < q.length; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    q[i] = (seed / 4294967296 - 0.5) * 2;
  }
  orthonormalizeCols(q, MNIST_DIM, PCA_COMPONENTS);

  const y = new Float64Array(fitCount * PCA_COMPONENTS);
  const z = new Float64Array(MNIST_DIM * PCA_COMPONENTS);

  for (let iter = 0; iter < 4; iter++) {
    y.fill(0);
    for (let i = 0; i < fitCount; i++) {
      const xOff = i * MNIST_DIM;
      const yOff = i * PCA_COMPONENTS;
      for (let c = 0; c < PCA_COMPONENTS; c++) {
        const qOff = c * MNIST_DIM;
        let dot = 0;
        for (let j = 0; j < MNIST_DIM; j++) dot += centeredFit[xOff + j] * q[qOff + j];
        y[yOff + c] = dot;
      }
    }

    z.fill(0);
    for (let i = 0; i < fitCount; i++) {
      const xOff = i * MNIST_DIM;
      const yOff = i * PCA_COMPONENTS;
      for (let c = 0; c < PCA_COMPONENTS; c++) {
        const yVal = y[yOff + c];
        const zOff = c * MNIST_DIM;
        for (let j = 0; j < MNIST_DIM; j++) z[zOff + j] += centeredFit[xOff + j] * yVal;
      }
    }

    q.set(z);
    orthonormalizeCols(q, MNIST_DIM, PCA_COMPONENTS);
  }

  const components = new Float32Array(PCA_COMPONENTS * MNIST_DIM);
  for (let c = 0; c < PCA_COMPONENTS; c++) {
    const off = c * MNIST_DIM;
    for (let j = 0; j < MNIST_DIM; j++) components[off + j] = q[off + j];
  }

  const projectBatch = (images: Float32Array, count: number): Float32Array => {
    const projected = new Float32Array(count * PCA_COMPONENTS);
    const centered = new Float32Array(MNIST_DIM);
    for (let i = 0; i < count; i++) {
      const srcOff = i * MNIST_DIM;
      for (let j = 0; j < MNIST_DIM; j++) centered[j] = images[srcOff + j] - mean[j];
      const dstOff = i * PCA_COMPONENTS;
      for (let c = 0; c < PCA_COMPONENTS; c++) {
        const compOff = c * MNIST_DIM;
        let dot = 0;
        for (let j = 0; j < MNIST_DIM; j++) dot += centered[j] * components[compOff + j];
        projected[dstOff + c] = dot;
      }
    }
    return projected;
  };

  const trainProjected = projectBatch(trainImages, trainCount);
  const testProjected = projectBatch(testImages, testCount);

  postProgress(jobId, 18, `PCA complete: 784 features -> ${PCA_COMPONENTS} components.`);
  return {
    pca: { numComponents: PCA_COMPONENTS, mean, components },
    trainProjected,
    testProjected,
  };
}

function buildGramMatrix(
  features: Float32Array,
  n: number,
  dim: number,
  kernel: SvmKernelType,
  gamma: number,
  jobId: number
): Float32Array {
  const gram = new Float32Array(n * n);
  const norms = new Float32Array(n);

  if (kernel === 'rbf') {
    for (let i = 0; i < n; i++) {
      const off = i * dim;
      let s = 0;
      for (let d = 0; d < dim; d++) {
        const v = features[off + d];
        s += v * v;
      }
      norms[i] = s;
    }
  }

  const reportStep = Math.max(1, Math.floor(n / 6));
  for (let i = 0; i < n; i++) {
    const offI = i * dim;
    const rowI = i * n;
    for (let j = i; j < n; j++) {
      const offJ = j * dim;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += features[offI + d] * features[offJ + d];
      let kval: number;
      if (kernel === 'linear') {
        kval = dot;
      } else if (kernel === 'poly') {
        kval = Math.pow(gamma * dot + 1.0, 3);
      } else {
        const sqDist = Math.max(0, norms[i] + norms[j] - 2.0 * dot);
        kval = Math.exp(-gamma * sqDist);
      }
      gram[rowI + j] = kval;
      gram[j * n + i] = kval;
    }
    if (i > 0 && i % reportStep === 0) {
      postProgress(jobId, 20 + Math.round((i / n) * 18));
    }
  }
  return gram;
}

function trainBinarySmo(
  gram: Float32Array,
  labels: Int8Array,
  n: number,
  C: number
): { alphas: Float32Array; bias: number } {
  const alphas = new Float32Array(n);
  const fCache = new Float32Array(n);
  let bias = 0;
  const tol = 5e-3;
  const eps = 1e-4;
  const maxPasses = 6;
  const maxSteps = n * 45;

  let passes = 0;
  let steps = 0;
  let examineAll = true;

  while ((passes < maxPasses || examineAll) && steps < maxSteps) {
    let numChanged = 0;

    for (let i = 0; i < n; i++) {
      if (!examineAll && (alphas[i] <= 0 || alphas[i] >= C)) continue;
      steps++;
      if (steps >= maxSteps) break;

      const yi = labels[i];
      const Ei = fCache[i] - yi;
      const r = Ei * yi;

      if ((r < -tol && alphas[i] < C) || (r > tol && alphas[i] > 0)) {
        let j = -1;
        let maxDelta = -1;
        for (let k = 0; k < n; k++) {
          if (k === i) continue;
          const delta = Math.abs(Ei - (fCache[k] - labels[k]));
          if (delta > maxDelta) {
            maxDelta = delta;
            j = k;
          }
        }
        if (j === -1) continue;

        const yj = labels[j];
        const Ej = fCache[j] - yj;
        const alphaIOld = alphas[i];
        const alphaJOld = alphas[j];

        let L: number;
        let H: number;
        if (yi !== yj) {
          L = Math.max(0, alphaJOld - alphaIOld);
          H = Math.min(C, C + alphaJOld - alphaIOld);
        } else {
          L = Math.max(0, alphaIOld + alphaJOld - C);
          H = Math.min(C, alphaIOld + alphaJOld);
        }
        if (L === H) continue;

        const kii = gram[i * n + i];
        const kjj = gram[j * n + j];
        const kij = gram[i * n + j];
        const eta = 2.0 * kij - kii - kjj;
        if (eta >= -1e-9) continue;

        let alphaJNew = alphaJOld - (yj * (Ei - Ej)) / eta;
        if (alphaJNew > H) alphaJNew = H;
        else if (alphaJNew < L) alphaJNew = L;

        if (Math.abs(alphaJNew - alphaJOld) < eps) continue;

        const alphaINew = alphaIOld + yi * yj * (alphaJOld - alphaJNew);

        const b1 =
          bias -
          Ei -
          yi * (alphaINew - alphaIOld) * kii -
          yj * (alphaJNew - alphaJOld) * kij;
        const b2 =
          bias -
          Ej -
          yi * (alphaINew - alphaIOld) * kij -
          yj * (alphaJNew - alphaJOld) * kjj;

        let bNew: number;
        if (alphaINew > 0 && alphaINew < C) bNew = b1;
        else if (alphaJNew > 0 && alphaJNew < C) bNew = b2;
        else bNew = 0.5 * (b1 + b2);

        const dI = (alphaINew - alphaIOld) * yi;
        const dJ = (alphaJNew - alphaJOld) * yj;
        const dB = bNew - bias;

        const rowI = i * n;
        const rowJ = j * n;
        for (let k = 0; k < n; k++) {
          fCache[k] += dI * gram[rowI + k] + dJ * gram[rowJ + k] + dB;
        }

        alphas[i] = alphaINew;
        alphas[j] = alphaJNew;
        bias = bNew;
        numChanged++;
      }
    }

    if (examineAll) {
      examineAll = false;
      if (numChanged === 0) passes++;
      else passes = 0;
    } else if (numChanged === 0) {
      examineAll = true;
      passes++;
    } else {
      passes = 0;
    }
  }

  return { alphas, bias };
}

function evalKernelPair(
  bufA: Float32Array,
  offA: number,
  bufB: Float32Array,
  offB: number,
  dim: number,
  kernel: SvmKernelType,
  gamma: number
): number {
  if (kernel === 'linear') {
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += bufA[offA + d] * bufB[offB + d];
    return dot;
  }
  if (kernel === 'poly') {
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += bufA[offA + d] * bufB[offB + d];
    return Math.pow(gamma * dot + 1.0, 3);
  }
  let sqDist = 0;
  for (let d = 0; d < dim; d++) {
    const diff = bufA[offA + d] - bufB[offB + d];
    sqDist += diff * diff;
  }
  return Math.exp(-gamma * sqDist);
}

self.onmessage = (event: MessageEvent<WorkerTrainRequest>) => {
  const req = event.data;
  if (req.type !== 'train') return;

  const { jobId, trainImages, trainLabels, testImages, testLabels, config } = req;
  const startTime = performance.now();

  try {
    const n = trainLabels.length;
    const testCount = testLabels.length;

    postProgress(
      jobId,
      2,
      `Starting ${config.kernel.toUpperCase()} SVM (N=${n}, C=${config.C}, γ=${config.gamma}, PCA=${config.usePca ? '50' : '784'})`
    );

    let pca: PcaBasis | null = null;
    let trainFeatures: Float32Array;
    let testFeatures: Float32Array;
    let dim: number;

    if (config.usePca) {
      const res = fitAndTransformPca(trainImages, n, testImages, testCount, jobId);
      pca = res.pca;
      trainFeatures = res.trainProjected;
      testFeatures = res.testProjected;
      dim = PCA_COMPONENTS;
    } else {
      trainFeatures = trainImages.subarray(0, n * MNIST_DIM);
      testFeatures = testImages;
      dim = MNIST_DIM;
      postProgress(jobId, 16, `Using full 784 raw normalized pixel features (PCA disabled).`);
    }

    postProgress(jobId, 20, `Building ${n}×${n} ${config.kernel.toUpperCase()} Gram matrix...`);
    const gram = buildGramMatrix(trainFeatures, n, dim, config.kernel, config.gamma, jobId);
    postProgress(jobId, 40, `Gram matrix ready. Running One-vs-Rest SMO optimization...`);

    const rawAlphasList: Float32Array[] = [];
    const biases: number[] = [];
    const binaryLabels = new Int8Array(n);
    const globalSvMap = new Map<number, number>();

    for (let digit = 0; digit < 10; digit++) {
      const tClassStart = performance.now();
      for (let i = 0; i < n; i++) {
        binaryLabels[i] = trainLabels[i] === digit ? 1 : -1;
      }

      const { alphas, bias } = trainBinarySmo(gram, binaryLabels, n, config.C);
      rawAlphasList.push(alphas);
      biases.push(bias);

      let svCount = 0;
      for (let i = 0; i < n; i++) {
        if (alphas[i] > 1e-4) {
          svCount++;
          if (!globalSvMap.has(i)) globalSvMap.set(i, globalSvMap.size);
        }
      }

      const elapsed = Math.round(performance.now() - tClassStart);
      postProgress(
        jobId,
        40 + Math.round(((digit + 1) / 10) * 48),
        `Digit ${digit} vs Rest: ${svCount} SVs (${elapsed} ms)`
      );
    }

    const numUniqueSVs = globalSvMap.size;
    const supportVectors = new Float32Array(numUniqueSVs * dim);
    for (const [trainIdx, svId] of globalSvMap.entries()) {
      supportVectors.set(trainFeatures.subarray(trainIdx * dim, trainIdx * dim + dim), svId * dim);
    }

    const classifiers: BinaryClassifier[] = [];
    for (let digit = 0; digit < 10; digit++) {
      const alphas = rawAlphasList[digit];
      const indices: number[] = [];
      const coefs: number[] = [];
      for (let i = 0; i < n; i++) {
        if (alphas[i] > 1e-4) {
          const svId = globalSvMap.get(i)!;
          const y = trainLabels[i] === digit ? 1 : -1;
          indices.push(svId);
          coefs.push(alphas[i] * y);
        }
      }
      classifiers.push({
        targetDigit: digit,
        svIndices: Uint32Array.from(indices),
        dualCoefs: Float32Array.from(coefs),
        bias: biases[digit],
      });
    }

    postProgress(jobId, 92, `Evaluating SVM on ${testCount} held-out test samples...`);

    const confusionMatrix = Array.from({ length: 10 }, () => Array<number>(10).fill(0));
    const testPredictions = new Uint8Array(testCount);
    const testProbabilities = new Float32Array(testCount * 10);
    const kernelCache = new Float32Array(numUniqueSVs);
    const kernelValid = new Uint8Array(numUniqueSVs);
    const scores = new Array<number>(10);
    let correct = 0;

    const evalStart = performance.now();

    for (let t = 0; t < testCount; t++) {
      const tOff = t * dim;
      kernelValid.fill(0);
      let bestDigit = 0;
      let bestScore = -Infinity;

      for (let digit = 0; digit < 10; digit++) {
        const clf = classifiers[digit];
        let score = clf.bias;
        const svIndices = clf.svIndices;
        const dualCoefs = clf.dualCoefs;
        for (let s = 0; s < svIndices.length; s++) {
          const svId = svIndices[s];
          let kVal: number;
          if (kernelValid[svId] === 1) {
            kVal = kernelCache[svId];
          } else {
            kVal = evalKernelPair(
              supportVectors,
              svId * dim,
              testFeatures,
              tOff,
              dim,
              config.kernel,
              config.gamma
            );
            kernelCache[svId] = kVal;
            kernelValid[svId] = 1;
          }
          score += dualCoefs[s] * kVal;
        }
        scores[digit] = score;
        if (score > bestScore) {
          bestScore = score;
          bestDigit = digit;
        }
      }

      // Softmax over decision margins
      let expSum = 0;
      const pOff = t * 10;
      for (let d = 0; d < 10; d++) {
        const e = Math.exp((scores[d] - bestScore) * 2.0);
        testProbabilities[pOff + d] = e;
        expSum += e;
      }
      for (let d = 0; d < 10; d++) testProbabilities[pOff + d] /= expSum;

      const trueDigit = testLabels[t];
      testPredictions[t] = bestDigit;
      confusionMatrix[trueDigit][bestDigit]++;
      if (bestDigit === trueDigit) correct++;
    }

    const avgPredMs = (performance.now() - evalStart) / testCount;

    const classMetrics: ClassEvaluationMetric[] = [];
    for (let d = 0; d < 10; d++) {
      const tp = confusionMatrix[d][d];
      let fp = 0;
      let fn = 0;
      let support = 0;
      for (let k = 0; k < 10; k++) {
        support += confusionMatrix[d][k];
        if (k !== d) {
          fn += confusionMatrix[d][k];
          fp += confusionMatrix[k][d];
        }
      }
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      classMetrics.push({ digit: d, precision, recall, f1Score, support });
    }

    const testAccuracy = correct / testCount;
    const trainTimeMs = performance.now() - startTime;

    postProgress(
      jobId,
      100,
      `SVM complete in ${(trainTimeMs / 1000).toFixed(2)}s — Accuracy: ${(testAccuracy * 100).toFixed(2)}% (${avgPredMs.toFixed(2)} ms/digit)`
    );

    const model: HydratedSvmModel = {
      config,
      featureDim: dim,
      pca,
      supportVectors,
      numUniqueSVs,
      classifiers,
      evaluation: {
        testAccuracy,
        confusionMatrix,
        classMetrics,
        testPredictions,
        testProbabilities,
        avgPredMs,
      },
      trainTimeMs,
      trainedAtIso: new Date().toISOString(),
    };

    self.postMessage({ type: 'complete', jobId, model });
  } catch (err) {
    self.postMessage({
      type: 'error',
      jobId,
      message: err instanceof Error ? err.message : 'SVM Web Worker training failed.',
    });
  }
};
