import {
  HydratedSvmModel,
  PcaBasis,
  PredictionOutput,
  SvmKernelType,
  SvmTrainConfig,
} from './svmTypes';

export const MODEL_STORAGE_KEY = 'digit_recognizer_svm_model_v2';

export function projectVectorWithPca(pixels784: Float32Array, pca: PcaBasis): Float32Array {
  const { numComponents, mean, components } = pca;
  const out = new Float32Array(numComponents);
  for (let c = 0; c < numComponents; c++) {
    const rowOffset = c * 784;
    let sum = 0;
    for (let j = 0; j < 784; j++) {
      sum += (pixels784[j] - mean[j]) * components[rowOffset + j];
    }
    out[c] = sum;
  }
  return out;
}

export function evalKernel(
  svBuffer: Float32Array,
  svOffset: number,
  x: Float32Array,
  dim: number,
  kernel: SvmKernelType,
  gamma: number
): number {
  if (kernel === 'linear') {
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += svBuffer[svOffset + i] * x[i];
    return dot;
  }
  if (kernel === 'poly') {
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += svBuffer[svOffset + i] * x[i];
    return Math.pow(gamma * dot + 1.0, 3);
  }
  let sqDist = 0;
  for (let i = 0; i < dim; i++) {
    const d = svBuffer[svOffset + i] - x[i];
    sqDist += d * d;
  }
  return Math.exp(-gamma * sqDist);
}

export function scoresToSoftmax(scores: number[], temperature = 2.0): number[] {
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - maxScore) * temperature));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExp);
}

export function predictDigitWithSvm(
  model: HydratedSvmModel,
  pixels28: Float32Array
): PredictionOutput {
  const t0 = performance.now();
  const features = model.pca ? projectVectorWithPca(pixels28, model.pca) : pixels28;
  const dim = model.featureDim;
  const { kernel, gamma } = model.config;
  const svBuf = model.supportVectors;

  const kernelCache = new Float32Array(model.numUniqueSVs);
  const kernelValid = new Uint8Array(model.numUniqueSVs);

  const decisionScores = new Array<number>(10).fill(0);
  for (let digit = 0; digit < 10; digit++) {
    const clf = model.classifiers[digit];
    let score = clf.bias;
    const svIndices = clf.svIndices;
    const dualCoefs = clf.dualCoefs;
    for (let s = 0; s < svIndices.length; s++) {
      const svId = svIndices[s];
      let kVal: number;
      if (kernelValid[svId] === 1) {
        kVal = kernelCache[svId];
      } else {
        kVal = evalKernel(svBuf, svId * dim, features, dim, kernel, gamma);
        kernelCache[svId] = kVal;
        kernelValid[svId] = 1;
      }
      score += dualCoefs[s] * kVal;
    }
    decisionScores[digit] = score;
  }

  const probabilities = scoresToSoftmax(decisionScores, 2.0);
  let predictedDigit = 0;
  for (let d = 1; d < 10; d++) {
    if (decisionScores[d] > decisionScores[predictedDigit]) {
      predictedDigit = d;
    }
  }

  return {
    predictedDigit,
    decisionScores,
    probabilities,
    confidence: probabilities[predictedDigit],
    inferenceMs: performance.now() - t0,
  };
}

export function bytesToBase64(bytes: Uint8Array): string {
  const chunkSize = 0x8000;
  const parts: string[] = [];
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const sub = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    parts.push(String.fromCharCode(...sub));
  }
  return btoa(parts.join(''));
}

export function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function float32ToBase64(arr: Float32Array): string {
  return bytesToBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
}

function base64ToFloat32(b64: string): Float32Array {
  const bytes = base64ToBytes(b64);
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

interface StoredSvmPayload {
  version: number;
  config: SvmTrainConfig;
  featureDim: number;
  numUniqueSVs: number;
  svEncoding: 'float32' | 'uint8';
  supportVectorsB64: string;
  pca: {
    numComponents: number;
    meanB64: string;
    componentsB64: string;
  } | null;
  classifiers: {
    targetDigit: number;
    svIndicesB64: string;
    dualCoefsB64: string;
    bias: number;
  }[];
  evaluation: {
    testAccuracy: number;
    confusionMatrix: number[][];
    classMetrics: HydratedSvmModel['evaluation']['classMetrics'];
    testPredictionsB64: string;
    testProbsUint8B64: string;
    avgPredMs: number;
  };
  trainTimeMs: number;
  trainedAtIso: string;
}

export function saveModelToLocalStorage(
  model: HydratedSvmModel
): { saved: boolean; sizeKb: number; error?: string } {
  try {
    const useUint8Sv = !model.pca;
    let svB64: string;
    if (useUint8Sv) {
      const q = new Uint8Array(model.supportVectors.length);
      for (let i = 0; i < model.supportVectors.length; i++) {
        q[i] = Math.max(0, Math.min(255, Math.round(model.supportVectors[i] * 255)));
      }
      svB64 = bytesToBase64(q);
    } else {
      svB64 = float32ToBase64(model.supportVectors);
    }

    const probsQ = new Uint8Array(model.evaluation.testProbabilities.length);
    for (let i = 0; i < probsQ.length; i++) {
      probsQ[i] = Math.max(0, Math.min(255, Math.round(model.evaluation.testProbabilities[i] * 255)));
    }

    const payload: StoredSvmPayload = {
      version: 2,
      config: model.config,
      featureDim: model.featureDim,
      numUniqueSVs: model.numUniqueSVs,
      svEncoding: useUint8Sv ? 'uint8' : 'float32',
      supportVectorsB64: svB64,
      pca: model.pca
        ? {
            numComponents: model.pca.numComponents,
            meanB64: float32ToBase64(model.pca.mean),
            componentsB64: float32ToBase64(model.pca.components),
          }
        : null,
      classifiers: model.classifiers.map((c) => ({
        targetDigit: c.targetDigit,
        svIndicesB64: bytesToBase64(
          new Uint8Array(c.svIndices.buffer, c.svIndices.byteOffset, c.svIndices.byteLength)
        ),
        dualCoefsB64: float32ToBase64(c.dualCoefs),
        bias: c.bias,
      })),
      evaluation: {
        testAccuracy: model.evaluation.testAccuracy,
        confusionMatrix: model.evaluation.confusionMatrix,
        classMetrics: model.evaluation.classMetrics,
        testPredictionsB64: bytesToBase64(model.evaluation.testPredictions),
        testProbsUint8B64: bytesToBase64(probsQ),
        avgPredMs: model.evaluation.avgPredMs,
      },
      trainTimeMs: model.trainTimeMs,
      trainedAtIso: model.trainedAtIso,
    };

    const raw = JSON.stringify(payload);
    localStorage.setItem(MODEL_STORAGE_KEY, raw);
    return { saved: true, sizeKb: Math.round(raw.length / 1024) };
  } catch (err) {
    return {
      saved: false,
      sizeKb: 0,
      error: err instanceof Error ? err.message : 'localStorage quota exceeded',
    };
  }
}

export function loadModelFromLocalStorage(): HydratedSvmModel | null {
  try {
    const raw = localStorage.getItem(MODEL_STORAGE_KEY);
    if (!raw) return null;
    const payload = JSON.parse(raw) as StoredSvmPayload;
    if (payload.version !== 2) return null;

    let supportVectors: Float32Array;
    if (payload.svEncoding === 'uint8') {
      const q = base64ToBytes(payload.supportVectorsB64);
      supportVectors = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) supportVectors[i] = q[i] / 255.0;
    } else {
      supportVectors = base64ToFloat32(payload.supportVectorsB64);
    }

    const pca: PcaBasis | null = payload.pca
      ? {
          numComponents: payload.pca.numComponents,
          mean: base64ToFloat32(payload.pca.meanB64),
          components: base64ToFloat32(payload.pca.componentsB64),
        }
      : null;

    const classifiers = payload.classifiers.map((c) => {
      const idxBytes = base64ToBytes(c.svIndicesB64);
      const copy = new Uint8Array(idxBytes.length);
      copy.set(idxBytes);
      return {
        targetDigit: c.targetDigit,
        svIndices: new Uint32Array(copy.buffer),
        dualCoefs: base64ToFloat32(c.dualCoefsB64),
        bias: c.bias,
      };
    });

    const probsQ = base64ToBytes(payload.evaluation.testProbsUint8B64);
    const testProbabilities = new Float32Array(probsQ.length);
    for (let i = 0; i < probsQ.length; i++) testProbabilities[i] = probsQ[i] / 255.0;

    return {
      config: payload.config,
      featureDim: payload.featureDim,
      pca,
      supportVectors,
      numUniqueSVs: payload.numUniqueSVs,
      classifiers,
      evaluation: {
        testAccuracy: payload.evaluation.testAccuracy,
        confusionMatrix: payload.evaluation.confusionMatrix,
        classMetrics: payload.evaluation.classMetrics,
        testPredictions: base64ToBytes(payload.evaluation.testPredictionsB64),
        testProbabilities,
        avgPredMs: payload.evaluation.avgPredMs,
      },
      trainTimeMs: payload.trainTimeMs,
      trainedAtIso: payload.trainedAtIso,
    };
  } catch {
    return null;
  }
}

export function clearSavedModelFromLocalStorage(): void {
  try {
    localStorage.removeItem(MODEL_STORAGE_KEY);
  } catch {
    // Ignore storage errors
  }
}
