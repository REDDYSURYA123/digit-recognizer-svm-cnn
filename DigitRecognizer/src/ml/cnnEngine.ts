import { base64ToBytes, bytesToBase64 } from './svmInference';
import { ClassEvaluationMetric, ModelEvaluationSummary, PredictionOutput } from './svmTypes';

export type CnnBatchSizeOption = 32 | 64 | 128;
export type CnnSampleSizeOption = 3000 | 5000 | 10000;

export interface CnnTrainConfig {
  epochs: number;
  batchSize: CnnBatchSizeOption;
  learningRate: number;
  sampleSize: CnnSampleSizeOption;
}

export interface CnnEpochPoint {
  epoch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
}

export interface HydratedCnnModel {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tfModel: any;
  config: CnnTrainConfig;
  epochHistory: CnnEpochPoint[];
  evaluation: ModelEvaluationSummary;
  trainTimeMs: number;
  trainedAtIso: string;
}

export const CNN_INDEXEDDB_URI = 'indexeddb://digit-recognizer-cnn-v1';
export const CNN_META_STORAGE_KEY = 'digit_recognizer_cnn_meta_v1';

const TFJS_CDN_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tf?: any;
  }
}

export async function ensureTfLoaded(): Promise<typeof window.tf> {
  if (window.tf) return window.tf;

  await new Promise<void>((resolve, reject) => {
    const existing = document.querySelector(`script[src="${TFJS_CDN_URL}"]`) as HTMLScriptElement | null;
    if (existing) {
      if (window.tf) {
        resolve();
        return;
      }
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error('Failed to load TensorFlow.js CDN.')), {
        once: true,
      });
      return;
    }

    const script = document.createElement('script');
    script.src = TFJS_CDN_URL;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load TensorFlow.js from jsDelivr CDN.'));
    document.head.appendChild(script);
  });

  if (!window.tf) {
    throw new Error('TensorFlow.js global object unavailable after script load.');
  }
  return window.tf;
}

export function buildCnnEvaluationFromProbs(
  probsFlat: Float32Array,
  testLabels: Uint8Array,
  avgPredMs: number
): ModelEvaluationSummary {
  const testCount = testLabels.length;
  const confusionMatrix = Array.from({ length: 10 }, () => Array<number>(10).fill(0));
  const testPredictions = new Uint8Array(testCount);
  let correct = 0;

  for (let i = 0; i < testCount; i++) {
    const pOff = i * 10;
    let bestDigit = 0;
    let bestProb = probsFlat[pOff];
    for (let d = 1; d < 10; d++) {
      if (probsFlat[pOff + d] > bestProb) {
        bestProb = probsFlat[pOff + d];
        bestDigit = d;
      }
    }
    const trueDigit = testLabels[i];
    testPredictions[i] = bestDigit;
    confusionMatrix[trueDigit][bestDigit]++;
    if (bestDigit === trueDigit) correct++;
  }

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

  return {
    testAccuracy: correct / testCount,
    confusionMatrix,
    classMetrics,
    testPredictions,
    testProbabilities: probsFlat,
    avgPredMs,
  };
}

export async function trainCnnModel(
  trainImages: Float32Array,
  trainLabels: Uint8Array,
  testImages: Float32Array,
  testLabels: Uint8Array,
  config: CnnTrainConfig,
  callbacks: {
    onProgress: (percent: number, log?: string) => void;
    onEpochPoint: (point: CnnEpochPoint) => void;
    isCancelled?: () => boolean;
  }
): Promise<HydratedCnnModel> {
  const tf = await ensureTfLoaded();
  const startTime = performance.now();
  const n = config.sampleSize;
  const testCount = testLabels.length;

  callbacks.onProgress(
    3,
    `Initializing TensorFlow.js (${tf.getBackend()}) CNN (N=${n}, epochs=${config.epochs}, batch=${config.batchSize}, lr=${config.learningRate})`
  );

  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Prepare training tensors
  const trainSlice = trainImages.subarray(0, n * 784);
  const xsTrain = tf.tensor4d(trainSlice, [n, 28, 28, 1]);
  const labelIndices = tf.tensor1d(Int32Array.from(trainLabels.subarray(0, n)), 'int32');
  const ysTrain = tf.oneHot(labelIndices, 10);
  labelIndices.dispose();

  const epochHistory: CnnEpochPoint[] = [];
  const totalBatchesPerEpoch = Math.ceil((n * 0.9) / config.batchSize);

  try {
    await model.fit(xsTrain, ysTrain, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: {
        onBatchEnd: async (batch: number) => {
          if (callbacks.isCancelled?.()) {
            model.stopTraining = true;
            return;
          }
          if (batch % 12 === 0) {
            const currentEpoch = epochHistory.length;
            const epochFrac = Math.min(1, (batch + 1) / Math.max(1, totalBatchesPerEpoch));
            const pct = 5 + Math.round(((currentEpoch + epochFrac) / config.epochs) * 84);
            callbacks.onProgress(pct);
            await tf.nextFrame();
          }
        },
        onEpochEnd: async (epoch: number, logs: Record<string, number>) => {
          const loss = logs.loss ?? 0;
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const valLoss = logs.val_loss ?? loss;
          const valAcc = logs.val_acc ?? logs.val_accuracy ?? acc;

          const point: CnnEpochPoint = {
            epoch: epoch + 1,
            loss,
            acc,
            valLoss,
            valAcc,
          };
          epochHistory.push(point);
          callbacks.onEpochPoint(point);

          const pct = 5 + Math.round(((epoch + 1) / config.epochs) * 84);
          callbacks.onProgress(
            pct,
            `Epoch ${epoch + 1}/${config.epochs} — loss: ${loss.toFixed(4)}, acc: ${(acc * 100).toFixed(2)}%, val_acc: ${(valAcc * 100).toFixed(2)}%`
          );
          await tf.nextFrame();
        },
      },
    });
  } finally {
    xsTrain.dispose();
    ysTrain.dispose();
  }

  if (callbacks.isCancelled?.()) {
    model.dispose();
    throw new Error('CNN training cancelled.');
  }

  callbacks.onProgress(92, `Evaluating CNN on ${testCount} held-out test samples...`);
  await tf.nextFrame();

  const evalStart = performance.now();
  const xsTest = tf.tensor4d(testImages, [testCount, 28, 28, 1]);
  const predTensor = model.predict(xsTest);
  const probsRaw = (await predTensor.data()) as Float32Array;
  const probsCopy = new Float32Array(probsRaw);
  xsTest.dispose();
  predTensor.dispose();

  const avgPredMs = (performance.now() - evalStart) / testCount;
  const evaluation = buildCnnEvaluationFromProbs(probsCopy, testLabels, avgPredMs);
  const trainTimeMs = performance.now() - startTime;

  callbacks.onProgress(
    100,
    `CNN complete in ${(trainTimeMs / 1000).toFixed(2)}s — Test accuracy: ${(evaluation.testAccuracy * 100).toFixed(2)}% (${avgPredMs.toFixed(2)} ms/digit)`
  );

  return {
    tfModel: model,
    config,
    epochHistory,
    evaluation,
    trainTimeMs,
    trainedAtIso: new Date().toISOString(),
  };
}

export function predictDigitWithCnn(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tfModel: any,
  pixels28: Float32Array
): PredictionOutput {
  const tf = window.tf;
  if (!tf || !tfModel) {
    throw new Error('CNN model is not ready.');
  }

  const t0 = performance.now();
  const probsData = tf.tidy(() => {
    const input = tf.tensor4d(pixels28, [1, 28, 28, 1]);
    const output = tfModel.predict(input);
    return output.dataSync() as Float32Array;
  });

  const probabilities = Array.from(probsData) as number[];
  let predictedDigit = 0;
  for (let d = 1; d < 10; d++) {
    if (probabilities[d] > probabilities[predictedDigit]) predictedDigit = d;
  }

  return {
    predictedDigit,
    decisionScores: probabilities,
    probabilities,
    confidence: probabilities[predictedDigit],
    inferenceMs: performance.now() - t0,
  };
}

interface StoredCnnMetadata {
  version: number;
  config: CnnTrainConfig;
  epochHistory: CnnEpochPoint[];
  evaluation: {
    testAccuracy: number;
    confusionMatrix: number[][];
    classMetrics: ClassEvaluationMetric[];
    testPredictionsB64: string;
    testProbsUint8B64: string;
    avgPredMs: number;
  };
  trainTimeMs: number;
  trainedAtIso: string;
}

export function makeCnnVersionUri(versionId: string): string {
  return `indexeddb://digit-recognizer-cnn-${versionId}`;
}

export async function saveCnnModelWeightsAtUri(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tfModel: any,
  uri: string
): Promise<void> {
  await tfModel.save(uri);
}

export async function loadCnnModelWeightsAtUri(
  uri: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Promise<any> {
  const tf = await ensureTfLoaded();
  return tf.loadLayersModel(uri);
}

export async function fineTuneCnnModel(
  sourceUri: string | null,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fallbackCurrentTfModel: any,
  prevConfig: CnnTrainConfig,
  prevEpochHistory: CnnEpochPoint[],
  combinedImages: Float32Array,
  combinedLabels: Uint8Array,
  testImages: Float32Array,
  testLabels: Uint8Array,
  fineTuneEpochs: 1 | 2,
  callbacks: {
    onProgress: (percent: number, log?: string) => void;
    onEpochPoint?: (point: CnnEpochPoint) => void;
  }
): Promise<HydratedCnnModel> {
  const tf = await ensureTfLoaded();
  const startTime = performance.now();
  const n = combinedLabels.length;
  const testCount = testLabels.length;
  const lr = 0.0001;
  const batchSize = prevConfig.batchSize;

  callbacks.onProgress(
    4,
    `Cloning current CNN weights for guarded fine-tuning (N=${n}, epochs=${fineTuneEpochs}, lr=${lr})...`
  );

  // Load an isolated copy so the active model is never mutated if accuracy guard rejects
  let candidateModel;
  const tempCloneUri = `indexeddb://digit-recognizer-cnn-temp-clone`;
  try {
    if (sourceUri) {
      candidateModel = await tf.loadLayersModel(sourceUri);
    } else {
      await fallbackCurrentTfModel.save(tempCloneUri);
      candidateModel = await tf.loadLayersModel(tempCloneUri);
    }
  } catch {
    await fallbackCurrentTfModel.save(tempCloneUri);
    candidateModel = await tf.loadLayersModel(tempCloneUri);
  } finally {
    try {
      await tf.io.removeModel(tempCloneUri);
    } catch {
      // Ignore temp cleanup
    }
  }

  candidateModel.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const xsTrain = tf.tensor4d(combinedImages, [n, 28, 28, 1]);
  const labelIndices = tf.tensor1d(Int32Array.from(combinedLabels), 'int32');
  const ysTrain = tf.oneHot(labelIndices, 10);
  labelIndices.dispose();

  const epochHistory: CnnEpochPoint[] = [...prevEpochHistory];
  const baseEpochOffset = prevEpochHistory.length;
  const totalBatchesPerEpoch = Math.ceil((n * 0.9) / batchSize);

  try {
    await candidateModel.fit(xsTrain, ysTrain, {
      epochs: fineTuneEpochs,
      batchSize,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: {
        onBatchEnd: async (batch: number) => {
          if (batch % 12 === 0) {
            const doneEpochs = epochHistory.length - baseEpochOffset;
            const epochFrac = Math.min(1, (batch + 1) / Math.max(1, totalBatchesPerEpoch));
            const pct = 8 + Math.round(((doneEpochs + epochFrac) / fineTuneEpochs) * 78);
            callbacks.onProgress(pct);
            await tf.nextFrame();
          }
        },
        onEpochEnd: async (epoch: number, logs: Record<string, number>) => {
          const loss = logs.loss ?? 0;
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const valLoss = logs.val_loss ?? loss;
          const valAcc = logs.val_acc ?? logs.val_accuracy ?? acc;
          const point: CnnEpochPoint = {
            epoch: baseEpochOffset + epoch + 1,
            loss,
            acc,
            valLoss,
            valAcc,
          };
          epochHistory.push(point);
          callbacks.onEpochPoint?.(point);

          const pct = 8 + Math.round(((epoch + 1) / fineTuneEpochs) * 78);
          callbacks.onProgress(
            pct,
            `Fine-tune Epoch ${epoch + 1}/${fineTuneEpochs} — loss: ${loss.toFixed(4)}, acc: ${(acc * 100).toFixed(2)}%`
          );
          await tf.nextFrame();
        },
      },
    });
  } finally {
    xsTrain.dispose();
    ysTrain.dispose();
  }

  callbacks.onProgress(90, `Evaluating fine-tuned CNN on fixed ${testCount}-sample MNIST test set...`);
  await tf.nextFrame();

  const evalStart = performance.now();
  const xsTest = tf.tensor4d(testImages, [testCount, 28, 28, 1]);
  const predTensor = candidateModel.predict(xsTest);
  const probsRaw = (await predTensor.data()) as Float32Array;
  const probsCopy = new Float32Array(probsRaw);
  xsTest.dispose();
  predTensor.dispose();

  const avgPredMs = (performance.now() - evalStart) / testCount;
  const evaluation = buildCnnEvaluationFromProbs(probsCopy, testLabels, avgPredMs);
  const trainTimeMs = performance.now() - startTime;

  callbacks.onProgress(
    100,
    `Fine-tuning finished in ${(trainTimeMs / 1000).toFixed(2)}s — Fixed test accuracy: ${(evaluation.testAccuracy * 100).toFixed(2)}%`
  );

  return {
    tfModel: candidateModel,
    config: { ...prevConfig, learningRate: lr },
    epochHistory,
    evaluation,
    trainTimeMs,
    trainedAtIso: new Date().toISOString(),
  };
}

export async function saveCnnToIndexedDb(model: HydratedCnnModel): Promise<{ saved: boolean; error?: string }> {
  try {
    await model.tfModel.save(CNN_INDEXEDDB_URI);

    const probsQ = new Uint8Array(model.evaluation.testProbabilities.length);
    for (let i = 0; i < probsQ.length; i++) {
      probsQ[i] = Math.max(0, Math.min(255, Math.round(model.evaluation.testProbabilities[i] * 255)));
    }

    const meta: StoredCnnMetadata = {
      version: 1,
      config: model.config,
      epochHistory: model.epochHistory,
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

    localStorage.setItem(CNN_META_STORAGE_KEY, JSON.stringify(meta));
    return { saved: true };
  } catch (err) {
    return {
      saved: false,
      error: err instanceof Error ? err.message : 'IndexedDB save failed',
    };
  }
}

export async function loadCnnFromIndexedDb(): Promise<HydratedCnnModel | null> {
  try {
    const rawMeta = localStorage.getItem(CNN_META_STORAGE_KEY);
    if (!rawMeta) return null;
    const meta = JSON.parse(rawMeta) as StoredCnnMetadata;
    if (meta.version !== 1) return null;

    const tf = await ensureTfLoaded();
    const tfModel = await tf.loadLayersModel(CNN_INDEXEDDB_URI);

    const probsQ = base64ToBytes(meta.evaluation.testProbsUint8B64);
    const testProbabilities = new Float32Array(probsQ.length);
    for (let i = 0; i < probsQ.length; i++) testProbabilities[i] = probsQ[i] / 255.0;

    return {
      tfModel,
      config: meta.config,
      epochHistory: meta.epochHistory,
      evaluation: {
        testAccuracy: meta.evaluation.testAccuracy,
        confusionMatrix: meta.evaluation.confusionMatrix,
        classMetrics: meta.evaluation.classMetrics,
        testPredictions: base64ToBytes(meta.evaluation.testPredictionsB64),
        testProbabilities,
        avgPredMs: meta.evaluation.avgPredMs,
      },
      trainTimeMs: meta.trainTimeMs,
      trainedAtIso: meta.trainedAtIso,
    };
  } catch {
    return null;
  }
}

export async function clearSavedCnnCache(): Promise<void> {
  try {
    localStorage.removeItem(CNN_META_STORAGE_KEY);
    const tf = window.tf;
    if (tf?.io?.removeModel) {
      await tf.io.removeModel(CNN_INDEXEDDB_URI);
    }
  } catch {
    // Ignore cleanup errors
  }
}
