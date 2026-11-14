import { CnnEpochPoint, CnnTrainConfig } from './cnnEngine';
import {
  ClassEvaluationMetric,
  HydratedSvmModel,
  ModelEvaluationSummary,
  SvmTrainConfig,
} from './svmTypes';

const DB_NAME = 'digit_recognizer_db_v1';
const DB_VERSION = 1;

const STORE_VERSIONS = 'modelVersions';
const STORE_FEEDBACK = 'feedbackSamples';
const STORE_META = 'appMeta';

export interface SerializedSvmIdb {
  config: SvmTrainConfig;
  featureDim: number;
  numUniqueSVs: number;
  supportVectors: Float32Array;
  pca: {
    numComponents: number;
    mean: Float32Array;
    components: Float32Array;
  } | null;
  classifiers: {
    targetDigit: number;
    svIndices: Uint32Array;
    dualCoefs: Float32Array;
    bias: number;
  }[];
  evaluation: ModelEvaluationSummary;
  trainTimeMs: number;
  trainedAtIso: string;
}

export interface ModelVersionRecord {
  id: string;
  modelType: 'svm' | 'cnn';
  versionNumber: number;
  versionLabel: string;
  isBaseline: boolean;
  createdAtIso: string;
  accuracy: number;
  trainTimeMs: number;
  svmPayload?: SerializedSvmIdb;
  cnnUri?: string;
  cnnConfig?: CnnTrainConfig;
  cnnEpochHistory?: CnnEpochPoint[];
  cnnEvaluation?: ModelEvaluationSummary;
}

export interface FeedbackSampleRecord {
  id: string;
  label: number;
  pixels28: Float32Array;
  hash: string;
  predictedDigit: number;
  isDuplicateFlagged: boolean;
  createdAtIso: string;
}

export interface UserRatedStats {
  totalRated: number;
  correctCount: number;
}

export interface AccuracyHistoryEntry {
  id: string;
  timestampIso: string;
  modelType: 'svm' | 'cnn';
  action: string;
  beforeAccuracy: number | null;
  afterAccuracy: number;
  accepted: boolean;
  note: string;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_VERSIONS)) {
        db.createObjectStore(STORE_VERSIONS, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_FEEDBACK)) {
        db.createObjectStore(STORE_FEEDBACK, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_META)) {
        db.createObjectStore(STORE_META, { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Failed to open IndexedDB.'));
  });
}

async function idbGet<T>(storeName: string, key: string): Promise<T | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).get(key);
    req.onsuccess = () => resolve((req.result as T | undefined) ?? null);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut<T>(storeName: string, value: T): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).put(value);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function idbGetAll<T>(storeName: string): Promise<T[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).getAll();
    req.onsuccess = () => resolve((req.result as T[]) ?? []);
    req.onerror = () => reject(req.error);
  });
}

async function idbClear(storeName: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function idbDelete(storeName: string, key: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export function serializeSvmForIdb(model: HydratedSvmModel): SerializedSvmIdb {
  return {
    config: model.config,
    featureDim: model.featureDim,
    numUniqueSVs: model.numUniqueSVs,
    supportVectors: new Float32Array(model.supportVectors),
    pca: model.pca
      ? {
          numComponents: model.pca.numComponents,
          mean: new Float32Array(model.pca.mean),
          components: new Float32Array(model.pca.components),
        }
      : null,
    classifiers: model.classifiers.map((c) => ({
      targetDigit: c.targetDigit,
      svIndices: new Uint32Array(c.svIndices),
      dualCoefs: new Float32Array(c.dualCoefs),
      bias: c.bias,
    })),
    evaluation: {
      testAccuracy: model.evaluation.testAccuracy,
      confusionMatrix: model.evaluation.confusionMatrix.map((r) => [...r]),
      classMetrics: model.evaluation.classMetrics.map((m: ClassEvaluationMetric) => ({ ...m })),
      testPredictions: new Uint8Array(model.evaluation.testPredictions),
      testProbabilities: new Float32Array(model.evaluation.testProbabilities),
      avgPredMs: model.evaluation.avgPredMs,
    },
    trainTimeMs: model.trainTimeMs,
    trainedAtIso: model.trainedAtIso,
  };
}

export function hydrateSvmFromIdb(payload: SerializedSvmIdb): HydratedSvmModel {
  return {
    config: payload.config,
    featureDim: payload.featureDim,
    pca: payload.pca
      ? {
          numComponents: payload.pca.numComponents,
          mean: new Float32Array(payload.pca.mean),
          components: new Float32Array(payload.pca.components),
        }
      : null,
    supportVectors: new Float32Array(payload.supportVectors),
    numUniqueSVs: payload.numUniqueSVs,
    classifiers: payload.classifiers.map((c) => ({
      targetDigit: c.targetDigit,
      svIndices: new Uint32Array(c.svIndices),
      dualCoefs: new Float32Array(c.dualCoefs),
      bias: c.bias,
    })),
    evaluation: {
      testAccuracy: payload.evaluation.testAccuracy,
      confusionMatrix: payload.evaluation.confusionMatrix.map((r) => [...r]),
      classMetrics: payload.evaluation.classMetrics.map((m) => ({ ...m })),
      testPredictions: new Uint8Array(payload.evaluation.testPredictions),
      testProbabilities: new Float32Array(payload.evaluation.testProbabilities),
      avgPredMs: payload.evaluation.avgPredMs,
    },
    trainTimeMs: payload.trainTimeMs,
    trainedAtIso: payload.trainedAtIso,
  };
}

export async function getMetaValue<T>(key: string, fallback: T): Promise<T> {
  const row = await idbGet<{ key: string; value: T }>(STORE_META, key);
  return row ? row.value : fallback;
}

export async function setMetaValue<T>(key: string, value: T): Promise<void> {
  await idbPut(STORE_META, { key, value });
}

export async function listModelVersions(): Promise<ModelVersionRecord[]> {
  const all = await idbGetAll<ModelVersionRecord>(STORE_VERSIONS);
  return all.sort((a, b) => b.createdAtIso.localeCompare(a.createdAtIso));
}

export async function saveModelVersion(record: ModelVersionRecord): Promise<void> {
  await idbPut(STORE_VERSIONS, record);
}

export async function deleteModelVersion(id: string): Promise<void> {
  await idbDelete(STORE_VERSIONS, id);
}

export async function listFeedbackSamples(): Promise<FeedbackSampleRecord[]> {
  const all = await idbGetAll<FeedbackSampleRecord>(STORE_FEEDBACK);
  return all.sort((a, b) => b.createdAtIso.localeCompare(a.createdAtIso));
}

export async function addFeedbackSample(sample: FeedbackSampleRecord): Promise<void> {
  await idbPut(STORE_FEEDBACK, sample);
}

export async function clearFeedbackSamplesOnly(): Promise<void> {
  await idbClear(STORE_FEEDBACK);
  await setMetaValue<UserRatedStats>('userRatedStats', { totalRated: 0, correctCount: 0 });
}

export async function clearAccuracyStatsOnly(): Promise<void> {
  await setMetaValue<AccuracyHistoryEntry[]>('accuracyLogs', []);
  // Keep active & baseline model snapshots intact while pruning intermediate non-baseline versions
  const all = await listModelVersions();
  const activeSvmId = await getMetaValue<string | null>('activeSvmVersionId', null);
  const activeCnnId = await getMetaValue<string | null>('activeCnnVersionId', null);
  for (const v of all) {
    if (!v.isBaseline && v.id !== activeSvmId && v.id !== activeCnnId) {
      if (v.cnnUri && window.tf?.io?.removeModel) {
        try {
          await window.tf.io.removeModel(v.cnnUri);
        } catch {
          // Ignore TF.js IO cleanup errors
        }
      }
      await deleteModelVersion(v.id);
    }
  }
}

export async function factoryResetAllStores(): Promise<void> {
  const versions = await listModelVersions();
  if (window.tf?.io?.removeModel) {
    for (const v of versions) {
      if (v.cnnUri) {
        try {
          await window.tf.io.removeModel(v.cnnUri);
        } catch {
          // Ignore
        }
      }
    }
    try {
      await window.tf.io.removeModel('indexeddb://digit-recognizer-cnn-v1');
    } catch {
      // Ignore
    }
  }
  await Promise.all([idbClear(STORE_VERSIONS), idbClear(STORE_FEEDBACK), idbClear(STORE_META)]);
  try {
    localStorage.clear();
  } catch {
    // Ignore localStorage restrictions
  }
}

/**
 * Computes a compact 7x7 spatial fingerprint hash for fast duplicate detection.
 */
export function computeDigitFingerprint(pixels28: Float32Array): string {
  const vals: number[] = [];
  for (let by = 0; by < 7; by++) {
    for (let bx = 0; bx < 7; bx++) {
      let sum = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          sum += pixels28[(by * 4 + dy) * 28 + (bx * 4 + dx)];
        }
      }
      const avg = sum / 16;
      vals.push(avg > 0.45 ? 2 : avg > 0.14 ? 1 : 0);
    }
  }
  return vals.join('');
}

export function isNearDuplicateDigit(
  pixels28: Float32Array,
  label: number,
  existing: FeedbackSampleRecord[]
): boolean {
  for (const item of existing) {
    if (item.label !== label) continue;
    let sqDist = 0;
    for (let i = 0; i < 784; i++) {
      const d = pixels28[i] - item.pixels28[i];
      sqDist += d * d;
    }
    if (sqDist < 4.0) return true;
  }
  return false;
}

export function validateFeedbackDigits(
  typedText: string,
  expectedCount: number
): { valid: boolean; digits: number[]; error?: string } {
  const clean = typedText.trim();
  if (!clean) {
    return { valid: false, digits: [], error: 'Enter the true number before saving feedback.' };
  }
  if (!/^[0-9]+$/.test(clean)) {
    return { valid: false, digits: [], error: 'Only digits 0–9 are allowed.' };
  }
  if (clean.length !== expectedCount) {
    return {
      valid: false,
      digits: [],
      error: `Expected ${expectedCount} digit${expectedCount === 1 ? '' : 's'} to match the ${expectedCount} segmented digit${expectedCount === 1 ? '' : 's'}, got ${clean.length}.`,
    };
  }
  const numVal = Number.parseInt(clean, 10);
  if (numVal < 0 || numVal > 100) {
    return { valid: false, digits: [], error: 'Number must be within 0–100.' };
  }
  return {
    valid: true,
    digits: clean.split('').map((ch) => Number.parseInt(ch, 10)),
  };
}

/**
 * Lightly augments a 28x28 digit (1px horizontal/vertical shift) so feedback
 * duplicates up to 3x generalize without overfitting exact pixels.
 */
function shiftPixels28(pixels28: Float32Array, dx: number, dy: number): Float32Array {
  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sx = x - dx;
      const sy = y - dy;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) {
        out[y * 28 + x] = pixels28[sy * 28 + sx];
      }
    }
  }
  return out;
}

/**
 * Builds combined training buffers: original MNIST subset + capped feedback (<= 10% of MNIST subset,
 * each unique sample duplicated at most 3x with light 1px shifts), deterministically interleaved.
 */
export function buildCombinedTrainingSet(
  mnistImages: Float32Array,
  mnistLabels: Uint8Array,
  mnistCount: number,
  feedback: FeedbackSampleRecord[]
): { images: Float32Array; labels: Uint8Array; addedCount: number } {
  const maxFeedbackTotal = Math.max(1, Math.floor(mnistCount * 0.1));
  const augmented: { pixels: Float32Array; label: number }[] = [];

  for (const fb of feedback) {
    if (augmented.length >= maxFeedbackTotal) break;
    augmented.push({ pixels: fb.pixels28, label: fb.label });

    if (augmented.length < maxFeedbackTotal) {
      augmented.push({ pixels: shiftPixels28(fb.pixels28, 1, 0), label: fb.label });
    }
    if (augmented.length < maxFeedbackTotal) {
      augmented.push({ pixels: shiftPixels28(fb.pixels28, -1, 1), label: fb.label });
    }
  }

  const totalCount = mnistCount + augmented.length;
  const images = new Float32Array(totalCount * 784);
  const labels = new Uint8Array(totalCount);

  // Copy original MNIST subset
  images.set(mnistImages.subarray(0, mnistCount * 784), 0);
  labels.set(mnistLabels.subarray(0, mnistCount), 0);

  // Append feedback samples
  for (let i = 0; i < augmented.length; i++) {
    images.set(augmented[i].pixels, (mnistCount + i) * 784);
    labels[mnistCount + i] = augmented[i].label;
  }

  // Deterministic Fisher-Yates index shuffle so CNN mini-batches mix MNIST + feedback evenly
  const order = new Int32Array(totalCount);
  for (let i = 0; i < totalCount; i++) order[i] = i;
  let seed = 424242;
  for (let i = totalCount - 1; i > 0; i--) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const j = seed % (i + 1);
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }

  const shuffledImages = new Float32Array(totalCount * 784);
  const shuffledLabels = new Uint8Array(totalCount);
  for (let i = 0; i < totalCount; i++) {
    const src = order[i];
    shuffledImages.set(images.subarray(src * 784, (src + 1) * 784), i * 784);
    shuffledLabels[i] = labels[src];
  }

  return { images: shuffledImages, labels: shuffledLabels, addedCount: augmented.length };
}
