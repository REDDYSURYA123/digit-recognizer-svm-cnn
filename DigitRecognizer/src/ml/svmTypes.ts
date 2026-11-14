export type SvmKernelType = 'linear' | 'rbf' | 'poly';

export type SampleSizeOption = 1000 | 3000 | 5000;

export type PredictionModelMode = 'svm' | 'cnn' | 'ensemble';

export interface SvmTrainConfig {
  kernel: SvmKernelType;
  C: number;
  gamma: number;
  sampleSize: SampleSizeOption;
  usePca: boolean;
}

export interface PcaBasis {
  numComponents: number;
  mean: Float32Array;
  /** Row-major [numComponents x 784] orthonormal principal axes */
  components: Float32Array;
}

export interface BinaryClassifier {
  targetDigit: number;
  /** Indices into uniqueSupportVectors */
  svIndices: Uint32Array;
  /** Signed dual coefficients: alpha_i * y_i */
  dualCoefs: Float32Array;
  bias: number;
}

export interface ClassEvaluationMetric {
  digit: number;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ModelEvaluationSummary {
  testAccuracy: number;
  confusionMatrix: number[][];
  classMetrics: ClassEvaluationMetric[];
  /** Predicted label for each of the 1,000 test samples */
  testPredictions: Uint8Array;
  /** Flattened [1000 x 10] class probabilities for Ensemble evaluation */
  testProbabilities: Float32Array;
  /** Average inference latency per digit in milliseconds */
  avgPredMs: number;
}

export interface HydratedSvmModel {
  config: SvmTrainConfig;
  featureDim: number;
  pca: PcaBasis | null;
  /** Flattened [numUniqueSVs x featureDim] support vectors */
  supportVectors: Float32Array;
  numUniqueSVs: number;
  classifiers: BinaryClassifier[];
  evaluation: ModelEvaluationSummary;
  trainTimeMs: number;
  trainedAtIso: string;
}

export interface PredictionOutput {
  predictedDigit: number;
  decisionScores: number[];
  probabilities: number[];
  confidence: number;
  inferenceMs: number;
}

export interface WorkerTrainRequest {
  type: 'train';
  jobId: number;
  trainImages: Float32Array;
  trainLabels: Uint8Array;
  testImages: Float32Array;
  testLabels: Uint8Array;
  config: SvmTrainConfig;
}

export interface WorkerProgressMessage {
  type: 'progress';
  jobId: number;
  percent: number;
  log?: string;
}

export interface WorkerCompleteMessage {
  type: 'complete';
  jobId: number;
  model: HydratedSvmModel;
}

export interface WorkerErrorMessage {
  type: 'error';
  jobId: number;
  message: string;
}

export type WorkerResponseMessage =
  | WorkerProgressMessage
  | WorkerCompleteMessage
  | WorkerErrorMessage;
