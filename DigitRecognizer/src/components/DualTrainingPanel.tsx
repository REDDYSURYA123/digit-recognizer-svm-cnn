import { Play, RotateCcw, Smartphone, Trash2 } from 'lucide-react';
import {
  CnnBatchSizeOption,
  CnnEpochPoint,
  CnnSampleSizeOption,
  CnnTrainConfig,
  HydratedCnnModel,
} from '../ml/cnnEngine';
import {
  HydratedSvmModel,
  SampleSizeOption,
  SvmKernelType,
  SvmTrainConfig,
} from '../ml/svmTypes';

interface DualTrainingPanelProps {
  isDatasetReady: boolean;
  isMobileViewport: boolean;
  // SVM state
  svmConfig: SvmTrainConfig;
  onSvmConfigChange: (next: SvmTrainConfig) => void;
  onTrainSvm: () => void;
  onClearSvmCache: () => void;
  isSvmTraining: boolean;
  svmProgress: number;
  svmLogs: string[];
  svmModel: HydratedSvmModel | null;
  svmCacheInfo: string | null;
  // CNN state
  cnnConfig: CnnTrainConfig;
  onCnnConfigChange: (next: CnnTrainConfig) => void;
  onTrainCnn: () => void;
  onClearCnnCache: () => void;
  isCnnTraining: boolean;
  cnnProgress: number;
  cnnLogs: string[];
  cnnHistory: CnnEpochPoint[];
  cnnModel: HydratedCnnModel | null;
  cnnCacheInfo: string | null;
}

function sliderToC(pos: number): number {
  const val = 0.1 * Math.pow(1000, pos / 100);
  if (val < 1) return Math.round(val * 10) / 10;
  if (val < 10) return Math.round(val * 5) / 5;
  return Math.round(val);
}

function cToSlider(c: number): number {
  const clamped = Math.max(0.1, Math.min(100, c));
  return Math.round((Math.log(clamped / 0.1) / Math.log(1000)) * 100);
}

function CnnLiveCharts({ history }: { history: CnnEpochPoint[] }) {
  if (history.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 dark:border-[#1E243D] py-5 text-center font-mono text-[11px] text-slate-400 dark:text-slate-500">
        Epoch loss & accuracy curves appear here during CNN training.
      </div>
    );
  }

  const w = 240;
  const h = 88;
  const padX = 24;
  const padY = 10;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;
  const maxEpoch = Math.max(2, history.length);

  const maxLoss = Math.max(
    0.5,
    ...history.map((p) => Math.max(p.loss, p.valLoss))
  );

  const buildPath = (vals: number[], yMax: number, yMin = 0) =>
    vals
      .map((v, i) => {
        const x = padX + (history.length === 1 ? innerW / 2 : (i / (maxEpoch - 1)) * innerW);
        const norm = Math.max(0, Math.min(1, (v - yMin) / Math.max(1e-6, yMax - yMin)));
        const y = padY + (1 - norm) * innerH;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  const lossPath = buildPath(history.map((p) => p.loss), maxLoss, 0);
  const valLossPath = buildPath(history.map((p) => p.valLoss), maxLoss, 0);
  const accPath = buildPath(history.map((p) => p.acc), 1.0, 0.5);
  const valAccPath = buildPath(history.map((p) => p.valAcc), 1.0, 0.5);
  const latest = history[history.length - 1];

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {/* Loss Chart */}
      <div className="rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-3">
        <div className="flex items-center justify-between font-mono text-[10px] mb-1">
          <span className="text-slate-500 dark:text-slate-400">LOSS (TRAIN / VAL)</span>
          <span className="text-violet-600 dark:text-violet-400 tabular-nums">
            {latest.loss.toFixed(3)} / {latest.valLoss.toFixed(3)}
          </span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto overflow-visible">
          <line x1={padX} y1={padY} x2={padX} y2={h - padY} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth={1} />
          <line x1={padX} y1={h - padY} x2={w - padX} y2={h - padY} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth={1} />
          <path d={lossPath} fill="none" stroke="#8b5cf6" strokeWidth={2} />
          <path d={valLossPath} fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 2" />
        </svg>
      </div>

      {/* Accuracy Chart */}
      <div className="rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-3">
        <div className="flex items-center justify-between font-mono text-[10px] mb-1">
          <span className="text-slate-500 dark:text-slate-400">ACCURACY (TRAIN / VAL)</span>
          <span className="text-teal-600 dark:text-teal-400 tabular-nums">
            {(latest.acc * 100).toFixed(1)}% / {(latest.valAcc * 100).toFixed(1)}%
          </span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto overflow-visible">
          <line x1={padX} y1={padY} x2={padX} y2={h - padY} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth={1} />
          <line x1={padX} y1={h - padY} x2={w - padX} y2={h - padY} stroke="currentColor" className="text-slate-300 dark:text-slate-700" strokeWidth={1} />
          <path d={accPath} fill="none" stroke="#14b8a6" strokeWidth={2} />
          <path d={valAccPath} fill="none" stroke="#6366f1" strokeWidth={1.5} strokeDasharray="3 2" />
        </svg>
      </div>
    </div>
  );
}

export function DualTrainingPanel({
  isDatasetReady,
  isMobileViewport,
  svmConfig,
  onSvmConfigChange,
  onTrainSvm,
  onClearSvmCache,
  isSvmTraining,
  svmProgress,
  svmLogs,
  svmModel,
  svmCacheInfo,
  cnnConfig,
  onCnnConfigChange,
  onTrainCnn,
  onClearCnnCache,
  isCnnTraining,
  cnnProgress,
  cnnLogs,
  cnnHistory,
  cnnModel,
  cnnCacheInfo,
}: DualTrainingPanelProps) {
  const sliderPos = cToSlider(svmConfig.C);

  const handleKernelChange = (kernel: SvmKernelType) => {
    const nextGamma =
      kernel === 'poly' && svmConfig.gamma > 0.05
        ? 0.02
        : kernel === 'rbf' && svmConfig.gamma < 0.01
        ? 0.05
        : svmConfig.gamma;
    onSvmConfigChange({ ...svmConfig, kernel, gamma: nextGamma });
  };

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-indigo-200/80 dark:border-[#1E243D] bg-gradient-to-r from-indigo-600/10 via-violet-600/10 to-teal-500/10 p-5 sm:p-6 shadow-xs flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
            Model Training: SVM (Web Worker) vs. CNN (TensorFlow.js)
          </h2>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
            Train either model or both for Ensemble predictions. Baseline snapshots and version history persist offline in IndexedDB.
          </p>
        </div>

        {(isMobileViewport || svmConfig.sampleSize <= 3000 || cnnConfig.epochs <= 3) && (
          <div className="inline-flex items-center gap-2 rounded-xl border border-amber-300/80 dark:border-amber-800/70 bg-amber-50 dark:bg-amber-950/40 px-3.5 py-2 text-xs text-amber-800 dark:text-amber-200 shrink-0">
            <Smartphone className="h-4 w-4 shrink-0" />
            <span>Large sample counts or high epochs may be slow on phones.</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 items-start">
        {/* MODEL 1: SVM */}
        <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between gap-3 border-b border-slate-200 dark:border-[#1E243D] pb-3">
            <div>
              <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                Model 1 · Support Vector Machine (SMO)
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Plain-JS One-vs-Rest SMO in a non-blocking Web Worker
              </p>
            </div>
            {svmModel && (
              <span className="font-mono text-xs font-bold text-indigo-600 dark:text-indigo-400 tabular-nums">
                {(svmModel.evaluation.testAccuracy * 100).toFixed(2)}% acc
              </span>
            )}
          </div>

          <div>
            <div className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
              Kernel Function
            </div>
            <div className="grid grid-cols-3 gap-1 rounded-xl bg-slate-100 dark:bg-[#0B0F1D] p-1 border border-slate-200 dark:border-[#1E243D]">
              {(['linear', 'rbf', 'poly'] as SvmKernelType[]).map((k) => {
                const active = svmConfig.kernel === k;
                const label = k === 'linear' ? 'Linear' : k === 'rbf' ? 'RBF' : 'Polynomial';
                return (
                  <button
                    key={k}
                    type="button"
                    disabled={isSvmTraining}
                    onClick={() => handleKernelChange(k)}
                    className={`min-h-[40px] rounded-lg px-3 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-50 ${
                      active
                        ? 'bg-indigo-600 text-white shadow-xs'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="svm-c-range" className="font-medium text-slate-700 dark:text-slate-300">
                  Regularization C
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                  {svmConfig.C}
                </span>
              </div>
              <input
                id="svm-c-range"
                type="range"
                min={0}
                max={100}
                step={1}
                disabled={isSvmTraining}
                value={sliderPos}
                onChange={(e) =>
                  onSvmConfigChange({ ...svmConfig, C: sliderToC(Number(e.target.value)) })
                }
                className="w-full h-6 accent-indigo-600 cursor-pointer"
              />
              <div className="flex justify-between font-mono text-[10px] text-slate-400">
                <span>0.1</span>
                <span>100</span>
              </div>
            </div>

            <div className={svmConfig.kernel === 'linear' ? 'opacity-45 pointer-events-none' : ''}>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="svm-gamma-range" className="font-medium text-slate-700 dark:text-slate-300">
                  Gamma (γ)
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                  {svmConfig.gamma.toFixed(3)}
                </span>
              </div>
              <input
                id="svm-gamma-range"
                type="range"
                min={0.002}
                max={0.25}
                step={0.002}
                disabled={isSvmTraining || svmConfig.kernel === 'linear'}
                value={svmConfig.gamma}
                onChange={(e) =>
                  onSvmConfigChange({ ...svmConfig, gamma: Number(e.target.value) })
                }
                className="w-full h-6 accent-indigo-600 cursor-pointer"
              />
              <div className="flex justify-between font-mono text-[10px] text-slate-400">
                <span>0.002</span>
                <span>0.250</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 items-center">
            <div>
              <label
                htmlFor="svm-samples-select"
                className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1"
              >
                Training Samples
              </label>
              <select
                id="svm-samples-select"
                disabled={isSvmTraining}
                value={svmConfig.sampleSize}
                onChange={(e) =>
                  onSvmConfigChange({
                    ...svmConfig,
                    sampleSize: Number(e.target.value) as SampleSizeOption,
                  })
                }
                className="w-full min-h-[44px] rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-3 font-mono text-xs text-slate-900 dark:text-slate-100 focus:border-indigo-500 focus:outline-none"
              >
                <option value={1000}>1,000 samples (Fast / Mobile)</option>
                <option value={3000}>3,000 samples (Balanced)</option>
                <option value={5000}>5,000 samples (Full Desktop)</option>
              </select>
            </div>

            <div className="min-h-[44px] flex items-center justify-between rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] px-3.5 py-2 sm:mt-5">
              <div>
                <div className="text-xs font-medium text-slate-800 dark:text-slate-200">
                  PCA (784 → 50)
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">
                  {svmConfig.usePca ? '50 features' : '784 pixels'}
                </div>
              </div>
              <button
                type="button"
                role="switch"
                aria-label="Toggle 50-component PCA dimensionality reduction"
                aria-checked={svmConfig.usePca}
                disabled={isSvmTraining}
                onClick={() => onSvmConfigChange({ ...svmConfig, usePca: !svmConfig.usePca })}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
                  svmConfig.usePca ? 'bg-indigo-600' : 'bg-slate-300 dark:bg-slate-700'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                    svmConfig.usePca ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              disabled={!isDatasetReady || isSvmTraining}
              onClick={onTrainSvm}
              className="min-h-[44px] inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-4 text-xs font-semibold text-white hover:from-indigo-700 hover:to-violet-700 disabled:opacity-50 transition-all cursor-pointer whitespace-nowrap"
            >
              {isSvmTraining ? (
                <>
                  <RotateCcw className="h-4 w-4 animate-spin" />
                  Training SVM ({svmProgress}%)
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Train SVM
                </>
              )}
            </button>
            {svmModel && (
              <button
                type="button"
                disabled={isSvmTraining}
                onClick={onClearSvmCache}
                aria-label="Clear active SVM model from cache"
                title="Clear cached SVM model"
                className="min-h-[44px] min-w-[44px] inline-flex items-center justify-center rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-3 text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            )}
          </div>

          {/* Progress Bar & Log */}
          <div className="space-y-2 border-t border-slate-200 dark:border-[#1E243D] pt-3">
            <div className="flex justify-between font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
              <span>SVM PROGRESS</span>
              <span>{svmProgress}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-600 to-violet-500 transition-all duration-150"
                style={{ width: `${svmProgress}%` }}
              />
            </div>
            <div className="h-28 overflow-y-auto rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-3 font-mono text-[11px] leading-relaxed text-slate-700 dark:text-slate-300 tabular-nums">
              {svmLogs.length === 0 ? (
                <span className="text-slate-400 dark:text-slate-500">
                  Ready to train SVM in Web Worker...
                </span>
              ) : (
                svmLogs.map((l, i) => (
                  <div key={i} className="truncate">
                    {l}
                  </div>
                ))
              )}
            </div>
            {svmCacheInfo && (
              <div className="font-mono text-[11px] text-slate-500 dark:text-slate-400">
                {svmCacheInfo}
              </div>
            )}
          </div>
        </div>

        {/* MODEL 2: CNN */}
        <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between gap-3 border-b border-slate-200 dark:border-[#1E243D] pb-3">
            <div>
              <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                Model 2 · Convolutional Neural Network
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Conv2D(32) → Pool → Conv2D(64) → Pool → Dropout(0.3) → Dense(128) → Softmax(10)
              </p>
            </div>
            {cnnModel && (
              <span className="font-mono text-xs font-bold text-teal-600 dark:text-teal-400 tabular-nums">
                {(cnnModel.evaluation.testAccuracy * 100).toFixed(2)}% acc
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="cnn-epochs-range" className="font-medium text-slate-700 dark:text-slate-300">
                  Epochs
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                  {cnnConfig.epochs}
                </span>
              </div>
              <input
                id="cnn-epochs-range"
                type="range"
                min={1}
                max={10}
                step={1}
                disabled={isCnnTraining}
                value={cnnConfig.epochs}
                onChange={(e) =>
                  onCnnConfigChange({ ...cnnConfig, epochs: Number(e.target.value) })
                }
                className="w-full h-6 accent-teal-600 cursor-pointer"
              />
              <div className="flex justify-between font-mono text-[10px] text-slate-400">
                <span>1</span>
                <span>5</span>
                <span>10</span>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="cnn-lr-select" className="font-medium text-slate-700 dark:text-slate-300">
                  Adam Learning Rate
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                  {cnnConfig.learningRate}
                </span>
              </div>
              <select
                id="cnn-lr-select"
                disabled={isCnnTraining}
                value={cnnConfig.learningRate}
                onChange={(e) =>
                  onCnnConfigChange({ ...cnnConfig, learningRate: Number(e.target.value) })
                }
                className="w-full min-h-[44px] rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-3 font-mono text-xs text-slate-900 dark:text-slate-100 focus:border-teal-500 focus:outline-none"
              >
                <option value={0.0005}>0.0005</option>
                <option value={0.001}>0.001 (Recommended)</option>
                <option value={0.002}>0.002</option>
                <option value={0.005}>0.005</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                Batch Size
              </div>
              <div className="grid grid-cols-3 gap-1 rounded-xl bg-slate-100 dark:bg-[#0B0F1D] p-1 border border-slate-200 dark:border-[#1E243D]">
                {([32, 64, 128] as CnnBatchSizeOption[]).map((bs) => {
                  const active = cnnConfig.batchSize === bs;
                  return (
                    <button
                      key={bs}
                      type="button"
                      disabled={isCnnTraining}
                      onClick={() => onCnnConfigChange({ ...cnnConfig, batchSize: bs })}
                      className={`min-h-[38px] rounded-lg font-mono text-xs font-semibold transition-colors cursor-pointer disabled:opacity-50 ${
                        active
                          ? 'bg-teal-600 text-white shadow-xs'
                          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                      }`}
                    >
                      {bs}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <label
                htmlFor="cnn-samples-select"
                className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1"
              >
                Training Samples
              </label>
              <select
                id="cnn-samples-select"
                disabled={isCnnTraining}
                value={cnnConfig.sampleSize}
                onChange={(e) =>
                  onCnnConfigChange({
                    ...cnnConfig,
                    sampleSize: Number(e.target.value) as CnnSampleSizeOption,
                  })
                }
                className="w-full min-h-[44px] rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-3 font-mono text-xs text-slate-900 dark:text-slate-100 focus:border-teal-500 focus:outline-none"
              >
                <option value={3000}>3,000 samples (Fast / Mobile)</option>
                <option value={5000}>5,000 samples</option>
                <option value={10000}>10,000 samples (Full)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              disabled={!isDatasetReady || isCnnTraining}
              onClick={onTrainCnn}
              className="min-h-[44px] inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-600 to-emerald-600 px-4 text-xs font-semibold text-white hover:from-teal-700 hover:to-emerald-700 disabled:opacity-50 transition-all cursor-pointer whitespace-nowrap"
            >
              {isCnnTraining ? (
                <>
                  <RotateCcw className="h-4 w-4 animate-spin" />
                  Training CNN ({cnnProgress}%)
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Train CNN
                </>
              )}
            </button>
            {cnnModel && (
              <button
                type="button"
                disabled={isCnnTraining}
                onClick={onClearCnnCache}
                aria-label="Clear active CNN model from cache"
                title="Clear cached CNN model from IndexedDB"
                className="min-h-[44px] min-w-[44px] inline-flex items-center justify-center rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-3 text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            )}
          </div>

          {/* Live Epoch Charts */}
          <CnnLiveCharts history={cnnHistory} />

          {/* Progress Bar & Log */}
          <div className="space-y-2 border-t border-slate-200 dark:border-[#1E243D] pt-3">
            <div className="flex justify-between font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
              <span>CNN PROGRESS</span>
              <span>{cnnProgress}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-teal-500 to-emerald-500 transition-all duration-150"
                style={{ width: `${cnnProgress}%` }}
              />
            </div>
            <div className="h-24 overflow-y-auto rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-3 font-mono text-[11px] leading-relaxed text-slate-700 dark:text-slate-300 tabular-nums">
              {cnnLogs.length === 0 ? (
                <span className="text-slate-400 dark:text-slate-500">
                  Click Train CNN to build and train the TensorFlow.js network...
                </span>
              ) : (
                cnnLogs.map((l, i) => (
                  <div key={i} className="truncate">
                    {l}
                  </div>
                ))
              )}
            </div>
            {cnnCacheInfo && (
              <div className="font-mono text-[11px] text-slate-500 dark:text-slate-400">
                {cnnCacheInfo}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
