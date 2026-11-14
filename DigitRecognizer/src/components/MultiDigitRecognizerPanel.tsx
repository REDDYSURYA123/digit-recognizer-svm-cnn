import { useCallback, useRef, useState } from 'react';
import {
  Check,
  Cpu,
  Eraser,
  ImageUp,
  MessageSquarePlus,
  Play,
  ThumbsUp,
} from 'lucide-react';
import {
  SegmentedDigitRegion,
  segmentGridToDigits,
  segmentUploadedFileToDigits,
} from '../ml/mnistCdnLoader';
import { HydratedCnnModel, predictDigitWithCnn } from '../ml/cnnEngine';
import { predictDigitWithSvm } from '../ml/svmInference';
import {
  HydratedSvmModel,
  PredictionModelMode,
  PredictionOutput,
} from '../ml/svmTypes';
import { MnistPixelCanvas } from './MnistPixelCanvas';

export interface FeedbackSubmissionRegion {
  pixels28: Float32Array;
  predictedDigit: number;
}

export interface QuickFeedbackResponse {
  ok: boolean;
  message: string;
  duplicateFlagged?: boolean;
}

interface MultiDigitRecognizerPanelProps {
  svmModel: HydratedSvmModel | null;
  cnnModel: HydratedCnnModel | null;
  mode: PredictionModelMode;
  onModeChange: (mode: PredictionModelMode) => void;
  isDatasetReady: boolean;
  isHydratingModels: boolean;
  isSvmTraining: boolean;
  isCnnTraining: boolean;
  onMarkPredictionRight: (regions: FeedbackSubmissionRegion[]) => void;
  onSubmitCorrectionFeedback: (
    regions: FeedbackSubmissionRegion[],
    typedNumber: string
  ) => Promise<QuickFeedbackResponse>;
  onOpenFeedbackTab: () => void;
  onGoToTrain: () => void;
}

interface SegmentedPredictionItem {
  region: SegmentedDigitRegion;
  activeOutput: PredictionOutput;
  svmOutput: PredictionOutput | null;
  cnnOutput: PredictionOutput | null;
}

const CANVAS_W = 560;
const CANVAS_H = 280;

export function MultiDigitRecognizerPanel({
  svmModel,
  cnnModel,
  mode,
  onModeChange,
  isDatasetReady,
  isHydratingModels,
  isSvmTraining,
  isCnnTraining,
  onMarkPredictionRight,
  onSubmitCorrectionFeedback,
  onOpenFeedbackTab,
  onGoToTrain,
}: MultiDigitRecognizerPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isDrawingRef = useRef(false);
  const lastPtRef = useRef<{ x: number; y: number } | null>(null);
  const hasDrawnRef = useRef(false);

  const [results, setResults] = useState<SegmentedPredictionItem[]>([]);
  const [selectedDigitIdx, setSelectedDigitIdx] = useState<number>(0);
  const [totalInferenceMs, setTotalInferenceMs] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [correctionInput, setCorrectionInput] = useState<string>('');
  const [feedbackStatus, setFeedbackStatus] = useState<{
    type: 'success' | 'error' | 'warning';
    text: string;
  } | null>(null);
  const [ratedThisPrediction, setRatedThisPrediction] = useState<boolean>(false);

  const attachCanvasRef = useCallback((el: HTMLCanvasElement | null) => {
    canvasRef.current = el;
    if (el && !hasDrawnRef.current) {
      const ctx = el.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      }
    }
  }, []);

  const clearCanvasSurface = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    hasDrawnRef.current = false;
  }, []);

  const getPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * CANVAS_W,
      y: ((e.clientY - rect.top) / rect.height) * CANVAS_H,
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    isDrawingRef.current = true;
    hasDrawnRef.current = true;
    setNotice(null);
    setFeedbackStatus(null);

    const pt = getPos(e);
    lastPtRef.current = pt;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 9, 0, Math.PI * 2);
    ctx.fill();
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current || !lastPtRef.current) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const pt = getPos(e);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(lastPtRef.current.x, lastPtRef.current.y);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();

    lastPtRef.current = pt;
  };

  const handlePointerUp = () => {
    isDrawingRef.current = false;
    lastPtRef.current = null;
  };

  const runPredictionOnRegions = useCallback(
    (regions: SegmentedDigitRegion[], activeMode: PredictionModelMode) => {
      setFeedbackStatus(null);
      setRatedThisPrediction(false);
      setCorrectionInput('');

      if (regions.length === 0) {
        setResults([]);
        setTotalInferenceMs(null);
        setNotice('Canvas is empty. Draw a number from 0 to 100 or upload an image first.');
        return;
      }

      if (activeMode === 'svm' && !svmModel) {
        setNotice(
          isSvmTraining
            ? 'SVM is currently training. Please wait or switch to CNN mode.'
            : 'Train the SVM model first in the Train tab before predicting in SVM mode.'
        );
        return;
      }
      if (activeMode === 'cnn' && !cnnModel) {
        setNotice(
          isCnnTraining
            ? 'CNN is currently training. Please wait or switch to SVM mode.'
            : 'Train the CNN model first in the Train tab before predicting in CNN mode.'
        );
        return;
      }
      if (activeMode === 'ensemble' && (!svmModel || !cnnModel)) {
        setNotice('Ensemble mode requires both SVM and CNN to be trained.');
        return;
      }

      const tStart = performance.now();
      const items: SegmentedPredictionItem[] = regions.map((region) => {
        const svmOut = svmModel ? predictDigitWithSvm(svmModel, region.pixels28) : null;
        const cnnOut = cnnModel ? predictDigitWithCnn(cnnModel.tfModel, region.pixels28) : null;

        let activeOutput: PredictionOutput;
        if (activeMode === 'svm' && svmOut) {
          activeOutput = svmOut;
        } else if (activeMode === 'cnn' && cnnOut) {
          activeOutput = cnnOut;
        } else if (svmOut && cnnOut) {
          const avgProbs = svmOut.probabilities.map(
            (p, idx) => 0.5 * (p + cnnOut.probabilities[idx])
          );
          let bestDigit = 0;
          for (let d = 1; d < 10; d++) {
            if (avgProbs[d] > avgProbs[bestDigit]) bestDigit = d;
          }
          activeOutput = {
            predictedDigit: bestDigit,
            decisionScores: avgProbs,
            probabilities: avgProbs,
            confidence: avgProbs[bestDigit],
            inferenceMs: svmOut.inferenceMs + cnnOut.inferenceMs,
          };
        } else {
          activeOutput = (svmOut ?? cnnOut)!;
        }

        return {
          region,
          activeOutput,
          svmOutput: svmOut,
          cnnOutput: cnnOut,
        };
      });

      setTotalInferenceMs(performance.now() - tStart);
      setResults(items);
      setSelectedDigitIdx(0);
    },
    [svmModel, cnnModel, isSvmTraining, isCnnTraining]
  );

  const extractCanvasGrid = (): Float32Array | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    const { data } = ctx.getImageData(0, 0, CANVAS_W, CANVAS_H);
    const grid = new Float32Array(CANVAS_W * CANVAS_H);
    for (let i = 0; i < grid.length; i++) {
      grid[i] = data[i * 4] / 255.0;
    }
    return grid;
  };

  const handlePredict = () => {
    setNotice(null);
    const grid = extractCanvasGrid();
    if (!grid) return;
    const regions = segmentGridToDigits(grid, CANVAS_W, CANVAS_H);
    runPredictionOnRegions(regions, mode);
  };

  const handleModeSwitch = (nextMode: PredictionModelMode) => {
    onModeChange(nextMode);
    setNotice(null);
    if (results.length > 0) {
      const regions = results.map((r) => r.region);
      runPredictionOnRegions(regions, nextMode);
    }
  };

  const handleClear = () => {
    clearCanvasSurface();
    setResults([]);
    setSelectedDigitIdx(0);
    setTotalInferenceMs(null);
    setNotice(null);
    setFeedbackStatus(null);
    setRatedThisPrediction(false);
    setCorrectionInput('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setNotice(null);

    try {
      const { regions, previewDataUrl } = await segmentUploadedFileToDigits(
        file,
        CANVAS_W,
        CANVAS_H
      );

      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const img = new Image();
          img.onload = () => {
            ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
            ctx.drawImage(img, 0, 0, CANVAS_W, CANVAS_H);
            hasDrawnRef.current = true;
          };
          img.src = previewDataUrl;
        }
      }

      runPredictionOnRegions(regions, mode);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Failed to load uploaded image.');
    }
  };

  const handleMarkRight = () => {
    if (results.length === 0 || ratedThisPrediction) return;
    onMarkPredictionRight(
      results.map((r) => ({
        pixels28: r.region.pixels28,
        predictedDigit: r.activeOutput.predictedDigit,
      }))
    );
    setRatedThisPrediction(true);
    setFeedbackStatus({
      type: 'success',
      text: `Marked "${rawNumberString}" as correct!`,
    });
  };

  const handleSaveCorrection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (results.length === 0) return;

    const res = await onSubmitCorrectionFeedback(
      results.map((r) => ({
        pixels28: r.region.pixels28,
        predictedDigit: r.activeOutput.predictedDigit,
      })),
      correctionInput
    );

    if (!res.ok) {
      setFeedbackStatus({ type: 'error', text: res.message });
      return;
    }

    setRatedThisPrediction(true);
    setCorrectionInput('');
    setFeedbackStatus({
      type: res.duplicateFlagged ? 'warning' : 'success',
      text: res.message,
    });
  };

  const rawNumberString = results.map((r) => r.activeOutput.predictedDigit).join('');
  const parsedNumber = rawNumberString ? Number.parseInt(rawNumberString, 10) : null;
  const outOfRangeNote =
    parsedNumber !== null && parsedNumber > 100
      ? `Segmented ${results.length} digits (${rawNumberString}) — expected range is 0–100.`
      : null;

  const focusedItem = results[selectedDigitIdx] ?? results[0] ?? null;

  if (isHydratingModels && !svmModel && !cnnModel) {
    return (
      <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 sm:p-6 shadow-sm space-y-5 animate-pulse">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-2">
            <div className="h-5 w-48 rounded-lg bg-slate-200 dark:bg-slate-800" />
            <div className="h-3 w-72 rounded-lg bg-slate-200 dark:bg-slate-800" />
          </div>
          <div className="h-11 w-64 rounded-xl bg-slate-200 dark:bg-slate-800" />
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
          <div className="aspect-[2/1] w-full rounded-2xl bg-slate-200 dark:bg-slate-800/80" />
          <div className="h-80 rounded-2xl bg-slate-200 dark:bg-slate-800/80" />
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-4 sm:p-6 shadow-sm space-y-5">
      {/* Header + Model Mode Switcher */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">
            Draw a Number
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            560×280 pointer canvas · 28×28 MNIST centering · Left-to-right digit segmentation
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div
            role="group"
            aria-label="Prediction model mode"
            className="flex items-center gap-1 rounded-xl bg-slate-100 dark:bg-[#0B0F1D] p-1 border border-slate-200 dark:border-[#1E243D]"
          >
            {(
              [
                { id: 'svm', label: 'SVM Only', ready: Boolean(svmModel) },
                { id: 'cnn', label: 'CNN Only', ready: Boolean(cnnModel) },
                { id: 'ensemble', label: 'Ensemble', ready: Boolean(svmModel && cnnModel) },
              ] as const
            ).map((opt) => {
              const active = mode === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => handleModeSwitch(opt.id)}
                  className={`min-h-[38px] rounded-lg px-3 text-xs font-semibold transition-all whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'bg-indigo-600 text-white shadow-xs'
                      : opt.ready
                      ? 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                      : 'text-slate-400 dark:text-slate-600'
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={handleUpload}
            className="hidden"
            id="multi-digit-upload"
          />
          <label
            htmlFor="multi-digit-upload"
            className="min-h-[44px] inline-flex items-center gap-1.5 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] px-3.5 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap shrink-0"
          >
            <ImageUp className="h-4 w-4 text-indigo-500" />
            Upload Image
          </label>
        </div>
      </div>

      {!svmModel && !cnnModel && (
        <div
          role="status"
          className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between rounded-2xl border border-indigo-200 dark:border-indigo-800/70 bg-indigo-50/80 dark:bg-indigo-950/35 p-4 text-xs text-indigo-900 dark:text-indigo-200"
        >
          <div className="flex items-start gap-2.5">
            <Cpu className="h-4 w-4 shrink-0 text-indigo-600 dark:text-indigo-400 mt-0.5" />
            <div>
              <div className="font-semibold">
                {isSvmTraining || isCnnTraining
                  ? 'Training your first model in the background...'
                  : 'No trained model yet'}
              </div>
              <div className="mt-0.5 text-indigo-800/90 dark:text-indigo-300/90">
                {isSvmTraining || isCnnTraining
                  ? 'You can draw on the canvas now and view live training progress anytime in the Train tab.'
                  : 'Train an SVM or CNN model in the Train tab before running predictions.'}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={onGoToTrain}
            className="min-h-[44px] inline-flex items-center justify-center gap-1.5 rounded-xl bg-indigo-600 px-4 text-xs font-semibold text-white hover:bg-indigo-700 transition-colors cursor-pointer shrink-0"
          >
            <Cpu className="h-4 w-4" />
            Go to Train
          </button>
        </div>
      )}

      {/* Responsive Two-Column Workspace */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px] items-start">
        <div className="space-y-3">
          <div className="relative w-full overflow-hidden rounded-2xl border-2 border-slate-300 dark:border-indigo-500/30 bg-black shadow-inner">
            <canvas
              ref={attachCanvasRef}
              width={CANVAS_W}
              height={CANVAS_H}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              aria-label="Handwritten number drawing canvas"
              className="block w-full h-auto aspect-[2/1] touch-none select-none cursor-crosshair bg-black"
            />
            {/* SVG Overlay showing segmented digit bounding boxes */}
            {results.length > 0 && (
              <svg
                viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
                className="pointer-events-none absolute inset-0 h-full w-full"
              >
                {results.map((item, idx) => {
                  const { minX, minY, maxX, maxY } = item.region.bbox;
                  const pad = 4;
                  const x = Math.max(0, minX - pad);
                  const y = Math.max(0, minY - pad);
                  const w = Math.min(CANVAS_W - x, maxX - minX + 1 + pad * 2);
                  const h = Math.min(CANVAS_H - y, maxY - minY + 1 + pad * 2);
                  const isSelected = idx === selectedDigitIdx;
                  return (
                    <g key={idx}>
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        fill="none"
                        stroke={isSelected ? '#14b8a6' : '#818cf8'}
                        strokeWidth={isSelected ? 2.5 : 1.5}
                        strokeDasharray="4 3"
                        rx={6}
                      />
                      <text
                        x={x + 4}
                        y={Math.max(14, y - 4)}
                        fill={isSelected ? '#2dd4bf' : '#a5b4fc'}
                        fontSize="12"
                        fontFamily="JetBrains Mono, monospace"
                        fontWeight="600"
                      >
                        #{idx + 1}: {item.activeOutput.predictedDigit}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-1 sm:flex-initial items-center gap-2.5">
              <button
                type="button"
                onClick={handleClear}
                className="min-h-[44px] flex-1 sm:flex-initial inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-4 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap"
              >
                <Eraser className="h-4 w-4" />
                Clear
              </button>
              <button
                type="button"
                onClick={handlePredict}
                disabled={!isDatasetReady && !svmModel && !cnnModel}
                className="min-h-[44px] flex-1 sm:flex-initial inline-flex items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-6 text-xs font-semibold text-white hover:from-indigo-700 hover:to-violet-700 disabled:opacity-50 shadow-sm transition-all cursor-pointer whitespace-nowrap"
              >
                <Play className="h-4 w-4" />
                Predict Number
              </button>
            </div>

            <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
              {totalInferenceMs !== null
                ? `${results.length} digit${results.length === 1 ? '' : 's'} · ${totalInferenceMs.toFixed(1)} ms`
                : 'Leave a small gap between adjacent digits'}
            </span>
          </div>

          {notice && (
            <div
              role="status"
              className="rounded-xl border border-amber-300 dark:border-amber-800/70 bg-amber-50 dark:bg-amber-950/35 px-4 py-3 text-xs text-amber-800 dark:text-amber-200"
            >
              {notice}
            </div>
          )}

          {outOfRangeNote && (
            <div className="rounded-xl border border-amber-300 dark:border-amber-800/70 bg-amber-50 dark:bg-amber-950/35 px-4 py-3 text-xs text-amber-800 dark:text-amber-200">
              {outOfRangeNote}
            </div>
          )}

          {/* Quick Feedback Bar right under the canvas/prediction */}
          {results.length > 0 && (
            <div className="rounded-2xl border border-indigo-200/80 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
                  <MessageSquarePlus className="h-4 w-4 text-teal-500" />
                  <span>Quick Feedback for "{rawNumberString}"</span>
                </div>
                <button
                  type="button"
                  onClick={onOpenFeedbackTab}
                  className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer"
                >
                  Open Feedback Lab →
                </button>
              </div>

              <form onSubmit={handleSaveCorrection} className="flex flex-col gap-2.5 sm:flex-row sm:items-center">
                <button
                  type="button"
                  disabled={ratedThisPrediction}
                  onClick={handleMarkRight}
                  className="min-h-[44px] inline-flex items-center justify-center gap-1.5 rounded-xl border border-teal-300 dark:border-teal-800 bg-teal-50 dark:bg-teal-950/40 px-4 text-xs font-semibold text-teal-700 dark:text-teal-300 hover:bg-teal-100 dark:hover:bg-teal-900/50 disabled:opacity-50 transition-colors cursor-pointer whitespace-nowrap shrink-0"
                >
                  {ratedThisPrediction ? <Check className="h-4 w-4" /> : <ThumbsUp className="h-4 w-4" />}
                  {ratedThisPrediction ? 'Recorded' : 'Prediction Right'}
                </button>

                <div className="flex flex-1 items-center gap-2">
                  <input
                    type="text"
                    inputMode="numeric"
                    maxLength={3}
                    value={correctionInput}
                    onChange={(e) => setCorrectionInput(e.target.value)}
                    placeholder={`Wrong? Type ${results.length}-digit fix (0–100)`}
                    aria-label="Enter corrected digits for feedback"
                    className="min-h-[44px] flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] px-3.5 font-mono text-xs text-slate-900 dark:text-white placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none"
                  />
                  <button
                    type="submit"
                    className="min-h-[44px] rounded-xl bg-indigo-600 px-4 text-xs font-semibold text-white hover:bg-indigo-700 transition-colors cursor-pointer whitespace-nowrap shrink-0"
                  >
                    Save Fix
                  </button>
                </div>
              </form>

              {feedbackStatus && (
                <div
                  role="status"
                  className={`rounded-xl px-3.5 py-2 text-xs font-medium ${
                    feedbackStatus.type === 'error'
                      ? 'bg-rose-100 dark:bg-rose-950/50 text-rose-800 dark:text-rose-200'
                      : feedbackStatus.type === 'warning'
                      ? 'bg-amber-100 dark:bg-amber-950/50 text-amber-800 dark:text-amber-200'
                      : 'bg-teal-100 dark:bg-teal-950/50 text-teal-800 dark:text-teal-200'
                  }`}
                >
                  {feedbackStatus.text}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Side: Glowing Predicted Number Card + Horizontal Confidence Bars */}
        <div className="space-y-4">
          <div className="rounded-2xl border border-indigo-300/70 dark:border-indigo-500/40 bg-gradient-to-br from-indigo-600/15 via-violet-600/10 to-teal-500/15 p-5 shadow-[0_0_28px_-6px_rgba(99,102,241,0.35)]">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-indigo-700 dark:text-indigo-300">
                Predicted ({mode.toUpperCase()})
              </span>
              {results.length > 0 && (
                <span className="font-mono text-xs font-bold text-teal-600 dark:text-teal-400 tabular-nums">
                  Avg{' '}
                  {(
                    (results.reduce((a, r) => a + r.activeOutput.confidence, 0) / results.length) *
                    100
                  ).toFixed(1)}
                  %
                </span>
              )}
            </div>

            <div className="mt-2 flex items-baseline justify-between gap-2">
              <span className="font-mono text-5xl font-bold tracking-tight bg-gradient-to-r from-indigo-600 via-violet-500 to-teal-500 dark:from-indigo-300 dark:via-violet-300 dark:to-teal-300 bg-clip-text text-transparent tabular-nums">
                {rawNumberString || '—'}
              </span>
              {results.length > 0 && (
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  {results.length} digit{results.length === 1 ? '' : 's'}
                </span>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-4 space-y-4">
            {/* Segmented 28x28 Digit Strip */}
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2">
                Segmented Digits (28×28)
              </div>
              {results.length === 0 ? (
                <div className="rounded-xl border border-dashed border-slate-300 dark:border-[#1E243D] py-6 text-center text-xs text-slate-400 dark:text-slate-500">
                  Draw a number on the canvas and tap Predict Number.
                </div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {results.map((item, idx) => {
                    const selected = idx === selectedDigitIdx;
                    return (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => setSelectedDigitIdx(idx)}
                        className={`min-h-[44px] flex flex-col items-center rounded-xl border p-2 transition-all cursor-pointer ${
                          selected
                            ? 'border-teal-500 bg-teal-500/10 shadow-xs'
                            : 'border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] hover:border-indigo-400'
                        }`}
                      >
                        <MnistPixelCanvas pixels={item.region.pixels28} displaySize={52} />
                        <span className="mt-1.5 font-mono text-sm font-bold text-slate-900 dark:text-white tabular-nums">
                          {item.activeOutput.predictedDigit}
                        </span>
                        <span className="font-mono text-[10px] text-teal-600 dark:text-teal-400 tabular-nums">
                          {(item.activeOutput.confidence * 100).toFixed(0)}%
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Horizontal Confidence Bars for Focused Digit */}
            {focusedItem && (
              <div className="border-t border-slate-200 dark:border-[#1E243D] pt-3 space-y-2">
                <div className="flex items-center justify-between font-mono text-[11px] text-slate-500 dark:text-slate-400">
                  <span>DIGIT #{selectedDigitIdx + 1} CONFIDENCE</span>
                  {focusedItem.svmOutput && focusedItem.cnnOutput && (
                    <span>
                      SVM: {focusedItem.svmOutput.predictedDigit} · CNN: {focusedItem.cnnOutput.predictedDigit}
                    </span>
                  )}
                </div>
                <div className="space-y-1.5">
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => {
                    const prob = focusedItem.activeOutput.probabilities[d] ?? 0;
                    const isWin = d === focusedItem.activeOutput.predictedDigit;
                    return (
                      <div key={d} className="grid grid-cols-[16px_1fr_42px] items-center gap-2">
                        <span
                          className={`font-mono text-[11px] font-semibold tabular-nums ${
                            isWin
                              ? 'text-teal-600 dark:text-teal-400'
                              : 'text-slate-600 dark:text-slate-400'
                          }`}
                        >
                          {d}
                        </span>
                        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                          <div
                            className={`h-full rounded-full transition-all duration-200 ${
                              isWin
                                ? 'bg-gradient-to-r from-indigo-500 to-teal-400'
                                : 'bg-slate-400/80 dark:bg-slate-600'
                            }`}
                            style={{ width: `${Math.max(2, prob * 100)}%` }}
                          />
                        </div>
                        <span className="text-right font-mono text-[10px] text-slate-600 dark:text-slate-300 tabular-nums">
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
