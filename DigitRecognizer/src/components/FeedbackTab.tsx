import { useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  History,
  RotateCcw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { HydratedCnnModel } from '../ml/cnnEngine';
import {
  AccuracyHistoryEntry,
  FeedbackSampleRecord,
  ModelVersionRecord,
  UserRatedStats,
} from '../ml/modelDb';
import { HydratedSvmModel } from '../ml/svmTypes';
import { MnistPixelCanvas } from './MnistPixelCanvas';

export interface BeforeAfterComparison {
  modelType: 'svm' | 'cnn';
  beforeAccuracy: number;
  afterAccuracy: number;
  accepted: boolean;
  message: string;
  timestampIso: string;
}

interface FeedbackTabProps {
  feedbackSamples: FeedbackSampleRecord[];
  userRatedStats: UserRatedStats;
  svmModel: HydratedSvmModel | null;
  cnnModel: HydratedCnnModel | null;
  modelVersions: ModelVersionRecord[];
  activeSvmVersionId: string | null;
  activeCnnVersionId: string | null;
  accuracyLogs: AccuracyHistoryEntry[];
  isDatasetReady: boolean;
  isSvmTraining: boolean;
  isCnnTraining: boolean;
  isRetraining: 'svm' | 'cnn' | null;
  retrainProgress: number;
  retrainStatusText: string | null;
  lastComparison: BeforeAfterComparison | null;
  onRetrainSvmWithFeedback: () => void;
  onRetrainCnnWithFeedback: (epochs: 1 | 2) => void;
  onRestoreVersion: (versionId: string) => void;
  onRestoreBaseline: (modelType: 'svm' | 'cnn') => void;
}

export function FeedbackTab({
  feedbackSamples,
  userRatedStats,
  svmModel,
  cnnModel,
  modelVersions,
  activeSvmVersionId,
  activeCnnVersionId,
  accuracyLogs,
  isDatasetReady,
  isSvmTraining,
  isCnnTraining,
  isRetraining,
  retrainProgress,
  retrainStatusText,
  lastComparison,
  onRetrainSvmWithFeedback,
  onRetrainCnnWithFeedback,
  onRestoreVersion,
  onRestoreBaseline,
}: FeedbackTabProps) {
  const [fineTuneEpochs, setFineTuneEpochs] = useState<1 | 2>(2);

  const userRatedRate =
    userRatedStats.totalRated > 0
      ? (userRatedStats.correctCount / userRatedStats.totalRated) * 100
      : null;

  const svmBaseline = modelVersions.find((v) => v.modelType === 'svm' && v.isBaseline);
  const cnnBaseline = modelVersions.find((v) => v.modelType === 'cnn' && v.isBaseline);
  const duplicateCount = feedbackSamples.filter((f) => f.isDuplicateFlagged).length;

  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="rounded-2xl border border-indigo-200/80 dark:border-[#1E243D] bg-gradient-to-r from-indigo-600/10 via-violet-600/10 to-teal-500/10 p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
              <ShieldCheck className="h-4 w-4" />
              Guarded Retraining & Versioning
            </div>
            <h2 className="mt-1 text-xl font-bold tracking-tight text-slate-900 dark:text-white">
              Feedback Lab & Model Protection
            </h2>
            <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-300 max-w-2xl">
              Saved corrections combine with original MNIST training data (capped at 10%, max 3× augmentation). Candidates must keep fixed test accuracy at ≥90% with ≤1% drop or auto-rollback.
            </p>
          </div>

          {/* Live Counters */}
          <div className="grid grid-cols-3 gap-2.5 shrink-0 font-mono tabular-nums">
            <div className="rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white/90 dark:bg-[#111526]/90 px-3 py-2.5 text-center">
              <div className="text-[10px] text-slate-500 dark:text-slate-400">SAMPLES</div>
              <div className="text-lg font-bold text-slate-900 dark:text-white">
                {feedbackSamples.length}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white/90 dark:bg-[#111526]/90 px-3 py-2.5 text-center">
              <div className="text-[10px] text-slate-500 dark:text-slate-400">USER ACC</div>
              <div className="text-lg font-bold text-teal-600 dark:text-teal-400">
                {userRatedRate !== null ? `${userRatedRate.toFixed(0)}%` : '—'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white/90 dark:bg-[#111526]/90 px-3 py-2.5 text-center">
              <div className="text-[10px] text-slate-500 dark:text-slate-400">VERSIONS</div>
              <div className="text-lg font-bold text-indigo-600 dark:text-indigo-400">
                {modelVersions.length}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Rejection / Acceptance Before vs After Card */}
      {lastComparison && (
        <div
          role="status"
          className={`rounded-2xl border p-5 shadow-xs space-y-4 ${
            lastComparison.accepted
              ? 'border-teal-300 dark:border-teal-800/80 bg-teal-50/70 dark:bg-teal-950/30'
              : 'border-rose-300 dark:border-rose-800/80 bg-rose-50/80 dark:bg-rose-950/35'
          }`}
        >
          <div className="flex items-start gap-3">
            {lastComparison.accepted ? (
              <CheckCircle2 className="h-5 w-5 shrink-0 text-teal-600 dark:text-teal-400 mt-0.5" />
            ) : (
              <AlertTriangle className="h-5 w-5 shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
            )}
            <div className="flex-1">
              <div className="text-sm font-semibold text-slate-900 dark:text-white">
                {lastComparison.message}
              </div>
              <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                Evaluated on the fixed 1,000-sample MNIST test set ·{' '}
                {new Date(lastComparison.timestampIso).toLocaleTimeString()}
              </div>
            </div>
          </div>

          {/* Before vs After Accuracy Card */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 rounded-xl border border-slate-200/80 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-4 font-mono tabular-nums">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Before ({lastComparison.modelType.toUpperCase()})
              </div>
              <div className="mt-0.5 text-xl font-bold text-slate-900 dark:text-white">
                {(lastComparison.beforeAccuracy * 100).toFixed(2)}%
              </div>
            </div>
            <div className="flex items-center gap-2">
              <ArrowRight className="hidden sm:block h-4 w-4 text-slate-400 shrink-0" />
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Candidate After
                </div>
                <div
                  className={`mt-0.5 text-xl font-bold ${
                    lastComparison.accepted
                      ? 'text-teal-600 dark:text-teal-400'
                      : 'text-rose-600 dark:text-rose-400'
                  }`}
                >
                  {(lastComparison.afterAccuracy * 100).toFixed(2)}%
                </div>
              </div>
            </div>
            <div className="sm:text-right">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Outcome
              </div>
              <div
                className={`mt-0.5 text-sm font-bold ${
                  lastComparison.accepted
                    ? 'text-teal-600 dark:text-teal-400'
                    : 'text-rose-600 dark:text-rose-400'
                }`}
              >
                {lastComparison.accepted ? 'ACCEPTED' : 'ROLLED BACK'} (
                {lastComparison.afterAccuracy - lastComparison.beforeAccuracy >= 0 ? '+' : ''}
                {((lastComparison.afterAccuracy - lastComparison.beforeAccuracy) * 100).toFixed(2)}%)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Retrain Actions + Saved Feedback Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Retrain Controls (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-4">
          <h3 className="text-base font-semibold text-slate-900 dark:text-white">
            Retrain with Feedback
          </h3>
          <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            Combines original MNIST training images with up to 10% augmented feedback samples. Your current model is snapshotted before retraining begins.
          </p>

          <div className="space-y-3 pt-1">
            <button
              type="button"
              disabled={
                !isDatasetReady ||
                !svmModel ||
                feedbackSamples.length === 0 ||
                isSvmTraining ||
                isCnnTraining ||
                isRetraining !== null
              }
              onClick={onRetrainSvmWithFeedback}
              className="w-full min-h-[44px] inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-indigo-700 disabled:opacity-45 transition-colors cursor-pointer"
            >
              <Sparkles className="h-4 w-4" />
              {isRetraining === 'svm'
                ? `Retraining SVM (${retrainProgress}%)...`
                : 'Retrain SVM with Feedback'}
            </button>

            <div className="rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-3 space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-700 dark:text-slate-300">
                  CNN Fine-Tune (lr = 0.0001)
                </span>
                <div className="flex items-center gap-1 rounded-lg bg-white dark:bg-[#111526] p-1 border border-slate-200 dark:border-[#1E243D]">
                  {([1, 2] as const).map((ep) => (
                    <button
                      key={ep}
                      type="button"
                      onClick={() => setFineTuneEpochs(ep)}
                      className={`min-h-[32px] rounded-md px-2.5 font-mono text-[11px] font-medium cursor-pointer transition-colors ${
                        fineTuneEpochs === ep
                          ? 'bg-teal-600 text-white'
                          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                      }`}
                    >
                      {ep} ep
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button"
                disabled={
                  !isDatasetReady ||
                  !cnnModel ||
                  feedbackSamples.length === 0 ||
                  isSvmTraining ||
                  isCnnTraining ||
                  isRetraining !== null
                }
                onClick={() => onRetrainCnnWithFeedback(fineTuneEpochs)}
                className="w-full min-h-[44px] inline-flex items-center justify-center gap-2 rounded-xl bg-teal-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-teal-700 disabled:opacity-45 transition-colors cursor-pointer"
              >
                <Sparkles className="h-4 w-4" />
                {isRetraining === 'cnn'
                  ? `Fine-Tuning CNN (${retrainProgress}%)...`
                  : `Retrain CNN (${fineTuneEpochs} epoch${fineTuneEpochs > 1 ? 's' : ''})`}
              </button>
            </div>
          </div>

          {(isRetraining !== null || retrainStatusText) && (
            <div className="space-y-1.5 border-t border-slate-200 dark:border-[#1E243D] pt-3">
              <div className="flex justify-between font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                <span>{retrainStatusText ?? 'Processing...'}</span>
                <span>{retrainProgress}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-indigo-600 to-teal-500 transition-all duration-150"
                  style={{ width: `${retrainProgress}%` }}
                />
              </div>
            </div>
          )}

          {feedbackSamples.length === 0 && (
            <div className="rounded-xl border border-dashed border-slate-300 dark:border-[#1E243D] p-3 text-xs text-slate-500 dark:text-slate-400">
              No feedback digits saved yet. Draw a number in the <strong>Predict</strong> tab and use the quick feedback bar to collect samples.
            </div>
          )}
        </div>

        {/* Saved Feedback Gallery (7 cols) */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                Saved Feedback Digits ({feedbackSamples.length})
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Validated 0–9 labels stored offline in IndexedDB
                {duplicateCount > 0 ? ` · ${duplicateCount} near-duplicate flagged` : ''}.
              </p>
            </div>
            <div className="font-mono text-xs text-slate-500 dark:text-slate-400 tabular-nums">
              User Rating: {userRatedStats.correctCount}/{userRatedStats.totalRated} right
            </div>
          </div>

          {feedbackSamples.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-300 dark:border-[#1E243D] py-10 text-center text-xs text-slate-400 dark:text-slate-500">
              Corrected and confirmed digits from Predict will appear here.
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-2.5 sm:grid-cols-5 md:grid-cols-6 max-h-72 overflow-y-auto pr-1">
              {feedbackSamples.map((sample) => (
                <div
                  key={sample.id}
                  className="flex flex-col items-center rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-2.5"
                >
                  <MnistPixelCanvas pixels={sample.pixels28} displaySize={52} />
                  <div className="mt-1.5 flex items-center gap-1.5 font-mono text-xs tabular-nums">
                    <span className="font-bold text-indigo-600 dark:text-indigo-400">
                      Label {sample.label}
                    </span>
                  </div>
                  <div className="font-mono text-[10px] text-slate-500 dark:text-slate-400 tabular-nums">
                    Pred {sample.predictedDigit}
                  </div>
                  {sample.isDuplicateFlagged && (
                    <span className="mt-1 rounded-md bg-amber-100 dark:bg-amber-950/60 px-1.5 py-0.5 text-[9px] font-medium text-amber-800 dark:text-amber-300">
                      Duplicate
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Model Version History & Restore Baseline */}
      <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <History className="h-4 w-4 text-indigo-500" />
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Model Version History & Rollback
            </h3>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {svmBaseline && (
              <button
                type="button"
                onClick={() => onRestoreBaseline('svm')}
                className="min-h-[44px] inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-950/40 px-3.5 text-xs font-semibold text-indigo-700 dark:text-indigo-200 hover:bg-indigo-100 dark:hover:bg-indigo-900/50 transition-colors cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Restore SVM Baseline
              </button>
            )}
            {cnnBaseline && (
              <button
                type="button"
                onClick={() => onRestoreBaseline('cnn')}
                className="min-h-[44px] inline-flex items-center gap-1.5 rounded-xl border border-teal-200 dark:border-teal-800 bg-teal-50 dark:bg-teal-950/40 px-3.5 text-xs font-semibold text-teal-700 dark:text-teal-200 hover:bg-teal-100 dark:hover:bg-teal-900/50 transition-colors cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Restore CNN Baseline
              </button>
            )}
          </div>
        </div>

        {modelVersions.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 dark:border-[#1E243D] py-8 text-center text-xs text-slate-400 dark:text-slate-500">
            Train your SVM or CNN model to create Baseline snapshots.
          </div>
        ) : (
          <div className="space-y-2.5">
            {modelVersions.map((ver) => {
              const isActive =
                (ver.modelType === 'svm' && ver.id === activeSvmVersionId) ||
                (ver.modelType === 'cnn' && ver.id === activeCnnVersionId);
              return (
                <div
                  key={ver.id}
                  className={`flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between rounded-xl border p-3.5 transition-colors ${
                    isActive
                      ? 'border-indigo-400 dark:border-indigo-600 bg-indigo-50/50 dark:bg-indigo-950/25'
                      : 'border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D]'
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span
                      className={`rounded-lg px-2 py-1 font-mono text-[11px] font-bold uppercase ${
                        ver.modelType === 'svm'
                          ? 'bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300'
                          : 'bg-teal-100 dark:bg-teal-950 text-teal-700 dark:text-teal-300'
                      }`}
                    >
                      {ver.modelType}
                    </span>
                    <div>
                      <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-900 dark:text-white">
                        <span>{ver.versionLabel}</span>
                        {ver.isBaseline && (
                          <span className="rounded-md bg-slate-200 dark:bg-slate-800 px-1.5 py-0.5 text-[10px] font-medium text-slate-700 dark:text-slate-300">
                            Baseline
                          </span>
                        )}
                        {isActive && (
                          <span className="rounded-md bg-teal-600 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="mt-0.5 font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                        {new Date(ver.createdAtIso).toLocaleString()} · Accuracy:{' '}
                        <strong className="text-slate-800 dark:text-slate-200">
                          {(ver.accuracy * 100).toFixed(2)}%
                        </strong>
                      </div>
                    </div>
                  </div>

                  {!isActive && (
                    <button
                      type="button"
                      onClick={() => onRestoreVersion(ver.id)}
                      className="min-h-[44px] inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] px-4 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:border-indigo-400 hover:text-indigo-600 dark:hover:text-indigo-300 transition-colors cursor-pointer shrink-0"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Restore this version
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Recent Retrain Accuracy Logs */}
        {accuracyLogs.length > 0 && (
          <div className="border-t border-slate-200 dark:border-[#1E243D] pt-4">
            <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2">
              Accuracy Guard Audit Log
            </div>
            <div className="space-y-1.5 max-h-40 overflow-y-auto font-mono text-[11px] tabular-nums">
              {accuracyLogs.slice(0, 12).map((entry) => (
                <div
                  key={entry.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 dark:bg-[#0B0F1D] px-3 py-1.5 border border-slate-200/70 dark:border-[#1E243D]"
                >
                  <span className="text-slate-700 dark:text-slate-300">
                    [{entry.modelType.toUpperCase()}] {entry.note}
                  </span>
                  <span
                    className={
                      entry.accepted
                        ? 'text-teal-600 dark:text-teal-400 font-semibold'
                        : 'text-rose-600 dark:text-rose-400 font-semibold'
                    }
                  >
                    {entry.beforeAccuracy !== null
                      ? `${(entry.beforeAccuracy * 100).toFixed(2)}% -> `
                      : ''}
                    {(entry.afterAccuracy * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
