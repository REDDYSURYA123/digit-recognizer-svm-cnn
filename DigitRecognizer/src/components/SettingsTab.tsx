import { useState } from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  BarChart2,
  Database,
  MessageSquareOff,
  Moon,
  RotateCcw,
  ShieldAlert,
  Sun,
} from 'lucide-react';
import { PWAInstallButton } from './PWAControls';

export type ResetActionType =
  | 'clear-feedback'
  | 'reset-stats'
  | 'restore-baseline'
  | 'factory-reset';

interface ResetOptionDescriptor {
  id: ResetActionType;
  title: string;
  buttonLabel: string;
  summary: string;
  modalTitle: string;
  willDelete: string[];
  tone: 'indigo' | 'amber' | 'teal' | 'rose';
  icon: typeof MessageSquareOff;
}

const RESET_OPTIONS: ResetOptionDescriptor[] = [
  {
    id: 'clear-feedback',
    title: 'Clear feedback data',
    buttonLabel: 'Clear feedback data',
    summary: 'Remove all user-submitted digit samples and reset user-rated accuracy counters.',
    modalTitle: 'Clear all saved feedback data?',
    willDelete: [
      'All user-corrected 28×28 feedback digit samples stored in IndexedDB.',
      'Duplicate-detection history and saved feedback labels.',
      'User-rated accuracy counters (Right / Total predictions).',
      'Your trained SVM, CNN, and Baseline models are NOT deleted.',
    ],
    tone: 'indigo',
    icon: MessageSquareOff,
  },
  {
    id: 'reset-stats',
    title: 'Reset accuracy stats',
    buttonLabel: 'Reset accuracy stats',
    summary: 'Clear accuracy audit logs and prune intermediate retraining history snapshots.',
    modalTitle: 'Reset accuracy stats & audit logs?',
    willDelete: [
      'All timestamped accuracy audit log entries.',
      'Intermediate non-baseline model version snapshots.',
      'Your currently active SVM/CNN models and Baseline snapshots remain intact.',
    ],
    tone: 'amber',
    icon: BarChart2,
  },
  {
    id: 'restore-baseline',
    title: 'Restore baseline models',
    buttonLabel: 'Restore baseline models',
    summary: 'Revert both SVM and CNN back to their untouched Baseline MNIST snapshots.',
    modalTitle: 'Restore untouched Baseline models?',
    willDelete: [
      'Replaces the active SVM and CNN models with your original Baseline snapshots.',
      'Removes feedback influence from active predictions immediately.',
      'Saved feedback samples remain available if you decide to retrain later.',
    ],
    tone: 'teal',
    icon: RotateCcw,
  },
  {
    id: 'factory-reset',
    title: 'Factory reset',
    buttonLabel: 'Factory reset',
    summary: 'Erase all models, versions, feedback samples, TensorFlow.js weights, and saved settings.',
    modalTitle: 'Perform complete Factory Reset?',
    willDelete: [
      'All trained SVM and CNN weights, including Baseline and version history snapshots.',
      'All saved feedback digits, accuracy audit logs, and user-rated stats.',
      'All TensorFlow.js IndexedDB models and localStorage preferences.',
    ],
    tone: 'rose',
    icon: AlertOctagon,
  },
];

interface SettingsTabProps {
  isDark: boolean;
  onToggleTheme: () => void;
  showAccuracyWarning: boolean;
  onToggleAccuracyWarning: (enabled: boolean) => void;
  feedbackCount: number;
  versionCount: number;
  hasBaseline: boolean;
  onExecuteReset: (action: ResetActionType) => Promise<void>;
}

export function SettingsTab({
  isDark,
  onToggleTheme,
  showAccuracyWarning,
  onToggleAccuracyWarning,
  feedbackCount,
  versionCount,
  hasBaseline,
  onExecuteReset,
}: SettingsTabProps) {
  const [pendingAction, setPendingAction] = useState<ResetOptionDescriptor | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleConfirm = async () => {
    if (!pendingAction) return;
    setIsProcessing(true);
    try {
      await onExecuteReset(pendingAction.id);
      setPendingAction(null);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Appearance & PWA Card */}
      <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 sm:p-6 shadow-xs">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">
              Workspace & Offline Settings
            </h2>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Currently storing {versionCount} model snapshot{versionCount === 1 ? '' : 's'} and{' '}
              {feedbackCount} feedback digit{feedbackCount === 1 ? '' : 's'} in IndexedDB.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              onClick={onToggleTheme}
              className="min-h-[44px] inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] px-4 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
            >
              {isDark ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-indigo-600" />}
              {isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            </button>
            <PWAInstallButton />
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-200 dark:border-[#1E243D] flex items-center justify-between gap-4">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500 mt-0.5" />
            <div>
              <div className="text-xs font-semibold text-slate-900 dark:text-white">
                Low Accuracy Warning (&lt;90%)
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400">
                Show a warning banner when SVM or CNN test accuracy falls below 90%.
              </div>
            </div>
          </div>
          <button
            type="button"
            role="switch"
            aria-label="Toggle low accuracy warning banner"
            aria-checked={showAccuracyWarning}
            onClick={() => onToggleAccuracyWarning(!showAccuracyWarning)}
            className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
              showAccuracyWarning ? 'bg-indigo-600' : 'bg-slate-300 dark:bg-slate-700'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${
                showAccuracyWarning ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
      </div>

      {/* 4 Reset Options */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 px-1">
          <Database className="h-4 w-4 text-indigo-500" />
          <h3 className="text-base font-semibold text-slate-900 dark:text-white">
            Data Management & Reset Controls
          </h3>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {RESET_OPTIONS.map((opt) => {
            const Icon = opt.icon;
            const disabled = opt.id === 'restore-baseline' && !hasBaseline;

            const buttonTone =
              opt.tone === 'rose'
                ? 'bg-rose-600 hover:bg-rose-700 text-white'
                : opt.tone === 'teal'
                ? 'bg-teal-600 hover:bg-teal-700 text-white'
                : opt.tone === 'amber'
                ? 'bg-amber-600 hover:bg-amber-700 text-white'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white';

            return (
              <div
                key={opt.id}
                className="flex flex-col justify-between rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-4"
              >
                <div className="space-y-2">
                  <div className="flex items-center gap-2.5">
                    <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 dark:bg-[#0B0F1D] text-indigo-600 dark:text-indigo-400">
                      <Icon className="h-4 w-4" />
                    </div>
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-white">
                      {opt.title}
                    </h4>
                  </div>
                  <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                    {opt.summary}
                  </p>
                </div>

                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => setPendingAction(opt)}
                  className={`w-full min-h-[44px] inline-flex items-center justify-center gap-2 rounded-xl px-4 text-xs font-semibold transition-colors cursor-pointer disabled:opacity-45 ${buttonTone}`}
                >
                  <Icon className="h-4 w-4" />
                  {opt.buttonLabel}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Explicit Confirmation Modal */}
      {pendingAction && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="reset-confirm-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-xs"
        >
          <div className="w-full max-w-md rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-6 shadow-2xl space-y-4">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-rose-100 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400">
                <ShieldAlert className="h-5 w-5" />
              </div>
              <div>
                <h3
                  id="reset-confirm-title"
                  className="text-base font-bold text-slate-900 dark:text-white"
                >
                  {pendingAction.modalTitle}
                </h3>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Review exactly what will be changed before continuing:
                </p>
              </div>
            </div>

            <ul className="space-y-2 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] p-3.5 text-xs text-slate-700 dark:text-slate-200 list-disc pl-6">
              {pendingAction.willDelete.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>

            <div className="flex flex-col-reverse gap-2.5 sm:flex-row sm:justify-end pt-1">
              <button
                type="button"
                disabled={isProcessing}
                onClick={() => setPendingAction(null)}
                className="min-h-[44px] rounded-xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#0B0F1D] px-4 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isProcessing}
                onClick={handleConfirm}
                className="min-h-[44px] rounded-xl bg-rose-600 px-5 text-xs font-semibold text-white hover:bg-rose-700 disabled:opacity-50 transition-colors cursor-pointer"
              >
                {isProcessing ? 'Applying...' : `Confirm ${pendingAction.buttonLabel}`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
