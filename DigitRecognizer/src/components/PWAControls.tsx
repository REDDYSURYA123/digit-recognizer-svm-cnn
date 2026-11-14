import { useState } from 'react';
import { Download, Share, WifiOff, X } from 'lucide-react';
import { useOnlineStatus, usePWAInstall } from '../ml/pwaHooks';

export function PWAInstallButton({ compact = false }: { compact?: boolean }) {
  const { isInstallable, isInstalled, isIOS, install } = usePWAInstall();
  const [showIOSGuide, setShowIOSGuide] = useState(false);

  if (isInstalled) return null;
  if (!isInstallable && !isIOS) return null;

  const handleClick = async () => {
    if (isIOS) {
      setShowIOSGuide(true);
    } else {
      await install();
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={handleClick}
        aria-label="Install Digit Recognizer app"
        className={`inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0 min-h-[44px] ${
          compact
            ? 'border border-indigo-200 dark:border-indigo-800/80 bg-indigo-50 dark:bg-indigo-950/60 px-3 text-xs text-indigo-700 dark:text-indigo-200 hover:bg-indigo-100 dark:hover:bg-indigo-900/70'
            : 'bg-indigo-600 px-4 text-xs font-semibold text-white hover:bg-indigo-700 shadow-xs'
        }`}
      >
        <Download className="h-4 w-4" />
        <span>Install App</span>
      </button>

      {showIOSGuide && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="ios-install-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/65 p-4 backdrop-blur-xs"
        >
          <div className="w-full max-w-sm rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xl space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3
                  id="ios-install-title"
                  className="text-base font-semibold text-slate-900 dark:text-white"
                >
                  Install on iPhone / iPad
                </h3>
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Run Digit Recognizer offline from your home screen.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowIOSGuide(false)}
                aria-label="Close install instructions"
                className="inline-flex h-11 w-11 items-center justify-center rounded-xl text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <ol className="space-y-2.5 text-xs text-slate-700 dark:text-slate-200">
              <li className="flex items-center gap-2.5 rounded-xl bg-slate-50 dark:bg-[#0B0F1D] p-3 border border-slate-200 dark:border-[#1E243D]">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-600 font-mono text-[11px] font-bold text-white">
                  1
                </span>
                <span>
                  Tap the <Share className="inline h-3.5 w-3.5 text-indigo-500 mx-0.5" />{' '}
                  <strong>Share</strong> button in Safari.
                </span>
              </li>
              <li className="flex items-center gap-2.5 rounded-xl bg-slate-50 dark:bg-[#0B0F1D] p-3 border border-slate-200 dark:border-[#1E243D]">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-600 font-mono text-[11px] font-bold text-white">
                  2
                </span>
                <span>
                  Scroll down and tap <strong>Add to Home Screen</strong>.
                </span>
              </li>
            </ol>

            <button
              type="button"
              onClick={() => setShowIOSGuide(false)}
              className="w-full min-h-[44px] rounded-xl bg-indigo-600 text-xs font-semibold text-white hover:bg-indigo-700 transition-colors cursor-pointer"
            >
              Got it
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export function OfflineIndicator() {
  const isOnline = useOnlineStatus();
  if (isOnline) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="border-b border-amber-300/80 dark:border-amber-800/80 bg-amber-50 dark:bg-amber-950/60 px-4 py-2 text-xs text-amber-900 dark:text-amber-200"
    >
      <div className="mx-auto flex max-w-[1280px] items-center gap-2 font-medium">
        <WifiOff className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
        <span>
          Offline Mode — running locally from cached app assets and saved IndexedDB models.
        </span>
      </div>
    </div>
  );
}
