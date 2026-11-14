import { Cpu, Layers, ShieldCheck, Sparkles } from 'lucide-react';

export function AboutTab() {
  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-indigo-200/80 dark:border-[#1E243D] bg-gradient-to-r from-indigo-600/10 via-violet-600/10 to-teal-500/10 p-5 sm:p-6 shadow-xs">
        <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
          How Digit Recognizer Works
        </h2>
        <p className="mt-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-300 max-w-3xl">
          Everything runs 100% locally in your browser using a custom JavaScript Web Worker for the SVM, TensorFlow.js for the CNN, IndexedDB for model versioning, and Workbox caching for offline PWA reliability.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-2.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-100 dark:bg-indigo-950/70 text-indigo-600 dark:text-indigo-400">
              <Cpu className="h-4 w-4" />
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Support Vector Machine (SMO)
            </h3>
          </div>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            Finds maximum-margin separating hyperplanes across 10 one-vs-rest binary classifiers using Sequential Minimal Optimization (SMO). With optional 50-component PCA and RBF or Polynomial kernel tricks, it separates non-linear handwritten strokes cleanly without deep neural layers.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-2.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-100 dark:bg-teal-950/70 text-teal-600 dark:text-teal-400">
              <Layers className="h-4 w-4" />
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Convolutional Neural Network (CNN)
            </h3>
          </div>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            Processes the 28×28 grayscale grid directly with spatial filters: Conv2D(32, 3×3) → MaxPool → Conv2D(64, 3×3) → MaxPool → Flatten → Dropout(0.3) → Dense(128) → Softmax(10). Pooling makes the CNN naturally resilient to slight shifts and handwriting slant.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-2.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-100 dark:bg-violet-950/70 text-violet-600 dark:text-violet-400">
              <Sparkles className="h-4 w-4" />
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              0–100 Segmentation & Ensemble Fusion
            </h3>
          </div>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            Multi-digit drawings on the 560×280 canvas are split left-to-right via 8-connected components, merging stacked strokes (like 4, 5, or 7) and normalizing each digit to 20×20 centered by center-of-mass inside a 28×28 frame. Ensemble mode averages SVM softmax scores with CNN probabilities.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-xs space-y-2.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-100 dark:bg-teal-950/70 text-teal-600 dark:text-teal-400">
              <ShieldCheck className="h-4 w-4" />
            </div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              ≥90% Accuracy Guard & Rollback
            </h3>
          </div>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            The 1,000-sample MNIST test split stays locked and separate from feedback. Retraining mixes original MNIST training data with capped feedback (≤10%, max 3× shift augmentation) and fine-tunes CNN weights at lr=0.0001. Any update dropping below 90% or &gt;1% vs. the previous version rolls back automatically.
          </p>
        </div>
      </div>
    </div>
  );
}
