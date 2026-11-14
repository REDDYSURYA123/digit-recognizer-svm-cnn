import { useMemo, useState } from 'react';
import { Shuffle } from 'lucide-react';
import { HydratedCnnModel } from '../ml/cnnEngine';
import { MnistBrowserDataset, MNIST_IMAGE_SIZE } from '../ml/mnistCdnLoader';
import { HydratedSvmModel } from '../ml/svmTypes';
import { MnistPixelCanvas } from './MnistPixelCanvas';

interface TestSamplesSectionProps {
  dataset: MnistBrowserDataset | null;
  svmModel: HydratedSvmModel | null;
  cnnModel: HydratedCnnModel | null;
}

const DISPLAY_COUNT = 18;

export function TestSamplesSection({
  dataset,
  svmModel,
  cnnModel,
}: TestSamplesSectionProps) {
  const [filterMode, setFilterMode] = useState<'all' | 'wrong'>('all');
  const [shuffleSeed, setShuffleSeed] = useState<number>(1);

  const sampleIndices = useMemo(() => {
    if (!dataset) return [];
    const total = dataset.testLabels.length;
    const svmPreds = svmModel?.evaluation.testPredictions;
    const cnnPreds = cnnModel?.evaluation.testPredictions;

    const pool: number[] = [];
    for (let i = 0; i < total; i++) {
      if (filterMode === 'wrong') {
        const svmWrong = svmPreds ? svmPreds[i] !== dataset.testLabels[i] : false;
        const cnnWrong = cnnPreds ? cnnPreds[i] !== dataset.testLabels[i] : false;
        if (svmWrong || cnnWrong) pool.push(i);
      } else {
        pool.push(i);
      }
    }

    let s = shuffleSeed * 2654435761;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }

    return pool.slice(0, DISPLAY_COUNT);
  }, [dataset, svmModel, cnnModel, filterMode, shuffleSeed]);

  if (!dataset) return null;

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-sm space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h3 className="text-base font-bold tracking-tight text-slate-900 dark:text-white">
            Browse Fixed Test Samples
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Random digits from the immutable 1,000-sample MNIST test subset comparing SVM and CNN predictions against true labels.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 self-start sm:self-auto">
          <div className="flex items-center gap-1 rounded-xl bg-slate-100 dark:bg-[#0B0F1D] p-1 border border-slate-200 dark:border-[#1E243D]">
            <button
              type="button"
              onClick={() => setFilterMode('all')}
              className={`min-h-[38px] rounded-lg px-3 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                filterMode === 'all'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Random All
            </button>
            <button
              type="button"
              disabled={!svmModel && !cnnModel}
              onClick={() => setFilterMode('wrong')}
              className={`min-h-[38px] rounded-lg px-3 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-45 ${
                filterMode === 'wrong'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Any Model Error
            </button>
          </div>

          <button
            type="button"
            onClick={() => setShuffleSeed((v) => v + 1)}
            className="min-h-[44px] inline-flex items-center gap-1.5 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D] px-4 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer whitespace-nowrap shrink-0"
          >
            <Shuffle className="h-4 w-4 text-indigo-500" />
            Shuffle
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-6">
        {sampleIndices.map((idx) => {
          const actual = dataset.testLabels[idx];
          const svmPred = svmModel ? svmModel.evaluation.testPredictions[idx] : null;
          const cnnPred = cnnModel ? cnnModel.evaluation.testPredictions[idx] : null;
          const svmWrong = svmPred !== null && svmPred !== actual;
          const cnnWrong = cnnPred !== null && cnnPred !== actual;
          const anyWrong = svmWrong || cnnWrong;

          return (
            <div
              key={idx}
              className={`flex flex-col items-center rounded-xl border p-3 ${
                anyWrong
                  ? 'border-rose-400/80 bg-rose-50 dark:bg-rose-950/35'
                  : 'border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#0B0F1D]'
              }`}
            >
              <MnistPixelCanvas
                pixels={dataset.testImages}
                offset={idx * MNIST_IMAGE_SIZE}
                displaySize={72}
              />

              <div className="mt-2 w-full space-y-0.5 font-mono text-[11px] tabular-nums">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Actual</span>
                  <strong className="text-slate-900 dark:text-white">{actual}</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">SVM</span>
                  <span
                    className={
                      svmPred === null
                        ? 'text-slate-400'
                        : svmWrong
                        ? 'font-bold text-rose-600 dark:text-rose-400'
                        : 'font-semibold text-teal-600 dark:text-teal-400'
                    }
                  >
                    {svmPred ?? '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">CNN</span>
                  <span
                    className={
                      cnnPred === null
                        ? 'text-slate-400'
                        : cnnWrong
                        ? 'font-bold text-rose-600 dark:text-rose-400'
                        : 'font-semibold text-teal-600 dark:text-teal-400'
                    }
                  >
                    {cnnPred ?? '—'}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
