import { useMemo, useState } from 'react';
import { buildCnnEvaluationFromProbs, HydratedCnnModel } from '../ml/cnnEngine';
import { MnistBrowserDataset } from '../ml/mnistCdnLoader';
import { HydratedSvmModel, ModelEvaluationSummary } from '../ml/svmTypes';

interface ComparisonEvaluationSectionProps {
  dataset: MnistBrowserDataset | null;
  svmModel: HydratedSvmModel | null;
  cnnModel: HydratedCnnModel | null;
}

function ConfusionMatrixCard({
  title,
  accentLabel,
  summary,
}: {
  title: string;
  accentLabel: string;
  summary: ModelEvaluationSummary;
}) {
  const maxMatrixVal = useMemo(() => {
    let m = 1;
    for (const row of summary.confusionMatrix) {
      for (const v of row) if (v > m) m = v;
    }
    return m;
  }, [summary.confusionMatrix]);

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">{title}</h3>
        <span className="font-mono text-xs font-semibold text-teal-600 dark:text-teal-400 tabular-nums">
          {accentLabel}
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[380px]">
          <div className="grid grid-cols-[36px_repeat(10,1fr)] gap-1 mb-1 text-center font-mono text-[10px] font-semibold text-slate-500 dark:text-slate-400">
            <div className="text-right pr-1.5">T\P</div>
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((c) => (
              <div key={c}>{c}</div>
            ))}
          </div>

          <div className="space-y-1">
            {summary.confusionMatrix.map((row, rIdx) => (
              <div key={rIdx} className="grid grid-cols-[36px_repeat(10,1fr)] gap-1 items-center">
                <div className="text-right pr-1.5 font-mono text-[10px] font-semibold text-slate-500 dark:text-slate-400">
                  {rIdx}
                </div>
                {row.map((val, cIdx) => {
                  const isDiag = rIdx === cIdx;
                  const intensity = val / maxMatrixVal;

                  let cellClass =
                    'bg-slate-100 dark:bg-[#0B0F1D] text-slate-400 dark:text-slate-600';
                  let style: React.CSSProperties | undefined;

                  if (val > 0 && isDiag) {
                    const alpha = 0.28 + intensity * 0.72;
                    style = {
                      backgroundColor: `rgba(79, 70, 229, ${alpha.toFixed(2)})`,
                      color: '#ffffff',
                    };
                    cellClass = '';
                  } else if (val > 0 && !isDiag) {
                    const alpha = Math.min(0.85, 0.28 + (val / Math.max(4, maxMatrixVal * 0.12)) * 0.57);
                    style = {
                      backgroundColor: `rgba(225, 29, 72, ${alpha.toFixed(2)})`,
                      color: '#ffffff',
                    };
                    cellClass = '';
                  }

                  return (
                    <div
                      key={cIdx}
                      style={style}
                      title={`Actual ${rIdx}, Predicted ${cIdx}: ${val}`}
                      className={`flex h-7 items-center justify-center rounded-md font-mono text-[11px] font-semibold tabular-nums ${cellClass}`}
                    >
                      {val}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ClassMetricsTable({
  title,
  summary,
}: {
  title: string;
  summary: ModelEvaluationSummary;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left font-mono text-xs tabular-nums">
          <thead>
            <tr className="border-b border-slate-200 dark:border-[#1E243D] text-slate-500 dark:text-slate-400">
              <th className="py-2 font-medium">DIGIT</th>
              <th className="py-2 text-right font-medium">PRECISION</th>
              <th className="py-2 text-right font-medium">RECALL</th>
              <th className="py-2 text-right font-medium">F1</th>
              <th className="py-2 text-right font-medium">SUPPORT</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-[#1E243D]/70">
            {summary.classMetrics.map((row) => (
              <tr key={row.digit}>
                <td className="py-1.5 font-semibold text-slate-900 dark:text-white">{row.digit}</td>
                <td className="py-1.5 text-right text-slate-700 dark:text-slate-300">
                  {(row.precision * 100).toFixed(1)}%
                </td>
                <td className="py-1.5 text-right text-slate-700 dark:text-slate-300">
                  {(row.recall * 100).toFixed(1)}%
                </td>
                <td className="py-1.5 text-right font-semibold text-teal-600 dark:text-teal-400">
                  {(row.f1Score * 100).toFixed(1)}%
                </td>
                <td className="py-1.5 text-right text-slate-500 dark:text-slate-400">{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ComparisonEvaluationSection({
  dataset,
  svmModel,
  cnnModel,
}: ComparisonEvaluationSectionProps) {
  const [activeEvalTab, setActiveEvalTab] = useState<'svm' | 'cnn' | 'ensemble'>('svm');

  const ensembleSummary = useMemo<ModelEvaluationSummary | null>(() => {
    if (!dataset || !svmModel || !cnnModel) return null;
    const svmProbs = svmModel.evaluation.testProbabilities;
    const cnnProbs = cnnModel.evaluation.testProbabilities;
    const len = Math.min(svmProbs.length, cnnProbs.length);
    const avgProbs = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      avgProbs[i] = 0.5 * (svmProbs[i] + cnnProbs[i]);
    }
    return buildCnnEvaluationFromProbs(
      avgProbs,
      dataset.testLabels,
      svmModel.evaluation.avgPredMs + cnnModel.evaluation.avgPredMs
    );
  }, [dataset, svmModel, cnnModel]);

  const svmAcc = svmModel ? svmModel.evaluation.testAccuracy * 100 : null;
  const cnnAcc = cnnModel ? cnnModel.evaluation.testAccuracy * 100 : null;
  const ensAcc = ensembleSummary ? ensembleSummary.testAccuracy * 100 : null;

  const svmTrainSec = svmModel ? svmModel.trainTimeMs / 1000 : null;
  const cnnTrainSec = cnnModel ? cnnModel.trainTimeMs / 1000 : null;
  const maxTrainSec = Math.max(1, svmTrainSec ?? 0, cnnTrainSec ?? 0);

  const svmPredMs = svmModel ? svmModel.evaluation.avgPredMs : null;
  const cnnPredMs = cnnModel ? cnnModel.evaluation.avgPredMs : null;
  const maxPredMs = Math.max(0.1, svmPredMs ?? 0, cnnPredMs ?? 0);

  const selectedSummary =
    activeEvalTab === 'cnn'
      ? cnnModel?.evaluation ?? null
      : activeEvalTab === 'ensemble'
      ? ensembleSummary
      : svmModel?.evaluation ?? cnnModel?.evaluation ?? null;

  const selectedTitle =
    activeEvalTab === 'cnn'
      ? 'CNN (TensorFlow.js)'
      : activeEvalTab === 'ensemble'
      ? 'Ensemble (SVM + CNN Avg)'
      : svmModel
      ? 'SVM (SMO)'
      : 'CNN (TensorFlow.js)';

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="rounded-2xl border border-indigo-200/80 dark:border-[#1E243D] bg-gradient-to-r from-indigo-600/10 via-violet-600/10 to-teal-500/10 p-5 sm:p-6 shadow-xs">
        <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
          Model Comparison: SVM vs. CNN vs. Ensemble
        </h2>
        <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
          Side-by-side benchmark evaluated across the fixed 1,000 held-out MNIST test digits.
        </p>
      </div>

      {/* Metric Summary Cards + Bar Chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-stretch">
        <div className="lg:col-span-5 grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-1">
          <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-4 shadow-xs flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
                SVM (Web Worker SMO)
              </div>
              <div className="mt-0.5 font-mono text-2xl font-bold text-indigo-600 dark:text-indigo-400 tabular-nums">
                {svmAcc !== null ? `${svmAcc.toFixed(2)}%` : 'Not trained'}
              </div>
            </div>
            <div className="text-right font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
              <div>{svmTrainSec !== null ? `${svmTrainSec.toFixed(2)}s fit` : '—'}</div>
              <div>{svmPredMs !== null ? `${svmPredMs.toFixed(2)} ms/digit` : '—'}</div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-4 shadow-xs flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
                CNN (TensorFlow.js)
              </div>
              <div className="mt-0.5 font-mono text-2xl font-bold text-teal-600 dark:text-teal-400 tabular-nums">
                {cnnAcc !== null ? `${cnnAcc.toFixed(2)}%` : 'Not trained'}
              </div>
            </div>
            <div className="text-right font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
              <div>{cnnTrainSec !== null ? `${cnnTrainSec.toFixed(2)}s fit` : '—'}</div>
              <div>{cnnPredMs !== null ? `${cnnPredMs.toFixed(2)} ms/digit` : '—'}</div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-4 shadow-xs flex items-center justify-between">
            <div>
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
                Ensemble (Softmax Avg)
              </div>
              <div className="mt-0.5 font-mono text-2xl font-bold text-violet-600 dark:text-violet-400 tabular-nums">
                {ensAcc !== null ? `${ensAcc.toFixed(2)}%` : 'Train both'}
              </div>
            </div>
            <div className="text-right font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
              <div>0.5·P(SVM) + 0.5·P(CNN)</div>
              <div>
                {svmPredMs !== null && cnnPredMs !== null
                  ? `${(svmPredMs + cnnPredMs).toFixed(2)} ms/digit`
                  : '—'}
              </div>
            </div>
          </div>
        </div>

        {/* Comparative Bar Chart */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-5 shadow-sm flex flex-col justify-between space-y-5">
          <div>
            <div className="flex items-center justify-between text-xs font-semibold text-slate-900 dark:text-white mb-3">
              <span>Test Accuracy (%)</span>
              <span className="font-mono text-[11px] text-slate-400">1,000 test digits</span>
            </div>
            <div className="space-y-2.5 font-mono text-xs tabular-nums">
              {[
                { label: 'SVM', val: svmAcc, color: 'bg-indigo-600' },
                { label: 'CNN', val: cnnAcc, color: 'bg-teal-500' },
                { label: 'Ensemble', val: ensAcc, color: 'bg-gradient-to-r from-indigo-500 to-teal-400' },
              ].map((item) => (
                <div key={item.label} className="grid grid-cols-[74px_1fr_64px] items-center gap-3">
                  <span className="text-slate-600 dark:text-slate-300">{item.label}</span>
                  <div className="h-3 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${item.color}`}
                      style={{ width: `${item.val ?? 0}%` }}
                    />
                  </div>
                  <span className="text-right font-semibold text-slate-900 dark:text-white">
                    {item.val !== null ? `${item.val.toFixed(1)}%` : '—'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 border-t border-slate-200 dark:border-[#1E243D] pt-4">
            <div>
              <div className="text-xs font-semibold text-slate-900 dark:text-white mb-2">
                Training Time (s)
              </div>
              <div className="space-y-2 font-mono text-xs tabular-nums">
                {[
                  { label: 'SVM', val: svmTrainSec, color: 'bg-indigo-600' },
                  { label: 'CNN', val: cnnTrainSec, color: 'bg-teal-500' },
                ].map((item) => (
                  <div key={item.label} className="grid grid-cols-[44px_1fr_56px] items-center gap-2">
                    <span className="text-slate-600 dark:text-slate-300">{item.label}</span>
                    <div className="h-2.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${item.color}`}
                        style={{
                          width: item.val !== null ? `${Math.max(4, (item.val / maxTrainSec) * 100)}%` : '0%',
                        }}
                      />
                    </div>
                    <span className="text-right text-slate-900 dark:text-white">
                      {item.val !== null ? `${item.val.toFixed(2)}s` : '—'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-slate-900 dark:text-white mb-2">
                Prediction Time (ms / digit)
              </div>
              <div className="space-y-2 font-mono text-xs tabular-nums">
                {[
                  { label: 'SVM', val: svmPredMs, color: 'bg-indigo-600' },
                  { label: 'CNN', val: cnnPredMs, color: 'bg-teal-500' },
                ].map((item) => (
                  <div key={item.label} className="grid grid-cols-[44px_1fr_56px] items-center gap-2">
                    <span className="text-slate-600 dark:text-slate-300">{item.label}</span>
                    <div className="h-2.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${item.color}`}
                        style={{
                          width: item.val !== null ? `${Math.max(4, (item.val / maxPredMs) * 100)}%` : '0%',
                        }}
                      />
                    </div>
                    <span className="text-right text-slate-900 dark:text-white">
                      {item.val !== null ? `${item.val.toFixed(2)}ms` : '—'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Confusion Matrices + Per-Class Precision & Recall */}
      {(svmModel || cnnModel) && (
        <div className="space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Confusion Matrix & Per-Class Precision / Recall
            </h3>

            <div className="flex items-center gap-1 rounded-xl bg-slate-100 dark:bg-[#0B0F1D] p-1 border border-slate-200 dark:border-[#1E243D]">
              {(
                [
                  { id: 'svm', label: 'SVM', enabled: Boolean(svmModel) },
                  { id: 'cnn', label: 'CNN', enabled: Boolean(cnnModel) },
                  { id: 'ensemble', label: 'Ensemble', enabled: Boolean(ensembleSummary) },
                ] as const
              ).map((tab) => {
                const active = activeEvalTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    disabled={!tab.enabled}
                    onClick={() => setActiveEvalTab(tab.id)}
                    className={`min-h-[38px] rounded-lg px-3.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-40 ${
                      active
                        ? 'bg-indigo-600 text-white shadow-xs'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>

          {selectedSummary && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
              <div className="lg:col-span-7">
                <ConfusionMatrixCard
                  title={`${selectedTitle} · 10×10 Confusion Matrix`}
                  accentLabel={`${(selectedSummary.testAccuracy * 100).toFixed(2)}% Accuracy`}
                  summary={selectedSummary}
                />
              </div>
              <div className="lg:col-span-5">
                <ClassMetricsTable
                  title={`${selectedTitle} · Precision & Recall`}
                  summary={selectedSummary}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
