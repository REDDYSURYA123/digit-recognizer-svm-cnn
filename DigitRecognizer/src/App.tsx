import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Cpu,
  Info,
  MessageSquarePlus,
  Moon,
  PenTool,
  RefreshCw,
  RotateCcw,
  Settings,
  Sun,
  X,
} from 'lucide-react';
import { AboutTab } from './components/AboutTab';
import { ComparisonEvaluationSection } from './components/ComparisonEvaluationSection';
import { DualTrainingPanel } from './components/DualTrainingPanel';
import { BeforeAfterComparison, FeedbackTab } from './components/FeedbackTab';
import {
  FeedbackSubmissionRegion,
  MultiDigitRecognizerPanel,
  QuickFeedbackResponse,
} from './components/MultiDigitRecognizerPanel';
import { OfflineIndicator, PWAInstallButton } from './components/PWAControls';
import { ResetActionType, SettingsTab } from './components/SettingsTab';
import { TestSamplesSection } from './components/TestSamplesSection';
import {
  clearSavedCnnCache,
  CnnEpochPoint,
  CnnTrainConfig,
  fineTuneCnnModel,
  HydratedCnnModel,
  loadCnnModelWeightsAtUri,
  makeCnnVersionUri,
  saveCnnModelWeightsAtUri,
  trainCnnModel,
} from './ml/cnnEngine';
import { fetchMnistSubsetFromCdn, MnistBrowserDataset } from './ml/mnistCdnLoader';
import {
  AccuracyHistoryEntry,
  addFeedbackSample,
  buildCombinedTrainingSet,
  clearAccuracyStatsOnly,
  clearFeedbackSamplesOnly,
  computeDigitFingerprint,
  factoryResetAllStores,
  FeedbackSampleRecord,
  getMetaValue,
  hydrateSvmFromIdb,
  isNearDuplicateDigit,
  listFeedbackSamples,
  listModelVersions,
  ModelVersionRecord,
  saveModelVersion,
  serializeSvmForIdb,
  setMetaValue,
  UserRatedStats,
  validateFeedbackDigits,
} from './ml/modelDb';
import { clearSavedModelFromLocalStorage } from './ml/svmInference';
import {
  HydratedSvmModel,
  PredictionModelMode,
  SvmTrainConfig,
  WorkerResponseMessage,
  WorkerTrainRequest,
} from './ml/svmTypes';

type AppTabId = 'predict' | 'train' | 'compare' | 'feedback' | 'settings' | 'about';

const NAV_TABS: { id: AppTabId; label: string; icon: typeof PenTool }[] = [
  { id: 'predict', label: 'Predict', icon: PenTool },
  { id: 'train', label: 'Train', icon: Cpu },
  { id: 'compare', label: 'Compare', icon: BarChart3 },
  { id: 'feedback', label: 'Feedback', icon: MessageSquarePlus },
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'about', label: 'About', icon: Info },
];

const LOW_ACC_SUPPRESS_STORAGE_KEY = 'digit_recognizer_suppress_low_acc_warning';

function getDefaultSvmConfig(isMobile: boolean): SvmTrainConfig {
  return {
    kernel: 'rbf',
    C: 5,
    gamma: 0.05,
    sampleSize: isMobile ? 1000 : 5000,
    usePca: true,
  };
}

function getDefaultCnnConfig(isMobile: boolean): CnnTrainConfig {
  return {
    epochs: isMobile ? 3 : 5,
    batchSize: 64,
    learningRate: 0.001,
    sampleSize: isMobile ? 3000 : 10000,
  };
}

function createUid(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [isMobileViewport, setIsMobileViewport] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth < 640 : false
  );

  useEffect(() => {
    const onResize = () => setIsMobileViewport(window.innerWidth < 640);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const [isDark, setIsDark] = useState<boolean>(() => {
    try {
      const saved = localStorage.getItem('digit_recognizer_theme');
      if (saved === 'light') return false;
    } catch {
      // Ignore storage read errors
    }
    return true;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) root.classList.add('dark');
    else root.classList.remove('dark');
    try {
      localStorage.setItem('digit_recognizer_theme', isDark ? 'dark' : 'light');
    } catch {
      // Ignore storage write errors
    }
  }, [isDark]);

  const [activeTab, setActiveTab] = useState<AppTabId>('predict');
  const [isHydratingModels, setIsHydratingModels] = useState<boolean>(true);
  const [suppressLowAccuracyWarning, setSuppressLowAccuracyWarning] = useState<boolean>(() => {
    try {
      return localStorage.getItem(LOW_ACC_SUPPRESS_STORAGE_KEY) === 'true';
    } catch {
      return false;
    }
  });
  const [dontShowAgainChecked, setDontShowAgainChecked] = useState<boolean>(false);
  const [dismissedWarningSig, setDismissedWarningSig] = useState<string | null>(null);
  const [hasTrainedOrRetrained, setHasTrainedOrRetrained] = useState<boolean>(false);
  const [toast, setToast] = useState<{
    id: number;
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  const showToast = useCallback((text: string, type: 'success' | 'error' | 'info' = 'success') => {
    const id = Date.now();
    setToast({ id, type, text });
    window.setTimeout(() => {
      setToast((curr) => (curr?.id === id ? null : curr));
    }, 5200);
  }, []);

  // Dataset state
  const [dataset, setDataset] = useState<MnistBrowserDataset | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [datasetStatusText, setDatasetStatusText] = useState<string>(
    'Loading MNIST dataset from public CDN...'
  );
  const [datasetProgress, setDatasetProgress] = useState<number>(0);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  const [predictionMode, setPredictionMode] = useState<PredictionModelMode>('svm');

  // SVM state
  const [svmConfig, setSvmConfig] = useState<SvmTrainConfig>(() =>
    getDefaultSvmConfig(typeof window !== 'undefined' && window.innerWidth < 640)
  );
  const [svmModel, setSvmModel] = useState<HydratedSvmModel | null>(null);
  const [isSvmTraining, setIsSvmTraining] = useState<boolean>(false);
  const [svmProgress, setSvmProgress] = useState<number>(0);
  const [svmLogs, setSvmLogs] = useState<string[]>([]);
  const [svmCacheInfo, setSvmCacheInfo] = useState<string | null>(null);

  // CNN state
  const [cnnConfig, setCnnConfig] = useState<CnnTrainConfig>(() =>
    getDefaultCnnConfig(typeof window !== 'undefined' && window.innerWidth < 640)
  );
  const [cnnModel, setCnnModel] = useState<HydratedCnnModel | null>(null);
  const [isCnnTraining, setIsCnnTraining] = useState<boolean>(false);
  const [cnnProgress, setCnnProgress] = useState<number>(0);
  const [cnnLogs, setCnnLogs] = useState<string[]>([]);
  const [cnnHistory, setCnnHistory] = useState<CnnEpochPoint[]>([]);
  const [cnnCacheInfo, setCnnCacheInfo] = useState<string | null>(null);

  // Version history, feedback & guarded retraining state
  const [modelVersions, setModelVersions] = useState<ModelVersionRecord[]>([]);
  const [activeSvmVersionId, setActiveSvmVersionId] = useState<string | null>(null);
  const [activeCnnVersionId, setActiveCnnVersionId] = useState<string | null>(null);
  const [feedbackSamples, setFeedbackSamples] = useState<FeedbackSampleRecord[]>([]);
  const [userRatedStats, setUserRatedStats] = useState<UserRatedStats>({
    totalRated: 0,
    correctCount: 0,
  });
  const [accuracyLogs, setAccuracyLogs] = useState<AccuracyHistoryEntry[]>([]);

  const [isRetraining, setIsRetraining] = useState<'svm' | 'cnn' | null>(null);
  const [retrainProgress, setRetrainProgress] = useState<number>(0);
  const [retrainStatusText, setRetrainStatusText] = useState<string | null>(null);
  const [lastComparison, setLastComparison] = useState<BeforeAfterComparison | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const svmJobIdRef = useRef<number>(0);
  const cnnCancelledRef = useRef<boolean>(false);
  const startupCheckedRef = useRef<boolean>(false);

  const appendSvmLog = useCallback((line: string) => {
    setSvmLogs((prev) => [...prev.slice(-60), line]);
  }, []);

  const appendCnnLog = useCallback((line: string) => {
    setCnnLogs((prev) => [...prev.slice(-60), line]);
  }, []);

  const appendAccuracyLog = useCallback(
    async (entry: Omit<AccuracyHistoryEntry, 'id' | 'timestampIso'>) => {
      const full: AccuracyHistoryEntry = {
        ...entry,
        id: createUid('acc'),
        timestampIso: new Date().toISOString(),
      };
      setAccuracyLogs((prev) => {
        const next = [full, ...prev].slice(0, 40);
        setMetaValue('accuracyLogs', next).catch(() => undefined);
        return next;
      });
    },
    []
  );

  const persistNewVersion = useCallback(
    async (params: {
      modelType: 'svm' | 'cnn';
      svm?: HydratedSvmModel;
      cnn?: HydratedCnnModel;
      reasonLabel: string;
      beforeAccuracy: number | null;
      existingVersions: ModelVersionRecord[];
    }): Promise<{ record: ModelVersionRecord; allVersions: ModelVersionRecord[] }> => {
      const sameType = params.existingVersions.filter((v) => v.modelType === params.modelType);
      const hasBaseline = sameType.some((v) => v.isBaseline);
      const nextNum =
        sameType.reduce((max, v) => Math.max(max, v.versionNumber), 0) + 1;
      const versionId = createUid(params.modelType);
      const isBaseline = !hasBaseline;
      const versionLabel = isBaseline
        ? `v${nextNum} · Baseline`
        : `v${nextNum} · ${params.reasonLabel}`;

      let record: ModelVersionRecord;
      if (params.modelType === 'svm' && params.svm) {
        record = {
          id: versionId,
          modelType: 'svm',
          versionNumber: nextNum,
          versionLabel,
          isBaseline,
          createdAtIso: params.svm.trainedAtIso,
          accuracy: params.svm.evaluation.testAccuracy,
          trainTimeMs: params.svm.trainTimeMs,
          svmPayload: serializeSvmForIdb(params.svm),
        };
        await saveModelVersion(record);
        await setMetaValue('activeSvmVersionId', versionId);
        setActiveSvmVersionId(versionId);
      } else if (params.modelType === 'cnn' && params.cnn) {
        const uri = makeCnnVersionUri(versionId);
        await saveCnnModelWeightsAtUri(params.cnn.tfModel, uri);
        record = {
          id: versionId,
          modelType: 'cnn',
          versionNumber: nextNum,
          versionLabel,
          isBaseline,
          createdAtIso: params.cnn.trainedAtIso,
          accuracy: params.cnn.evaluation.testAccuracy,
          trainTimeMs: params.cnn.trainTimeMs,
          cnnUri: uri,
          cnnConfig: params.cnn.config,
          cnnEpochHistory: params.cnn.epochHistory,
          cnnEvaluation: params.cnn.evaluation,
        };
        await saveModelVersion(record);
        await setMetaValue('activeCnnVersionId', versionId);
        setActiveCnnVersionId(versionId);
      } else {
        throw new Error('Invalid model snapshot payload.');
      }

      const allVersions = await listModelVersions();
      setModelVersions(allVersions);

      await appendAccuracyLog({
        modelType: params.modelType,
        action: isBaseline ? 'baseline' : 'retrain',
        beforeAccuracy: params.beforeAccuracy,
        afterAccuracy: record.accuracy,
        accepted: true,
        note: versionLabel,
      });

      return { record, allVersions };
    },
    [appendAccuracyLog]
  );

  const runSvmWorkerOnce = useCallback(
    (
      trainImages: Float32Array,
      trainLabels: Uint8Array,
      testImages: Float32Array,
      testLabels: Uint8Array,
      targetConfig: SvmTrainConfig,
      onProgress: (percent: number, log?: string) => void
    ): Promise<HydratedSvmModel> => {
      return new Promise((resolve, reject) => {
        if (workerRef.current) {
          workerRef.current.terminate();
          workerRef.current = null;
        }

        const jobId = ++svmJobIdRef.current;
        const worker = new Worker(new URL('./ml/svmWorker.ts', import.meta.url), {
          type: 'module',
        });
        workerRef.current = worker;

        worker.onmessage = (e: MessageEvent<WorkerResponseMessage>) => {
          const msg = e.data;
          if (msg.jobId !== svmJobIdRef.current) return;

          if (msg.type === 'progress') {
            onProgress(msg.percent, msg.log);
          } else if (msg.type === 'complete') {
            worker.terminate();
            if (workerRef.current === worker) workerRef.current = null;
            resolve(msg.model);
          } else if (msg.type === 'error') {
            worker.terminate();
            if (workerRef.current === worker) workerRef.current = null;
            reject(new Error(msg.message));
          }
        };

        worker.onerror = (err) => {
          if (jobId !== svmJobIdRef.current) return;
          worker.terminate();
          if (workerRef.current === worker) workerRef.current = null;
          reject(new Error(err.message || 'SVM Web Worker error'));
        };

        const req: WorkerTrainRequest = {
          type: 'train',
          jobId,
          trainImages: new Float32Array(trainImages),
          trainLabels: new Uint8Array(trainLabels),
          testImages: new Float32Array(testImages),
          testLabels: new Uint8Array(testLabels),
          config: targetConfig,
        };

        worker.postMessage(req, [
          req.trainImages.buffer,
          req.trainLabels.buffer,
          req.testImages.buffer,
          req.testLabels.buffer,
        ]);
      });
    },
    []
  );

  const startSvmTraining = useCallback(
    async (targetDataset: MnistBrowserDataset, targetConfig: SvmTrainConfig) => {
      if (isSvmTraining || isRetraining) return;
      setIsSvmTraining(true);
      setSvmProgress(1);
      setSvmLogs([]);

      try {
        const beforeAcc = svmModel?.evaluation.testAccuracy ?? null;
        const model = await runSvmWorkerOnce(
          targetDataset.trainImages.subarray(0, targetConfig.sampleSize * 784),
          targetDataset.trainLabels.subarray(0, targetConfig.sampleSize),
          targetDataset.testImages,
          targetDataset.testLabels,
          targetConfig,
          (pct, log) => {
            setSvmProgress(pct);
            if (log) appendSvmLog(log);
          }
        );

        setSvmModel(model);
        setSvmProgress(100);
        setHasTrainedOrRetrained(true);
        const existing = await listModelVersions();
        const { record } = await persistNewVersion({
          modelType: 'svm',
          svm: model,
          reasonLabel: 'MNIST Train',
          beforeAccuracy: beforeAcc,
          existingVersions: existing,
        });
        setSvmCacheInfo(`Saved ${record.versionLabel} to IndexedDB.`);
      } catch (err) {
        appendSvmLog(`Error: ${err instanceof Error ? err.message : 'SVM training failed.'}`);
      } finally {
        setIsSvmTraining(false);
      }
    },
    [appendSvmLog, isRetraining, isSvmTraining, persistNewVersion, runSvmWorkerOnce, svmModel]
  );

  const startCnnTraining = useCallback(
    async (targetDataset: MnistBrowserDataset, targetConfig: CnnTrainConfig) => {
      if (isCnnTraining || isRetraining) return;
      cnnCancelledRef.current = false;
      setIsCnnTraining(true);
      setCnnProgress(1);
      setCnnLogs([]);
      setCnnHistory([]);

      try {
        const beforeAcc = cnnModel?.evaluation.testAccuracy ?? null;
        const trained = await trainCnnModel(
          targetDataset.trainImages,
          targetDataset.trainLabels,
          targetDataset.testImages,
          targetDataset.testLabels,
          targetConfig,
          {
            onProgress: (pct, log) => {
              setCnnProgress(pct);
              if (log) appendCnnLog(log);
            },
            onEpochPoint: (pt) => {
              setCnnHistory((prev) => [...prev, pt]);
            },
            isCancelled: () => cnnCancelledRef.current,
          }
        );

        const existing = await listModelVersions();
        const { record } = await persistNewVersion({
          modelType: 'cnn',
          cnn: trained,
          reasonLabel: 'MNIST Train',
          beforeAccuracy: beforeAcc,
          existingVersions: existing,
        });

        setCnnModel((prev) => {
          if (prev?.tfModel && prev.tfModel !== trained.tfModel) {
            try {
              prev.tfModel.dispose();
            } catch {
              // Ignore dispose errors
            }
          }
          return trained;
        });
        setCnnProgress(100);
        setHasTrainedOrRetrained(true);
        setCnnCacheInfo(`Saved ${record.versionLabel} weights to IndexedDB.`);
        if (svmModel) setPredictionMode('ensemble');
      } catch (err) {
        appendCnnLog(err instanceof Error ? err.message : 'CNN training failed.');
      } finally {
        setIsCnnTraining(false);
      }
    },
    [appendCnnLog, cnnModel, isCnnTraining, isRetraining, persistNewVersion, svmModel]
  );

  // Hydrate persisted IndexedDB state immediately on mount so user lands on Predict if ready
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const [versions, feedback, rated, logs, savedSvmId, savedCnnId] = await Promise.all([
          listModelVersions(),
          listFeedbackSamples(),
          getMetaValue<UserRatedStats>('userRatedStats', { totalRated: 0, correctCount: 0 }),
          getMetaValue<AccuracyHistoryEntry[]>('accuracyLogs', []),
          getMetaValue<string | null>('activeSvmVersionId', null),
          getMetaValue<string | null>('activeCnnVersionId', null),
        ]);

        if (!mounted) return;
        setModelVersions(versions);
        setFeedbackSamples(feedback);
        setUserRatedStats(rated);
        setAccuracyLogs(logs);

        const svmVer =
          versions.find((v) => v.id === savedSvmId && v.modelType === 'svm') ??
          versions.find((v) => v.modelType === 'svm') ??
          null;
        const cnnVer =
          versions.find((v) => v.id === savedCnnId && v.modelType === 'cnn') ??
          versions.find((v) => v.modelType === 'cnn') ??
          null;

        let restoredSvm: HydratedSvmModel | null = null;
        let restoredCnn: HydratedCnnModel | null = null;

        if (svmVer?.svmPayload) {
          restoredSvm = hydrateSvmFromIdb(svmVer.svmPayload);
          setSvmModel(restoredSvm);
          setSvmConfig(restoredSvm.config);
          setActiveSvmVersionId(svmVer.id);
          setSvmProgress(100);
          setSvmLogs([
            `Restored ${svmVer.versionLabel} from IndexedDB.`,
            `Fixed test accuracy: ${(restoredSvm.evaluation.testAccuracy * 100).toFixed(2)}%`,
          ]);
          setSvmCacheInfo(`Active: ${svmVer.versionLabel}`);
        }

        if (cnnVer?.cnnUri && cnnVer.cnnConfig && cnnVer.cnnEvaluation) {
          try {
            const tfModel = await loadCnnModelWeightsAtUri(cnnVer.cnnUri);
            if (!mounted) {
              tfModel.dispose();
              return;
            }
            restoredCnn = {
              tfModel,
              config: cnnVer.cnnConfig,
              epochHistory: cnnVer.cnnEpochHistory ?? [],
              evaluation: cnnVer.cnnEvaluation,
              trainTimeMs: cnnVer.trainTimeMs,
              trainedAtIso: cnnVer.createdAtIso,
            };
            setCnnModel(restoredCnn);
            setCnnConfig(cnnVer.cnnConfig);
            setCnnHistory(cnnVer.cnnEpochHistory ?? []);
            setActiveCnnVersionId(cnnVer.id);
            setCnnProgress(100);
            setCnnLogs([
              `Restored ${cnnVer.versionLabel} from IndexedDB.`,
              `Fixed test accuracy: ${(cnnVer.cnnEvaluation.testAccuracy * 100).toFixed(2)}%`,
            ]);
            setCnnCacheInfo(`Active: ${cnnVer.versionLabel}`);
          } catch {
            // Ignore missing TF.js weights if user cleared browser storage selectively
          }
        }

        if (restoredSvm && restoredCnn) {
          setPredictionMode('ensemble');
        } else if (restoredSvm) {
          setPredictionMode('svm');
        } else if (restoredCnn) {
          setPredictionMode('cnn');
        }
        setActiveTab('predict');
      } finally {
        if (mounted) setIsHydratingModels(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const loadDataset = useCallback(async () => {
    setDatasetStatus('loading');
    setDatasetError(null);
    setDatasetProgress(5);
    try {
      const ds = await fetchMnistSubsetFromCdn((msg, pct) => {
        setDatasetStatusText(msg);
        setDatasetProgress(pct);
      });
      setDataset(ds);
      setDatasetStatus('ready');
    } catch (err) {
      setDatasetStatus('error');
      setDatasetError(
        err instanceof Error ? err.message : 'Failed to load MNIST dataset from CDN.'
      );
    }
  }, []);

  useEffect(() => {
    loadDataset();
    return () => {
      cnnCancelledRef.current = true;
      if (workerRef.current) workerRef.current.terminate();
    };
  }, [loadDataset]);

  // Auto-train baseline SVM on first run once both dataset and IDB hydration settle
  useEffect(() => {
    if (
      !isHydratingModels &&
      datasetStatus === 'ready' &&
      dataset &&
      !startupCheckedRef.current
    ) {
      startupCheckedRef.current = true;
      if (!svmModel && modelVersions.filter((v) => v.modelType === 'svm').length === 0) {
        const mobile = window.innerWidth < 640;
        const initCfg = getDefaultSvmConfig(mobile);
        setSvmConfig(initCfg);
        startSvmTraining(dataset, initCfg);
      }
    }
  }, [isHydratingModels, datasetStatus, dataset, svmModel, modelVersions, startSvmTraining]);

  const handleClearSvmCache = async () => {
    clearSavedModelFromLocalStorage();
    setSvmModel(null);
    setSvmProgress(0);
    setSvmLogs([]);
    setSvmCacheInfo('Cleared active SVM from session.');
  };

  const handleClearCnnCache = async () => {
    await clearSavedCnnCache();
    setCnnModel((prev) => {
      if (prev?.tfModel) {
        try {
          prev.tfModel.dispose();
        } catch {
          // Ignore
        }
      }
      return null;
    });
    setCnnProgress(0);
    setCnnLogs([]);
    setCnnHistory([]);
    setCnnCacheInfo('Cleared active CNN from session.');
  };

  const handleMarkPredictionRight = useCallback((regions: FeedbackSubmissionRegion[]) => {
    if (regions.length === 0) return;
    setUserRatedStats((prev) => {
      const next = {
        totalRated: prev.totalRated + 1,
        correctCount: prev.correctCount + 1,
      };
      setMetaValue('userRatedStats', next).catch(() => undefined);
      return next;
    });
  }, []);

  const handleSubmitCorrectionFeedback = useCallback(
    async (
      regions: FeedbackSubmissionRegion[],
      typedNumber: string
    ): Promise<QuickFeedbackResponse> => {
      const validation = validateFeedbackDigits(typedNumber, regions.length);
      if (!validation.valid) {
        return { ok: false, message: validation.error ?? 'Invalid digits.' };
      }

      const existing = await listFeedbackSamples();
      let dupCount = 0;
      const workingList = [...existing];

      for (let i = 0; i < regions.length; i++) {
        const pixelsCopy = new Float32Array(regions[i].pixels28);
        const label = validation.digits[i];
        const isDup = isNearDuplicateDigit(pixelsCopy, label, workingList);
        if (isDup) dupCount++;

        const rec: FeedbackSampleRecord = {
          id: createUid('fb'),
          label,
          pixels28: pixelsCopy,
          hash: computeDigitFingerprint(pixelsCopy),
          predictedDigit: regions[i].predictedDigit,
          isDuplicateFlagged: isDup,
          createdAtIso: new Date().toISOString(),
        };
        await addFeedbackSample(rec);
        workingList.unshift(rec);
      }

      const updatedFeedback = await listFeedbackSamples();
      setFeedbackSamples(updatedFeedback);

      setUserRatedStats((prev) => {
        const next = { totalRated: prev.totalRated + 1, correctCount: prev.correctCount };
        setMetaValue('userRatedStats', next).catch(() => undefined);
        return next;
      });

      if (dupCount > 0) {
        return {
          ok: true,
          duplicateFlagged: true,
          message: `Saved ${regions.length} digit${regions.length > 1 ? 's' : ''} (${typedNumber.trim()}). ${dupCount} flagged as near-duplicate.`,
        };
      }

      return {
        ok: true,
        duplicateFlagged: false,
        message: `Saved ${regions.length} corrected digit${regions.length > 1 ? 's' : ''} (${typedNumber.trim()}) to IndexedDB.`,
      };
    },
    []
  );

  // Guarded SVM feedback retrain
  const handleRetrainSvmWithFeedback = useCallback(async () => {
    if (!dataset || !svmModel || feedbackSamples.length === 0 || isRetraining) return;

    setIsRetraining('svm');
    setRetrainProgress(5);
    setRetrainStatusText('Snapshotting current SVM and combining MNIST + feedback...');

    const prevAcc = svmModel.evaluation.testAccuracy;

    try {
      // Ensure current SVM is snapshotted in version history before retraining
      const existing = await listModelVersions();
      if (!activeSvmVersionId || !existing.some((v) => v.id === activeSvmVersionId)) {
        await persistNewVersion({
          modelType: 'svm',
          svm: svmModel,
          reasonLabel: 'Pre-Feedback Snapshot',
          beforeAccuracy: prevAcc,
          existingVersions: existing,
        });
      }

      const combined = buildCombinedTrainingSet(
        dataset.trainImages,
        dataset.trainLabels,
        svmModel.config.sampleSize,
        feedbackSamples
      );

      setRetrainStatusText(
        `Retraining SVM on ${svmModel.config.sampleSize} MNIST + ${combined.addedCount} feedback samples...`
      );

      const candidate = await runSvmWorkerOnce(
        combined.images,
        combined.labels,
        dataset.testImages,
        dataset.testLabels,
        svmModel.config,
        (pct) => setRetrainProgress(pct)
      );

      const newAcc = candidate.evaluation.testAccuracy;
      const drop = prevAcc - newAcc;

      if (newAcc < 0.9 || drop > 0.01) {
        const msg = `Retraining rejected: accuracy would drop to ${(newAcc * 100).toFixed(2)}%. Previous model kept.`;
        setLastComparison({
          modelType: 'svm',
          beforeAccuracy: prevAcc,
          afterAccuracy: newAcc,
          accepted: false,
          message: msg,
          timestampIso: new Date().toISOString(),
        });
        await appendAccuracyLog({
          modelType: 'svm',
          action: 'retrain-rejected',
          beforeAccuracy: prevAcc,
          afterAccuracy: newAcc,
          accepted: false,
          note: `Rejected (threshold >=90% & <=1% drop)`,
        });
        showToast(msg, 'error');
        return;
      }

      setSvmModel(candidate);
      setHasTrainedOrRetrained(true);
      const latestVersions = await listModelVersions();
      const { record } = await persistNewVersion({
        modelType: 'svm',
        svm: candidate,
        reasonLabel: `Feedback (+${combined.addedCount})`,
        beforeAccuracy: prevAcc,
        existingVersions: latestVersions,
      });

      const okMsg = `SVM retrained and accepted (${record.versionLabel}): ${(newAcc * 100).toFixed(2)}%.`;
      setLastComparison({
        modelType: 'svm',
        beforeAccuracy: prevAcc,
        afterAccuracy: newAcc,
        accepted: true,
        message: okMsg,
        timestampIso: new Date().toISOString(),
      });
      setSvmCacheInfo(`Active: ${record.versionLabel}`);
      showToast(okMsg, 'success');
    } catch (err) {
      showToast(
        err instanceof Error ? err.message : 'SVM feedback retraining failed.',
        'error'
      );
    } finally {
      setIsRetraining(null);
      setRetrainProgress(0);
      setRetrainStatusText(null);
    }
  }, [
    activeSvmVersionId,
    appendAccuracyLog,
    dataset,
    feedbackSamples,
    isRetraining,
    persistNewVersion,
    runSvmWorkerOnce,
    showToast,
    svmModel,
  ]);

  // Guarded CNN feedback fine-tune
  const handleRetrainCnnWithFeedback = useCallback(
    async (fineTuneEpochs: 1 | 2) => {
      if (!dataset || !cnnModel || feedbackSamples.length === 0 || isRetraining) return;

      setIsRetraining('cnn');
      setRetrainProgress(5);
      setRetrainStatusText('Snapshotting current CNN and fine-tuning at lr=0.0001...');

      const prevAcc = cnnModel.evaluation.testAccuracy;

      try {
        const existing = await listModelVersions();
        let sourceUri =
          existing.find((v) => v.id === activeCnnVersionId)?.cnnUri ?? null;
        if (!sourceUri) {
          const snap = await persistNewVersion({
            modelType: 'cnn',
            cnn: cnnModel,
            reasonLabel: 'Pre-Feedback Snapshot',
            beforeAccuracy: prevAcc,
            existingVersions: existing,
          });
          sourceUri = snap.record.cnnUri ?? null;
        }

        const combined = buildCombinedTrainingSet(
          dataset.trainImages,
          dataset.trainLabels,
          cnnModel.config.sampleSize,
          feedbackSamples
        );

        const candidate = await fineTuneCnnModel(
          sourceUri,
          cnnModel.tfModel,
          cnnModel.config,
          cnnModel.epochHistory,
          combined.images,
          combined.labels,
          dataset.testImages,
          dataset.testLabels,
          fineTuneEpochs,
          {
            onProgress: (pct, log) => {
              setRetrainProgress(pct);
              if (log) setRetrainStatusText(log);
            },
          }
        );

        const newAcc = candidate.evaluation.testAccuracy;
        const drop = prevAcc - newAcc;

        if (newAcc < 0.9 || drop > 0.01) {
          try {
            candidate.tfModel.dispose();
          } catch {
            // Ignore dispose errors
          }
          const msg = `Retraining rejected: accuracy would drop to ${(newAcc * 100).toFixed(2)}%. Previous model kept.`;
          setLastComparison({
            modelType: 'cnn',
            beforeAccuracy: prevAcc,
            afterAccuracy: newAcc,
            accepted: false,
            message: msg,
            timestampIso: new Date().toISOString(),
          });
          await appendAccuracyLog({
            modelType: 'cnn',
            action: 'retrain-rejected',
            beforeAccuracy: prevAcc,
            afterAccuracy: newAcc,
            accepted: false,
            note: `Rejected (threshold >=90% & <=1% drop)`,
          });
          showToast(msg, 'error');
          return;
        }

        const latestVersions = await listModelVersions();
        const { record } = await persistNewVersion({
          modelType: 'cnn',
          cnn: candidate,
          reasonLabel: `Feedback (+${combined.addedCount})`,
          beforeAccuracy: prevAcc,
          existingVersions: latestVersions,
        });

        setCnnModel((prev) => {
          if (prev?.tfModel && prev.tfModel !== candidate.tfModel) {
            try {
              prev.tfModel.dispose();
            } catch {
              // Ignore
            }
          }
          return candidate;
        });
        setCnnHistory(candidate.epochHistory);
        setHasTrainedOrRetrained(true);
        setCnnCacheInfo(`Active: ${record.versionLabel}`);

        const okMsg = `CNN fine-tuned and accepted (${record.versionLabel}): ${(newAcc * 100).toFixed(2)}%.`;
        setLastComparison({
          modelType: 'cnn',
          beforeAccuracy: prevAcc,
          afterAccuracy: newAcc,
          accepted: true,
          message: okMsg,
          timestampIso: new Date().toISOString(),
        });
        showToast(okMsg, 'success');
      } catch (err) {
        showToast(
          err instanceof Error ? err.message : 'CNN feedback fine-tuning failed.',
          'error'
        );
      } finally {
        setIsRetraining(null);
        setRetrainProgress(0);
        setRetrainStatusText(null);
      }
    },
    [
      activeCnnVersionId,
      appendAccuracyLog,
      cnnModel,
      dataset,
      feedbackSamples,
      isRetraining,
      persistNewVersion,
      showToast,
    ]
  );

  const handleRestoreVersion = useCallback(
    async (versionId: string) => {
      const target = modelVersions.find((v) => v.id === versionId);
      if (!target) return;

      if (target.modelType === 'svm' && target.svmPayload) {
        const hydrated = hydrateSvmFromIdb(target.svmPayload);
        setSvmModel(hydrated);
        setSvmConfig(hydrated.config);
        setActiveSvmVersionId(target.id);
        await setMetaValue('activeSvmVersionId', target.id);
        setSvmCacheInfo(`Restored: ${target.versionLabel}`);
        showToast(`Restored SVM ${target.versionLabel} (${(target.accuracy * 100).toFixed(2)}%).`);
      } else if (
        target.modelType === 'cnn' &&
        target.cnnUri &&
        target.cnnConfig &&
        target.cnnEvaluation
      ) {
        try {
          const tfModel = await loadCnnModelWeightsAtUri(target.cnnUri);
          const restored: HydratedCnnModel = {
            tfModel,
            config: target.cnnConfig,
            epochHistory: target.cnnEpochHistory ?? [],
            evaluation: target.cnnEvaluation,
            trainTimeMs: target.trainTimeMs,
            trainedAtIso: target.createdAtIso,
          };
          setCnnModel((prev) => {
            if (prev?.tfModel && prev.tfModel !== tfModel) {
              try {
                prev.tfModel.dispose();
              } catch {
                // Ignore
              }
            }
            return restored;
          });
          setCnnConfig(target.cnnConfig);
          setCnnHistory(target.cnnEpochHistory ?? []);
          setActiveCnnVersionId(target.id);
          await setMetaValue('activeCnnVersionId', target.id);
          setCnnCacheInfo(`Restored: ${target.versionLabel}`);
          showToast(`Restored CNN ${target.versionLabel} (${(target.accuracy * 100).toFixed(2)}%).`);
        } catch (err) {
          showToast(
            err instanceof Error ? err.message : 'Failed to load CNN version weights.',
            'error'
          );
        }
      }
    },
    [modelVersions, showToast]
  );

  const handleRestoreBaseline = useCallback(
    async (modelType: 'svm' | 'cnn') => {
      const base = modelVersions.find((v) => v.modelType === modelType && v.isBaseline);
      if (!base) {
        showToast(`No ${modelType.toUpperCase()} Baseline snapshot saved yet.`, 'error');
        return;
      }
      await handleRestoreVersion(base.id);
    },
    [handleRestoreVersion, modelVersions, showToast]
  );

  const handleExecuteReset = useCallback(
    async (action: ResetActionType) => {
      if (action === 'clear-feedback') {
        await clearFeedbackSamplesOnly();
        setFeedbackSamples([]);
        setUserRatedStats({ totalRated: 0, correctCount: 0 });
        setLastComparison(null);
        showToast('Feedback digits and user-rated counters cleared.');
        setActiveTab('predict');
      } else if (action === 'reset-stats') {
        await clearAccuracyStatsOnly();
        const remainingVersions = await listModelVersions();
        setModelVersions(remainingVersions);
        setAccuracyLogs([]);
        setLastComparison(null);
        showToast('Accuracy stats and intermediate logs reset.');
        setActiveTab('predict');
      } else if (action === 'restore-baseline') {
        const svmBase = modelVersions.find((v) => v.modelType === 'svm' && v.isBaseline);
        const cnnBase = modelVersions.find((v) => v.modelType === 'cnn' && v.isBaseline);
        if (svmBase) await handleRestoreVersion(svmBase.id);
        if (cnnBase) await handleRestoreVersion(cnnBase.id);
        setLastComparison(null);
        showToast('Untouched Baseline models restored.');
        setActiveTab('predict');
      } else if (action === 'factory-reset') {
        cnnCancelledRef.current = true;
        if (workerRef.current) {
          workerRef.current.terminate();
          workerRef.current = null;
        }
        setCnnModel((prev) => {
          if (prev?.tfModel) {
            try {
              prev.tfModel.dispose();
            } catch {
              // Ignore
            }
          }
          return null;
        });
        await factoryResetAllStores();
        setSvmModel(null);
        setModelVersions([]);
        setActiveSvmVersionId(null);
        setActiveCnnVersionId(null);
        setFeedbackSamples([]);
        setUserRatedStats({ totalRated: 0, correctCount: 0 });
        setAccuracyLogs([]);
        setLastComparison(null);
        setSvmProgress(0);
        setCnnProgress(0);
        setSvmLogs([]);
        setCnnLogs([]);
        setCnnHistory([]);
        setSvmCacheInfo(null);
        setCnnCacheInfo(null);
        setSuppressLowAccuracyWarning(false);
        setDontShowAgainChecked(false);
        setDismissedWarningSig(null);
        setHasTrainedOrRetrained(false);
        const mobile = window.innerWidth < 640;
        setSvmConfig(getDefaultSvmConfig(mobile));
        setCnnConfig(getDefaultCnnConfig(mobile));
        showToast('Factory reset complete. Ready to train fresh Baseline models.');
        setActiveTab('predict');
      }
    },
    [handleRestoreVersion, modelVersions, showToast]
  );

  const lowAccuracyWarning = useMemo(() => {
    const low: string[] = [];
    if (svmModel && svmModel.evaluation.testAccuracy < 0.9) {
      low.push(`SVM (${(svmModel.evaluation.testAccuracy * 100).toFixed(2)}%)`);
    }
    if (cnnModel && cnnModel.evaluation.testAccuracy < 0.9) {
      low.push(`CNN (${(cnnModel.evaluation.testAccuracy * 100).toFixed(2)}%)`);
    }
    return low;
  }, [svmModel, cnnModel]);

  const lowAccuracySignature = lowAccuracyWarning.join('|');
  const showLowAccuracyBanner =
    hasTrainedOrRetrained &&
    (activeTab === 'train' || activeTab === 'compare') &&
    lowAccuracyWarning.length > 0 &&
    !suppressLowAccuracyWarning &&
    dismissedWarningSig !== lowAccuracySignature;

  const handleDismissWarningBanner = () => {
    setDismissedWarningSig(lowAccuracySignature);
    if (dontShowAgainChecked) {
      setSuppressLowAccuracyWarning(true);
      try {
        localStorage.setItem(LOW_ACC_SUPPRESS_STORAGE_KEY, 'true');
      } catch {
        // Ignore storage write errors
      }
    }
  };

  const handleToggleDontShowAgain = (checked: boolean) => {
    setDontShowAgainChecked(checked);
    try {
      if (checked) localStorage.setItem(LOW_ACC_SUPPRESS_STORAGE_KEY, 'true');
      else localStorage.removeItem(LOW_ACC_SUPPRESS_STORAGE_KEY);
    } catch {
      // Ignore storage write errors
    }
  };

  const handleToggleAccuracyWarningSetting = (enabled: boolean) => {
    const suppressed = !enabled;
    setSuppressLowAccuracyWarning(suppressed);
    setDontShowAgainChecked(suppressed);
    if (enabled) {
      setDismissedWarningSig(null);
    }
    try {
      if (suppressed) localStorage.setItem(LOW_ACC_SUPPRESS_STORAGE_KEY, 'true');
      else localStorage.removeItem(LOW_ACC_SUPPRESS_STORAGE_KEY);
    } catch {
      // Ignore storage write errors
    }
  };

  const hasAnyBaseline = modelVersions.some((v) => v.isBaseline);

  return (
    <div className="min-h-screen overflow-x-hidden bg-slate-50 text-slate-900 dark:bg-[#090B14] dark:text-slate-100">
      <OfflineIndicator />

      {/* Sticky Header + Desktop Top Tabs */}
      <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-[#1E243D] bg-white/95 dark:bg-[#090B14]/95 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1280px] items-center justify-between gap-3 px-4 sm:px-6 py-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-teal-500 text-white shadow-xs">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5"
                aria-hidden="true"
              >
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4Z" />
                <path d="m15 5 3 3" />
              </svg>
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-sm sm:text-base font-bold tracking-tight text-slate-900 dark:text-white">
                Digit Recognizer
              </h1>
              <div className="hidden sm:flex items-center gap-3 font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                <span>
                  SVM:{' '}
                  <strong className="text-indigo-600 dark:text-indigo-400">
                    {svmModel ? `${(svmModel.evaluation.testAccuracy * 100).toFixed(2)}%` : '—'}
                  </strong>
                </span>
                <span>·</span>
                <span>
                  CNN:{' '}
                  <strong className="text-teal-600 dark:text-teal-400">
                    {cnnModel ? `${(cnnModel.evaluation.testAccuracy * 100).toFixed(2)}%` : '—'}
                  </strong>
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <PWAInstallButton compact />
            <button
              type="button"
              onClick={() => setIsDark((prev) => !prev)}
              aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              className="min-h-[44px] inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1E243D] bg-slate-50 dark:bg-[#111526] px-3 sm:px-3.5 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {isDark ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-indigo-600" />}
              <span className="hidden sm:inline">{isDark ? 'Light' : 'Dark'}</span>
            </button>
          </div>
        </div>

        {/* Tablet & PC Top Icon Tab Bar (>= 640px) */}
        <nav
          aria-label="Primary sections"
          className="hidden sm:block border-t border-slate-200/80 dark:border-[#1E243D] bg-slate-50/70 dark:bg-[#0B0F1D]/85"
        >
          <div className="mx-auto flex max-w-[1280px] items-center gap-1.5 px-6 py-1.5 overflow-x-auto">
            {NAV_TABS.map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`min-h-[44px] inline-flex items-center gap-2 rounded-xl px-4 text-xs font-semibold transition-all whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'bg-indigo-600 text-white shadow-xs'
                      : 'text-slate-600 dark:text-slate-300 hover:bg-slate-200/60 dark:hover:bg-[#111526] hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                  {tab.id === 'feedback' && feedbackSamples.length > 0 && (
                    <span
                      className={`rounded-md px-1.5 py-0.5 font-mono text-[10px] tabular-nums ${
                        active
                          ? 'bg-white/20 text-white'
                          : 'bg-teal-500/15 text-teal-600 dark:text-teal-400'
                      }`}
                    >
                      {feedbackSamples.length}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-[1280px] px-4 sm:px-6 pt-5 pb-28 sm:pb-12 space-y-5">
        {/* Global < 90% Accuracy Warning Banner (Dismissible) */}
        {showLowAccuracyBanner && (
          <div
            role="alert"
            className="flex flex-col gap-3 rounded-2xl border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 p-4 text-xs text-amber-900 dark:text-amber-200 shadow-xs"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
                <div>
                  <div className="font-semibold">
                    Accuracy Warning: {lowAccuracyWarning.join(' & ')} below 90% threshold
                  </div>
                  <div className="mt-0.5 text-amber-800/90 dark:text-amber-300/90">
                    Try restoring your untouched Baseline snapshot or resetting data in Settings.
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={handleDismissWarningBanner}
                aria-label="Dismiss accuracy warning"
                className="min-h-[44px] min-w-[44px] -mr-1.5 -mt-1.5 inline-flex items-center justify-center rounded-xl text-amber-700 dark:text-amber-300 hover:bg-amber-200/60 dark:hover:bg-amber-900/50 transition-colors cursor-pointer shrink-0"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-amber-200/80 dark:border-amber-800/60 pt-2.5">
              <label className="min-h-[44px] inline-flex items-center gap-2 text-xs font-medium text-amber-900 dark:text-amber-200 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={dontShowAgainChecked}
                  onChange={(e) => handleToggleDontShowAgain(e.target.checked)}
                  className="h-4 w-4 rounded accent-amber-600 cursor-pointer"
                />
                <span>Don't show again</span>
              </label>

              <div className="flex flex-wrap items-center gap-2">
                {hasAnyBaseline && (
                  <button
                    type="button"
                    onClick={() => handleExecuteReset('restore-baseline')}
                    className="min-h-[44px] inline-flex items-center justify-center gap-1.5 rounded-xl bg-amber-600 px-4 text-xs font-semibold text-white hover:bg-amber-700 transition-colors cursor-pointer shrink-0"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Restore baseline
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setActiveTab('settings')}
                  className="min-h-[44px] inline-flex items-center justify-center gap-1.5 rounded-xl border border-amber-400/80 dark:border-amber-700 bg-white/80 dark:bg-[#111526] px-4 text-xs font-semibold text-amber-900 dark:text-amber-200 hover:bg-amber-100/70 dark:hover:bg-amber-950/60 transition-colors cursor-pointer shrink-0"
                >
                  <Settings className="h-3.5 w-3.5" />
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}

        {/* MNIST Dataset Progress / Error */}
        {datasetStatus === 'loading' && (
          <div className="rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] p-4 shadow-xs">
            <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-slate-200 mb-2">
              <span>{datasetStatusText}</span>
              <span className="font-mono tabular-nums">{datasetProgress}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-600 to-teal-500 transition-all duration-150"
                style={{ width: `${datasetProgress}%` }}
              />
            </div>
          </div>
        )}

        {datasetStatus === 'error' && (
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-rose-300 dark:border-rose-900 bg-rose-50 dark:bg-rose-950/35 p-4 text-xs text-rose-800 dark:text-rose-200">
            <div>
              <div className="font-semibold">MNIST Dataset Load Error</div>
              <div className="mt-0.5">{datasetError}</div>
            </div>
            <button
              type="button"
              onClick={loadDataset}
              className="min-h-[44px] inline-flex items-center gap-1.5 rounded-xl bg-rose-600 px-4 text-xs font-semibold text-white hover:bg-rose-700 transition-colors cursor-pointer"
            >
              <RefreshCw className="h-4 w-4" />
              Retry CDN Fetch
            </button>
          </div>
        )}

        {/* Tab Views */}
        {activeTab === 'train' && (
          <DualTrainingPanel
            isDatasetReady={datasetStatus === 'ready'}
            isMobileViewport={isMobileViewport}
            svmConfig={svmConfig}
            onSvmConfigChange={setSvmConfig}
            onTrainSvm={() => dataset && startSvmTraining(dataset, svmConfig)}
            onClearSvmCache={handleClearSvmCache}
            isSvmTraining={isSvmTraining}
            svmProgress={svmProgress}
            svmLogs={svmLogs}
            svmModel={svmModel}
            svmCacheInfo={svmCacheInfo}
            cnnConfig={cnnConfig}
            onCnnConfigChange={setCnnConfig}
            onTrainCnn={() => dataset && startCnnTraining(dataset, cnnConfig)}
            onClearCnnCache={handleClearCnnCache}
            isCnnTraining={isCnnTraining}
            cnnProgress={cnnProgress}
            cnnLogs={cnnLogs}
            cnnHistory={cnnHistory}
            cnnModel={cnnModel}
            cnnCacheInfo={cnnCacheInfo}
          />
        )}

        {activeTab === 'predict' && (
          <MultiDigitRecognizerPanel
            svmModel={svmModel}
            cnnModel={cnnModel}
            mode={predictionMode}
            onModeChange={setPredictionMode}
            isDatasetReady={datasetStatus === 'ready'}
            isHydratingModels={isHydratingModels}
            isSvmTraining={isSvmTraining}
            isCnnTraining={isCnnTraining}
            onMarkPredictionRight={handleMarkPredictionRight}
            onSubmitCorrectionFeedback={handleSubmitCorrectionFeedback}
            onOpenFeedbackTab={() => setActiveTab('feedback')}
            onGoToTrain={() => setActiveTab('train')}
          />
        )}

        {activeTab === 'compare' && (
          <div className="space-y-8">
            <ComparisonEvaluationSection
              dataset={dataset}
              svmModel={svmModel}
              cnnModel={cnnModel}
            />
            <TestSamplesSection
              dataset={dataset}
              svmModel={svmModel}
              cnnModel={cnnModel}
            />
          </div>
        )}

        {activeTab === 'feedback' && (
          <FeedbackTab
            feedbackSamples={feedbackSamples}
            userRatedStats={userRatedStats}
            svmModel={svmModel}
            cnnModel={cnnModel}
            modelVersions={modelVersions}
            activeSvmVersionId={activeSvmVersionId}
            activeCnnVersionId={activeCnnVersionId}
            accuracyLogs={accuracyLogs}
            isDatasetReady={datasetStatus === 'ready'}
            isSvmTraining={isSvmTraining}
            isCnnTraining={isCnnTraining}
            isRetraining={isRetraining}
            retrainProgress={retrainProgress}
            retrainStatusText={retrainStatusText}
            lastComparison={lastComparison}
            onRetrainSvmWithFeedback={handleRetrainSvmWithFeedback}
            onRetrainCnnWithFeedback={handleRetrainCnnWithFeedback}
            onRestoreVersion={handleRestoreVersion}
            onRestoreBaseline={handleRestoreBaseline}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsTab
            isDark={isDark}
            onToggleTheme={() => setIsDark((prev) => !prev)}
            showAccuracyWarning={!suppressLowAccuracyWarning && !dontShowAgainChecked}
            onToggleAccuracyWarning={handleToggleAccuracyWarningSetting}
            feedbackCount={feedbackSamples.length}
            versionCount={modelVersions.length}
            hasBaseline={hasAnyBaseline}
            onExecuteReset={handleExecuteReset}
          />
        )}

        {activeTab === 'about' && <AboutTab />}
      </main>

      {/* Fixed Bottom Tab Bar for Mobile (< 640px) */}
      <nav
        aria-label="Mobile bottom navigation"
        className="sm:hidden fixed inset-x-0 bottom-0 z-40 border-t border-slate-200 dark:border-[#1E243D] bg-white/95 dark:bg-[#0B0F1D]/95 backdrop-blur-md pb-[env(safe-area-inset-bottom)]"
      >
        <div className="grid grid-cols-6 gap-0.5 px-1 py-1">
          {NAV_TABS.map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`min-h-[48px] flex flex-col items-center justify-center gap-0.5 rounded-xl px-1 text-[10px] font-semibold transition-colors cursor-pointer ${
                  active
                    ? 'bg-indigo-600/15 text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span className="truncate max-w-full">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Toast Notification */}
      {toast && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-20 sm:bottom-6 right-4 left-4 sm:left-auto sm:max-w-md z-50 flex items-center justify-between gap-3 rounded-2xl border border-slate-200 dark:border-[#1E243D] bg-white dark:bg-[#111526] px-4 py-3 shadow-xl"
        >
          <div className="flex items-center gap-2.5 text-xs font-medium text-slate-800 dark:text-slate-100">
            {toast.type === 'error' ? (
              <AlertTriangle className="h-4 w-4 shrink-0 text-rose-500" />
            ) : (
              <CheckCircle2 className="h-4 w-4 shrink-0 text-teal-500" />
            )}
            <span>{toast.text}</span>
          </div>
          <button
            type="button"
            onClick={() => setToast(null)}
            aria-label="Dismiss notification"
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-white cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}
