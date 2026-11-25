import { useEffect, useMemo, useState, type ChangeEvent } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Code2,
  Database,
  Download,
  FileText,
  Play,
  ShieldAlert,
  SlidersHorizontal,
  Upload,
} from 'lucide-react';
import {
  DatasetRow,
  FullAuditOutput,
  runClientSideAudit,
} from './fairnessEngine';
import { AuditOverviewCharts } from './components/AuditOverviewCharts';
import {
  DiagnosticsPlotlyCharts,
  MitigationTradeoffCharts,
} from './components/SecondaryCharts';

interface DatasetInfo {
  key: string;
  name: string;
  sourceType: 'synthetic' | 'historical' | 'uploaded';
  domain: string;
  targetColumn: string;
  filename: string;
  totalRows: number;
}

interface ProjectFileEntry {
  path: string;
  sizeBytes: number;
}

type WorkspaceView = 'overview' | 'mitigation' | 'diagnostics' | 'codebase';

function buildOfflineSummary(
  datasetName: string,
  audit: FullAuditOutput,
): string {
  const m = audit.modelMetrics;
  const lines = [
    `Offline Audit Summary — ${datasetName} (${m.algorithm})`,
    `Model Performance: Accuracy ${(m.accuracy * 100).toFixed(1)}%, Precision ${(m.precision * 100).toFixed(1)}%, Recall ${(m.recall * 100).toFixed(1)}%, F1 ${m.f1Score.toFixed(3)}, ROC-AUC ${m.rocAuc.toFixed(3)}.`,
    `Composite Bias Risk Score: ${audit.biasRiskScore}/100 (${audit.riskLevel}).`,
    ...audit.attributeAudits.map(
      (a) =>
        `${a.attribute} [${a.status}]: Demographic Parity Diff = ${a.dpDiff.toFixed(3)}, DP Ratio = ${a.dpRatio.toFixed(3)}, Equal Opportunity Diff = ${a.eoDiff.toFixed(3)}, Equalized Odds Diff = ${a.eqOddsDiff.toFixed(3)}.`,
    ),
    'Advisory Disclaimer: Measured disparities reflect statistical associations in evaluation data rather than causal proof. Review proxy features and mitigation trade-offs before deployment.',
  ];
  return lines.join('\n\n');
}

function buildMarkdownReport(
  spec: DatasetInfo,
  audit: FullAuditOutput,
  narrativeText: string | null,
): string {
  const m = audit.modelMetrics;
  const lines: string[] = [
    `# FairLens AI Audit Report — ${spec.name}`,
    '',
    `- **Domain:** ${spec.domain}`,
    `- **Source Type:** ${spec.sourceType.toUpperCase()}`,
    `- **Test Split Records:** ${audit.testCount}`,
    `- **Active Model:** ${m.algorithm}`,
    `- **Composite Bias Risk Score:** ${audit.biasRiskScore}/100 (${audit.riskLevel})`,
    '',
    '## Model Evaluation',
    `- Accuracy: ${(m.accuracy * 100).toFixed(2)}%`,
    `- Precision: ${(m.precision * 100).toFixed(2)}%`,
    `- Recall: ${(m.recall * 100).toFixed(2)}%`,
    `- F1-Score: ${m.f1Score.toFixed(4)}`,
    `- ROC-AUC: ${m.rocAuc.toFixed(4)}`,
    '',
    '## Fairness Metrics by Protected Attribute',
    ...audit.attributeAudits.map(
      (a) =>
        `- **${a.attribute} [${a.status}]**: DP Diff=${a.dpDiff.toFixed(4)}, DP Ratio=${a.dpRatio.toFixed(4)}, EO Diff=${a.eoDiff.toFixed(4)}, EqOdds Diff=${a.eqOddsDiff.toFixed(4)}`,
    ),
    '',
    '## Bias Mitigation Trade-Offs',
    ...audit.mitigationComparison.map(
      (r) =>
        `- **${r.strategy}**: Acc=${(r.accuracy * 100).toFixed(1)}%, DP Diff=${r.dpDiff.toFixed(4)}, EqOdds Diff=${r.eqOddsDiff.toFixed(4)}`,
    ),
    '',
    '## Individual Fairness & Root Cause',
    `- k-NN Consistency Score: ${audit.knnConsistencyScore.toFixed(4)}`,
    `- Counterfactual Flip Rate: ${(audit.counterfactualFlipRate * 100).toFixed(1)}%`,
    ...audit.rootCauses.map(
      (rc) => `- **${rc.driver} (${rc.contributionPct}%):** ${rc.evidence}`,
    ),
    '',
    '## Audit Summary',
    narrativeText || buildOfflineSummary(spec.name, audit),
  ];
  return lines.join('\n');
}

function parseClientCsv(text: string): {
  rows: DatasetRow[];
  binaryTargetCandidates: string[];
  headers: string[];
} {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) {
    throw new Error('CSV file must include a header row and data rows.');
  }
  const headers = lines[0].split(',').map((h) => h.trim());
  const rows: DatasetRow[] = [];
  for (let i = 1; i < Math.min(lines.length, 2501); i++) {
    const cols = lines[i].split(',');
    if (cols.length !== headers.length) continue;
    const obj: DatasetRow = {};
    for (let j = 0; j < headers.length; j++) {
      const raw = cols[j].trim();
      const num = Number(raw);
      obj[headers[j]] = raw !== '' && !Number.isNaN(num) ? num : raw;
    }
    rows.push(obj);
  }
  const binaryTargetCandidates = headers.filter((h) => {
    const uniq = new Set(rows.map((r) => String(r[h])).filter((v) => v !== ''));
    return uniq.size === 2 && !h.toLowerCase().includes('gender') && !h.toLowerCase().includes('sex');
  });
  return { rows, binaryTargetCandidates, headers };
}

function StatusLabel({ status }: { status: 'PASS' | 'WARN' | 'FAIL' }) {
  if (status === 'PASS') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700">
        <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
        PASS
      </span>
    );
  }
  if (status === 'WARN') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-700">
        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
        WARN
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-red-700">
      <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
      FAIL
    </span>
  );
}

export default function App() {
  const [view, setView] = useState<WorkspaceView>('overview');
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedKey, setSelectedKey] = useState<string>('loan');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<
    'Logistic Regression' | 'Decision Tree' | 'Random Forest'
  >('Logistic Regression');
  const [includeSensitive, setIncludeSensitive] = useState<boolean>(true);
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [totalRowsOnDisk, setTotalRowsOnDisk] = useState<number>(0);
  const [customSpec, setCustomSpec] = useState<DatasetInfo | null>(null);
  const [customTargetOptions, setCustomTargetOptions] = useState<string[]>([]);
  const [loadingDataset, setLoadingDataset] = useState<boolean>(true);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  const [narrative, setNarrative] = useState<{
    text: string;
    source: string;
  } | null>(null);
  const [narrativeLoading, setNarrativeLoading] = useState<boolean>(false);
  const [narrativeError, setNarrativeError] = useState<string | null>(null);

  const [projectFiles, setProjectFiles] = useState<ProjectFileEntry[]>([]);
  const [selectedFilePath, setSelectedFilePath] = useState<string>('app.py');
  const [selectedFileContent, setSelectedFileContent] = useState<string>('');
  const [fileLoading, setFileLoading] = useState<boolean>(false);

  useEffect(() => {
    fetch('/api/datasets')
      .then((r) => r.json())
      .then((d) => setDatasets(d.datasets || []))
      .catch(() => setDatasetError('Could not load dataset registry.'));

    fetch('/api/project-files')
      .then((r) => r.json())
      .then((d) => setProjectFiles(d.files || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedKey === 'custom') return;
    setLoadingDataset(true);
    setDatasetError(null);
    setNarrative(null);
    setNarrativeError(null);
    fetch(`/api/datasets/${selectedKey}`)
      .then((r) => {
        if (!r.ok) throw new Error('Dataset load failed');
        return r.json();
      })
      .then((data) => {
        setRows(data.rows || []);
        setTotalRowsOnDisk(data.totalRows || 0);
      })
      .catch((err) => {
        setDatasetError(err instanceof Error ? err.message : 'Dataset error');
      })
      .finally(() => setLoadingDataset(false));
  }, [selectedKey]);

  useEffect(() => {
    if (view !== 'codebase' || !selectedFilePath) return;
    setFileLoading(true);
    fetch(`/api/project-file?path=${encodeURIComponent(selectedFilePath)}`)
      .then((r) => r.json())
      .then((d) => setSelectedFileContent(d.content || d.error || ''))
      .catch(() => setSelectedFileContent('Unable to read file.'))
      .finally(() => setFileLoading(false));
  }, [view, selectedFilePath]);

  const activeSpec =
    selectedKey === 'custom'
      ? customSpec
      : datasets.find((d) => d.key === selectedKey) || null;

  const audit: FullAuditOutput | null = useMemo(() => {
    if (!activeSpec || rows.length < 40) return null;
    try {
      return runClientSideAudit(rows, activeSpec.targetColumn, selectedAlgorithm, includeSensitive);
    } catch {
      return null;
    }
  }, [rows, activeSpec, selectedAlgorithm, includeSensitive]);

  const handleCsvUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setDatasetError(null);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseClientCsv(String(reader.result || ''));
        if (parsed.rows.length < 40) {
          setDatasetError('Uploaded CSV must contain at least 40 valid rows.');
          return;
        }
        const defaultTarget =
          parsed.binaryTargetCandidates[0] || parsed.headers[parsed.headers.length - 1];
        setCustomTargetOptions(
          parsed.binaryTargetCandidates.length > 0 ? parsed.binaryTargetCandidates : parsed.headers,
        );
        setCustomSpec({
          key: 'custom',
          name: file.name.replace(/\.csv$/i, ''),
          sourceType: 'uploaded',
          domain: 'Custom Uploaded CSV Dataset',
          targetColumn: defaultTarget,
          filename: file.name,
          totalRows: parsed.rows.length,
        });
        setRows(parsed.rows);
        setTotalRowsOnDisk(parsed.rows.length);
        setSelectedKey('custom');
        setNarrative(null);
      } catch (err) {
        setDatasetError(err instanceof Error ? err.message : 'Failed to parse CSV.');
      }
    };
    reader.readAsText(file);
  };

  const handleGenerateNarrative = async (
    taskType: 'audit_summary' | 'demographic_disparity' | 'metric_comparison' | 'mitigation_advice',
  ) => {
    if (!audit || !activeSpec) return;
    setNarrativeLoading(true);
    setNarrativeError(null);
    try {
      const payload = {
        dataset: activeSpec.name,
        sourceType: activeSpec.sourceType,
        model: audit.modelMetrics,
        biasRiskScore: audit.biasRiskScore,
        riskLevel: audit.riskLevel,
        attributes: audit.attributeAudits.map((a) => ({
          attribute: a.attribute,
          status: a.status,
          dpDiff: Number(a.dpDiff.toFixed(4)),
          dpRatio: Number(a.dpRatio.toFixed(4)),
          eoDiff: Number(a.eoDiff.toFixed(4)),
          eqOddsDiff: Number(a.eqOddsDiff.toFixed(4)),
        })),
      };
      const resp = await fetch('/api/audit-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metricsPayload: payload, taskType }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setNarrativeError(data.error || 'Gemini API request failed.');
      } else {
        setNarrative({ text: data.text, source: `Gemini (${data.model})` });
      }
    } catch (err) {
      setNarrativeError(err instanceof Error ? err.message : 'Network error calling Gemini endpoint.');
    } finally {
      setNarrativeLoading(false);
    }
  };

  const handleDownloadJsonSummary = () => {
    if (!audit || !activeSpec) return;
    const blob = new Blob([JSON.stringify({ dataset: activeSpec, audit }, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fairlens_${activeSpec.key}_audit.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadMarkdownReport = () => {
    if (!audit || !activeSpec) return;
    const md = buildMarkdownReport(activeSpec, audit, narrative?.text || null);
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fairlens_${activeSpec.key}_report.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Bar Contract: Brand wordmark — 4 nav links — 1 primary action */}
      <header className="bg-slate-950 text-white border-b border-slate-800 px-6 py-3.5 flex items-center justify-between gap-8">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setView('overview');
          }}
          className="text-lg font-bold tracking-tight whitespace-nowrap shrink-0"
        >
          FairLens AI
        </a>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-300">
          <button
            type="button"
            onClick={() => setView('overview')}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-white ${
              view === 'overview' ? 'text-white underline underline-offset-4' : ''
            }`}
          >
            Audit Overview
          </button>
          <button
            type="button"
            onClick={() => setView('mitigation')}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-white ${
              view === 'mitigation' ? 'text-white underline underline-offset-4' : ''
            }`}
          >
            Mitigation
          </button>
          <button
            type="button"
            onClick={() => setView('diagnostics')}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-white ${
              view === 'diagnostics' ? 'text-white underline underline-offset-4' : ''
            }`}
          >
            Diagnostics
          </button>
          <button
            type="button"
            onClick={() => setView('codebase')}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-white ${
              view === 'codebase' ? 'text-white underline underline-offset-4' : ''
            }`}
          >
            Python Codebase
          </button>
        </nav>

        <div className="shrink-0">
          <button
            type="button"
            onClick={handleDownloadJsonSummary}
            disabled={!audit}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-md transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            Export Audit JSON
          </button>
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Workspace Sidebar */}
        <aside className="w-full lg:w-72 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 p-5 shrink-0 space-y-6">
          <div>
            <h2 className="text-xs font-semibold text-slate-500 mb-2.5 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-blue-700" />
              Dataset Selection
            </h2>
            <div className="space-y-2">
              {datasets.map((ds) => {
                const active = ds.key === selectedKey;
                return (
                  <button
                    key={ds.key}
                    type="button"
                    onClick={() => setSelectedKey(ds.key)}
                    className={`w-full text-left p-3 rounded-md border transition-colors ${
                      active
                        ? 'bg-blue-50 border-blue-700 text-slate-900'
                        : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300'
                    }`}
                  >
                    <div className="text-sm font-semibold">{ds.name}</div>
                    <div className="text-xs text-slate-500 mt-0.5 tabular-nums">
                      {ds.sourceType === 'synthetic' ? 'Synthetic' : 'Historical'} ·{' '}
                      {ds.totalRows.toLocaleString()} rows
                    </div>
                  </button>
                );
              })}

              {customSpec && (
                <button
                  type="button"
                  onClick={() => setSelectedKey('custom')}
                  className={`w-full text-left p-3 rounded-md border transition-colors ${
                    selectedKey === 'custom'
                      ? 'bg-blue-50 border-blue-700 text-slate-900'
                      : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300'
                  }`}
                >
                  <div className="text-sm font-semibold truncate">{customSpec.name}</div>
                  <div className="text-xs text-slate-500 mt-0.5 tabular-nums">
                    Uploaded · {customSpec.totalRows.toLocaleString()} rows
                  </div>
                </button>
              )}
            </div>

            <label className="mt-3 w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md cursor-pointer transition-colors">
              <Upload className="w-3.5 h-3.5 text-blue-700" />
              Upload Custom CSV
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleCsvUpload}
                className="sr-only"
              />
            </label>

            {selectedKey === 'custom' && customSpec && (
              <div className="mt-2.5">
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Binary Target Column
                </label>
                <select
                  value={customSpec.targetColumn}
                  onChange={(e) =>
                    setCustomSpec((prev) =>
                      prev ? { ...prev, targetColumn: e.target.value } : prev,
                    )
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-300 rounded-md text-slate-900"
                >
                  {customTargetOptions.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div>
            <h2 className="text-xs font-semibold text-slate-500 mb-2.5 flex items-center gap-1.5">
              <SlidersHorizontal className="w-3.5 h-3.5 text-blue-700" />
              Model Configuration
            </h2>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Classifier Algorithm
            </label>
            <select
              value={selectedAlgorithm}
              onChange={(e) =>
                setSelectedAlgorithm(
                  e.target.value as 'Logistic Regression' | 'Decision Tree' | 'Random Forest',
                )
              }
              className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-md text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              <option value="Logistic Regression">Logistic Regression</option>
              <option value="Decision Tree">Decision Tree (CART)</option>
              <option value="Random Forest">Random Forest</option>
            </select>

            <label className="flex items-start gap-2.5 mt-3.5 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={includeSensitive}
                onChange={(e) => setIncludeSensitive(e.target.checked)}
                className="mt-0.5 rounded border-slate-300 text-blue-700 focus:ring-blue-600"
              />
              <span>Include Gender, Age, and Race in training features</span>
            </label>
          </div>

          <div className="pt-4 border-t border-slate-200 space-y-2">
            <button
              type="button"
              onClick={handleDownloadMarkdownReport}
              disabled={!audit}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-slate-800 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 rounded-md transition-colors"
            >
              <FileText className="w-3.5 h-3.5 text-blue-700" />
              Download Report (.md)
            </button>

            <button
              type="button"
              onClick={() => setView('codebase')}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors"
            >
              <Code2 className="w-3.5 h-3.5" />
              Browse FairLens_AI ({projectFiles.length} files)
            </button>
          </div>
        </aside>

        {/* Main Content Viewport */}
        <main className="flex-1 p-6 lg:p-8 max-w-6xl">
          {datasetError && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md text-sm text-red-800">
              {datasetError}
            </div>
          )}

          {loadingDataset && (
            <div className="space-y-4">
              <div className="h-24 bg-slate-200/70 rounded-lg animate-pulse" />
              <div className="h-64 bg-slate-200/70 rounded-lg animate-pulse" />
            </div>
          )}

          {!loadingDataset && audit && activeSpec && view === 'overview' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-slate-900 text-balance">
                  Bias Detection Audit: {activeSpec.name}
                </h1>
                <p className="text-sm text-slate-600 mt-1">
                  {activeSpec.domain} ·{' '}
                  {activeSpec.sourceType === 'synthetic'
                    ? 'Synthetic Benchmark'
                    : activeSpec.sourceType === 'historical'
                      ? 'Historical Benchmark'
                      : 'Custom Upload'}{' '}
                  · {totalRowsOnDisk.toLocaleString()} total records ({audit.testCount} held-out test records evaluated)
                </p>
              </div>

              {/* 5-Column KPI Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Accuracy</div>
                  <div className="text-xl font-bold text-slate-900 mt-1 tabular-nums">
                    {(audit.modelMetrics.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">F1-Score</div>
                  <div className="text-xl font-bold text-slate-900 mt-1 tabular-nums">
                    {audit.modelMetrics.f1Score.toFixed(3)}
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">ROC-AUC</div>
                  <div className="text-xl font-bold text-slate-900 mt-1 tabular-nums">
                    {audit.modelMetrics.rocAuc.toFixed(3)}
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Bias Risk Score</div>
                  <div className="text-xl font-bold text-blue-700 mt-1 tabular-nums">
                    {audit.biasRiskScore}/100
                  </div>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Risk Classification</div>
                  <div className="text-xl font-bold text-slate-900 mt-1">
                    {audit.riskLevel}
                  </div>
                </div>
              </div>

              {/* Fairness Summary Table */}
              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-900">
                    Protected Attribute Threshold Audit (Gender, Grouped Age, Race)
                  </h2>
                  <span className="text-xs text-slate-500">
                    Age groups: ≤25, 26–35, 36–45, 46–55, 56+
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-xs text-slate-600 border-b border-slate-200">
                      <tr>
                        <th className="text-left px-5 py-2.5 font-semibold">Attribute</th>
                        <th className="text-left px-4 py-2.5 font-semibold">Status</th>
                        <th className="text-right px-4 py-2.5 font-semibold">DP Diff</th>
                        <th className="text-right px-4 py-2.5 font-semibold">DP Ratio</th>
                        <th className="text-right px-4 py-2.5 font-semibold">Equal Opp Diff</th>
                        <th className="text-right px-5 py-2.5 font-semibold">Equalized Odds Diff</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {audit.attributeAudits.map((a) => (
                        <tr key={a.attribute} className="hover:bg-slate-50">
                          <td className="px-5 py-3 font-medium text-slate-900">{a.attribute}</td>
                          <td className="px-4 py-3">
                            <StatusLabel status={a.status} />
                          </td>
                          <td className="px-4 py-3 text-right font-mono tabular-nums">
                            {a.dpDiff.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono tabular-nums">
                            {a.dpRatio.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono tabular-nums">
                            {a.eoDiff.toFixed(4)}
                          </td>
                          <td className="px-5 py-3 text-right font-mono tabular-nums">
                            {a.eqOddsDiff.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Interactive Plotly Scatter & Bar Charts */}
              <AuditOverviewCharts
                attributeAudits={audit.attributeAudits}
                samplePoints={audit.samplePoints}
              />

              {/* Gemini Audit Narrator Panel */}
              <div className="bg-white border border-slate-200 rounded-lg p-5">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h2 className="text-sm font-semibold text-slate-900">
                      AI Fairness Audit Narrator
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Explains only the calculated metrics above; never fabricates numbers or causal proof.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleGenerateNarrative('audit_summary')}
                      disabled={narrativeLoading}
                      className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold bg-blue-700 hover:bg-blue-600 text-white rounded-md transition-colors disabled:opacity-50 whitespace-nowrap shrink-0"
                    >
                      <Play className="w-3.5 h-3.5" />
                      {narrativeLoading ? 'Generating...' : 'Generate Gemini Summary'}
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setNarrative({
                          text: buildOfflineSummary(activeSpec.name, audit),
                          source: 'Offline Template Engine',
                        })
                      }
                      className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md transition-colors whitespace-nowrap shrink-0"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      Use Offline Template
                    </button>
                  </div>
                </div>

                {narrativeError && (
                  <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded text-xs text-amber-900">
                    {narrativeError}
                  </div>
                )}

                {narrative && (
                  <div className="mt-4 pt-4 border-t border-slate-200 space-y-2">
                    <div className="text-xs font-semibold text-blue-700">
                      Generated via {narrative.source}
                    </div>
                    <div className="text-sm text-slate-700 whitespace-pre-line leading-relaxed">
                      {narrative.text}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {!loadingDataset && audit && activeSpec && view === 'mitigation' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  Model Comparison & Bias Mitigation Trade-Offs
                </h1>
                <p className="text-sm text-slate-600 mt-1">
                  Evaluated on the same {audit.testCount} held-out test records. Mitigation reduces
                  measured disparity on this split but does not guarantee universal fairness.
                </p>
              </div>

              <MitigationTradeoffCharts
                modelComparisons={audit.modelComparisons}
                mitigationComparison={audit.mitigationComparison}
              />

              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-200">
                  <h2 className="text-sm font-semibold text-slate-900">
                    Mitigation Strategy Metrics ({audit.modelMetrics.algorithm} Baseline)
                  </h2>
                </div>
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 text-xs text-slate-600 border-b border-slate-200">
                    <tr>
                      <th className="text-left px-5 py-3 font-semibold">Strategy</th>
                      <th className="text-right px-4 py-3 font-semibold">Accuracy</th>
                      <th className="text-right px-4 py-3 font-semibold">F1-Score</th>
                      <th className="text-right px-4 py-3 font-semibold">DP Diff</th>
                      <th className="text-right px-4 py-3 font-semibold">DP Ratio</th>
                      <th className="text-right px-5 py-3 font-semibold">Equalized Odds Diff</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {audit.mitigationComparison.map((row) => (
                      <tr key={row.strategy} className="hover:bg-slate-50">
                        <td className="px-5 py-3 font-medium text-slate-900">{row.strategy}</td>
                        <td className="px-4 py-3 text-right font-mono tabular-nums">
                          {(row.accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-right font-mono tabular-nums">
                          {row.f1Score.toFixed(3)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono tabular-nums">
                          {row.dpDiff.toFixed(4)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono tabular-nums">
                          {row.dpRatio.toFixed(4)}
                        </td>
                        <td className="px-5 py-3 text-right font-mono tabular-nums">
                          {row.eqOddsDiff.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-white border border-slate-200 rounded-lg p-5">
                <h2 className="text-sm font-semibold text-slate-900">
                  Fairness Metric Conflict Note
                </h2>
                <p className="text-sm text-slate-600 mt-1.5 leading-relaxed">
                  When positive base rates differ across demographic groups, the Chouldechova and
                  Kleinberg impossibility results show that Demographic Parity, Equalized Odds, and
                  Predictive Calibration cannot be satisfied simultaneously. The Python Streamlit
                  module <span className="font-mono">FairLens_AI/modules/bias_mitigation.py</span>{' '}
                  also executes Fairlearn <span className="font-mono">ExponentiatedGradient</span>{' '}
                  and <span className="font-mono">ThresholdOptimizer</span>.
                </p>
              </div>
            </div>
          )}

          {!loadingDataset && audit && activeSpec && view === 'diagnostics' && (
            <div className="space-y-6">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  Error Rates, Explainability, Proxy Leakage & Individual Fairness
                </h1>
                <p className="text-sm text-slate-600 mt-1">
                  Diagnoses FPR/FNR disparities, feature attributions, proxy leakage, structural drivers, and counterfactual flips.
                </p>
              </div>

              <DiagnosticsPlotlyCharts
                attributeAudits={audit.attributeAudits}
                featureImportances={audit.featureImportances}
              />

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">k-NN Individual Fairness Consistency</div>
                  <div className="text-xl font-bold text-slate-900 mt-1 tabular-nums">
                    {audit.knnConsistencyScore.toFixed(4)}
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Similarity of predicted probability among 5 nearest non-sensitive neighbors.
                  </p>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-4">
                  <div className="text-xs text-slate-500">Counterfactual Flip Rate</div>
                  <div className="text-xl font-bold text-blue-700 mt-1 tabular-nums">
                    {(audit.counterfactualFlipRate * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Share of evaluated records whose decision flipped when swapping only the primary sensitive attribute.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-slate-200">
                    <h2 className="text-sm font-semibold text-slate-900">Proxy Leakage Scorer</h2>
                  </div>
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-xs text-slate-600 border-b border-slate-200">
                      <tr>
                        <th className="text-left px-5 py-2.5 font-semibold">Feature</th>
                        <th className="text-right px-4 py-2.5 font-semibold">Association</th>
                        <th className="text-right px-4 py-2.5 font-semibold">Score / 100</th>
                        <th className="text-right px-5 py-2.5 font-semibold">Risk Tier</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {audit.proxyScores.slice(0, 8).map((p) => (
                        <tr key={p.feature}>
                          <td className="px-5 py-2.5 font-mono text-xs text-slate-800">{p.feature}</td>
                          <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                            {p.associationStrength.toFixed(3)}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                            {p.leakageScore.toFixed(1)}
                          </td>
                          <td className="px-5 py-2.5 text-right text-xs font-semibold">
                            <span
                              className={
                                p.tier === 'HIGH'
                                  ? 'text-red-700'
                                  : p.tier === 'MODERATE'
                                    ? 'text-amber-700'
                                    : 'text-slate-600'
                              }
                            >
                              {p.tier}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-slate-200">
                    <h2 className="text-sm font-semibold text-slate-900">Root Cause Attribution</h2>
                  </div>
                  <div className="divide-y divide-slate-200">
                    {audit.rootCauses.map((rc) => (
                      <div key={rc.driver} className="p-4 space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-semibold text-slate-900">{rc.driver}</span>
                          <span className="font-mono tabular-nums font-semibold text-blue-700">
                            {rc.contributionPct.toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-xs text-slate-600">{rc.evidence}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {audit.counterfactualCases.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-slate-200">
                    <h2 className="text-sm font-semibold text-slate-900">
                      Counterfactual Sensitivity Checks (Sample Test Records)
                    </h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-50 text-xs text-slate-600 border-b border-slate-200">
                        <tr>
                          <th className="text-left px-5 py-2.5 font-semibold">Record</th>
                          <th className="text-left px-4 py-2.5 font-semibold">Original Group</th>
                          <th className="text-right px-4 py-2.5 font-semibold">Orig Prob</th>
                          <th className="text-left px-4 py-2.5 font-semibold">Counterfactual Group</th>
                          <th className="text-right px-4 py-2.5 font-semibold">Alt Prob</th>
                          <th className="text-right px-5 py-2.5 font-semibold">Decision Flipped</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200">
                        {audit.counterfactualCases.slice(0, 10).map((cf) => (
                          <tr key={cf.recordIndex}>
                            <td className="px-5 py-2.5 font-mono text-xs tabular-nums">
                              #{cf.recordIndex + 1}
                            </td>
                            <td className="px-4 py-2.5 text-slate-800">{cf.originalGroup}</td>
                            <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                              {(cf.originalProb * 100).toFixed(1)}%
                            </td>
                            <td className="px-4 py-2.5 text-slate-800">{cf.alternateGroup}</td>
                            <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                              {(cf.alternateProb * 100).toFixed(1)}%
                            </td>
                            <td className="px-5 py-2.5 text-right text-xs font-semibold">
                              <span className={cf.decisionFlipped ? 'text-red-700' : 'text-slate-600'}>
                                {cf.decisionFlipped ? 'YES' : 'NO'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-200">
                  <h2 className="text-sm font-semibold text-slate-900">
                    Intersectional Breakdown (Gender × Age Group)
                  </h2>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-xs text-slate-600 border-b border-slate-200">
                      <tr>
                        <th className="text-left px-5 py-2.5 font-semibold">Subgroup</th>
                        <th className="text-right px-4 py-2.5 font-semibold">Count</th>
                        <th className="text-right px-4 py-2.5 font-semibold">Selection Rate</th>
                        <th className="text-right px-4 py-2.5 font-semibold">FPR</th>
                        <th className="text-right px-5 py-2.5 font-semibold">FNR</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {audit.intersectionalRows.slice(0, 12).map((r) => (
                        <tr key={r.group}>
                          <td className="px-5 py-2.5 font-medium text-slate-900">{r.group}</td>
                          <td className="px-4 py-2.5 text-right font-mono tabular-nums">{r.count}</td>
                          <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                            {(r.selectionRate * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                            {(r.fpr * 100).toFixed(1)}%
                          </td>
                          <td className="px-5 py-2.5 text-right font-mono tabular-nums">
                            {(r.fnr * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {view === 'codebase' && (
            <div className="space-y-5">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  FairLens_AI Python & Streamlit Project Explorer
                </h1>
                <p className="text-sm text-slate-600 mt-1">
                  Complete VS Code project structure with 16 native Streamlit pages, Fairlearn, SHAP,
                  Supabase PostgreSQL schema, ReportLab PDF reports, Google GenAI SDK, and Pytest.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                <div className="bg-white border border-slate-200 rounded-lg p-3 max-h-[560px] overflow-y-auto">
                  <div className="text-xs font-semibold text-slate-500 px-2 py-1.5">
                    FairLens_AI/ ({projectFiles.length} files)
                  </div>
                  <div className="space-y-0.5 mt-1">
                    {projectFiles.map((f) => (
                      <button
                        key={f.path}
                        type="button"
                        onClick={() => setSelectedFilePath(f.path)}
                        className={`w-full text-left px-2.5 py-1.5 rounded text-xs font-mono truncate transition-colors ${
                          selectedFilePath === f.path
                            ? 'bg-blue-700 text-white'
                            : 'text-slate-700 hover:bg-slate-100'
                        }`}
                      >
                        {f.path}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="lg:col-span-2 bg-slate-950 text-slate-100 rounded-lg border border-slate-800 overflow-hidden flex flex-col max-h-[560px]">
                  <div className="px-4 py-2.5 border-b border-slate-800 flex items-center justify-between text-xs font-mono">
                    <span>FairLens_AI/{selectedFilePath}</span>
                    <span className="text-slate-400">
                      {fileLoading ? 'Loading...' : `${selectedFileContent.split('\n').length} lines`}
                    </span>
                  </div>
                  <pre className="p-4 text-xs font-mono overflow-auto flex-1 leading-relaxed text-slate-200">
                    {selectedFileContent}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
