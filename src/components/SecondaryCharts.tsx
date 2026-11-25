import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import type {
  AttributeAuditResult,
  FeatureImportanceRow,
  MitigationRow,
  ModelComparisonRow,
} from '../fairnessEngine';
import { PlotlyChart } from './PlotlyChart';

export function MitigationTradeoffCharts({
  modelComparisons,
  mitigationComparison,
}: {
  modelComparisons: ModelComparisonRow[];
  mitigationComparison: MitigationRow[];
}) {
  const modelBarData: Data[] = useMemo(
    () => [
      {
        type: 'bar',
        name: 'Accuracy (%)',
        x: modelComparisons.map((m) => m.algorithm),
        y: modelComparisons.map((m) => Number((m.accuracy * 100).toFixed(2))),
        marker: { color: '#1D4ED8' },
      },
      {
        type: 'bar',
        name: 'F1-Score (%)',
        x: modelComparisons.map((m) => m.algorithm),
        y: modelComparisons.map((m) => Number((m.f1Score * 100).toFixed(2))),
        marker: { color: '#0284C7' },
      },
      {
        type: 'bar',
        name: 'Max DP Diff (%)',
        x: modelComparisons.map((m) => m.algorithm),
        y: modelComparisons.map((m) => Number((m.maxDpDiff * 100).toFixed(2))),
        marker: { color: '#DC2626' },
      },
      {
        type: 'bar',
        name: 'Max EqOdds Diff (%)',
        x: modelComparisons.map((m) => m.algorithm),
        y: modelComparisons.map((m) => Number((m.maxEqOddsDiff * 100).toFixed(2))),
        marker: { color: '#D97706' },
      },
    ],
    [modelComparisons],
  );

  const modelBarLayout: Partial<Layout> = useMemo(
    () => ({
      barmode: 'group',
      margin: { l: 50, r: 20, t: 24, b: 44 },
      yaxis: { title: { text: 'Percentage (%)' }, range: [0, 105], gridcolor: '#E2E8F0' },
      legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0 },
    }),
    [],
  );

  const tradeoffData: Data[] = useMemo(
    () => [
      {
        type: 'scatter',
        mode: 'text+markers',
        x: mitigationComparison.map((r) => Number((r.dpDiff * 100).toFixed(2))),
        y: mitigationComparison.map((r) => Number((r.accuracy * 100).toFixed(2))),
        text: mitigationComparison.map((r) => r.strategy.replace(' (Pre-processing)', '').replace(' (Post-processing)', '')),
        textposition: 'top center',
        marker: {
          size: mitigationComparison.map((r) => Math.max(14, r.f1Score * 28)),
          color: ['#DC2626', '#1D4ED8', '#16A34A'],
          line: { width: 1.5, color: '#FFFFFF' },
        },
        customdata: mitigationComparison.map((r) => [
          r.strategy,
          (r.f1Score * 100).toFixed(1),
          r.dpRatio.toFixed(3),
          (r.eqOddsDiff * 100).toFixed(1),
        ]),
        hovertemplate:
          '<b>%{customdata[0]}</b><br>' +
          'Accuracy: %{y:.1f}%<br>' +
          'DP Diff: %{x:.1f}% · DP Ratio: %{customdata[2]}<br>' +
          'EqOdds Diff: %{customdata[3]}% · F1: %{customdata[1]}%<extra></extra>',
      },
    ],
    [mitigationComparison],
  );

  const tradeoffLayout: Partial<Layout> = useMemo(
    () => ({
      margin: { l: 56, r: 24, t: 24, b: 50 },
      xaxis: {
        title: { text: 'Demographic Parity Difference (%) — Lower is Fairer' },
        rangemode: 'tozero',
        gridcolor: '#E2E8F0',
      },
      yaxis: {
        title: { text: 'Accuracy (%) — Higher is Better' },
        gridcolor: '#E2E8F0',
      },
      showlegend: false,
    }),
    [],
  );

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h3 className="text-sm font-semibold text-slate-900">
          3-Model Benchmark Comparison (Identical Test Split)
        </h3>
        <p className="text-xs text-slate-500 mt-0.5 mb-3">
          Compares Logistic Regression, CART Decision Tree, and Bagged Random Forest across accuracy and peak disparities.
        </p>
        <PlotlyChart data={modelBarData} layout={modelBarLayout} className="w-full h-[330px]" />
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h3 className="text-sm font-semibold text-slate-900">
          Fairness–Accuracy Trade-Off Frontier
        </h3>
        <p className="text-xs text-slate-500 mt-0.5 mb-3">
          Top-left indicates higher accuracy with lower Demographic Parity disparity.
        </p>
        <PlotlyChart data={tradeoffData} layout={tradeoffLayout} className="w-full h-[330px]" />
      </div>
    </div>
  );
}

export function DiagnosticsPlotlyCharts({
  attributeAudits,
  featureImportances,
}: {
  attributeAudits: AttributeAuditResult[];
  featureImportances: FeatureImportanceRow[];
}) {
  const [selectedAttr, setSelectedAttr] = useState<string>(attributeAudits[0]?.attribute || 'Gender');
  const activeAttr =
    attributeAudits.find((a) => a.attribute === selectedAttr) || attributeAudits[0];

  const errorRateData: Data[] = useMemo(() => {
    if (!activeAttr) return [];
    const groups = activeAttr.groups.map((g) => g.group);
    return [
      {
        type: 'bar',
        name: 'False Positive Rate (%)',
        x: groups,
        y: activeAttr.groups.map((g) => Number((g.fpr * 100).toFixed(2))),
        marker: { color: '#0F172A' },
      },
      {
        type: 'bar',
        name: 'False Negative Rate (%)',
        x: groups,
        y: activeAttr.groups.map((g) => Number((g.fnr * 100).toFixed(2))),
        marker: { color: '#1D4ED8' },
      },
    ];
  }, [activeAttr]);

  const errorLayout: Partial<Layout> = useMemo(
    () => ({
      barmode: 'group',
      margin: { l: 50, r: 20, t: 20, b: 46 },
      yaxis: { title: { text: 'Error Rate (%)' }, range: [0, 105], gridcolor: '#E2E8F0' },
      legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0 },
    }),
    [],
  );

  const importanceData: Data[] = useMemo(() => {
    const ordered = [...featureImportances].reverse();
    return [
      {
        type: 'bar',
        orientation: 'h',
        x: ordered.map((f) => f.importance),
        y: ordered.map((f) => f.feature),
        marker: {
          color: ordered.map((f) =>
            f.direction === 'Negative' ? '#DC2626' : f.direction === 'Positive' ? '#1D4ED8' : '#0F172A',
          ),
        },
        customdata: ordered.map((f) => f.direction),
        hovertemplate: '<b>%{y}</b><br>Relative Share: %{x:.1f}%<br>Direction: %{customdata}<extra></extra>',
      },
    ];
  }, [featureImportances]);

  const importanceLayout: Partial<Layout> = useMemo(
    () => ({
      margin: { l: 140, r: 20, t: 20, b: 46 },
      xaxis: { title: { text: 'Attribution Share (%)' }, gridcolor: '#E2E8F0' },
      yaxis: { automargin: true },
      showlegend: false,
    }),
    [],
  );

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              False Positive & False Negative Rates
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Compares error burdens across protected subgroups.
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
            {attributeAudits.map((a) => (
              <button
                key={a.attribute}
                type="button"
                onClick={() => setSelectedAttr(a.attribute)}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                  activeAttr?.attribute === a.attribute
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {a.attribute}
              </button>
            ))}
          </div>
        </div>
        <PlotlyChart data={errorRateData} layout={errorLayout} className="w-full h-[320px]" />
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h3 className="text-sm font-semibold text-slate-900">
          Model Feature Attribution (Top 10)
        </h3>
        <p className="text-xs text-slate-500 mt-0.5 mb-3">
          Statistical feature weight / Gini gain share (association, not causal proof).
        </p>
        <PlotlyChart data={importanceData} layout={importanceLayout} className="w-full h-[320px]" />
      </div>
    </div>
  );
}
