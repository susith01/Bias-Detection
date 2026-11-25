import { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import type {
  AttributeAuditResult,
  TestPredictionPoint,
} from '../fairnessEngine';
import { PlotlyChart } from './PlotlyChart';

interface AuditOverviewChartsProps {
  attributeAudits: AttributeAuditResult[];
  samplePoints: TestPredictionPoint[];
}

const ATTR_COLORS: Record<string, string> = {
  Gender: '#1D4ED8',
  'Age Group': '#0F172A',
  Race: '#0284C7',
};

const GROUP_PALETTE = [
  '#1D4ED8',
  '#0F172A',
  '#0284C7',
  '#D97706',
  '#16A34A',
  '#DC2626',
];

const CONF_BINS = [
  { label: '50–60%', min: 0.5, max: 0.6 },
  { label: '60–70%', min: 0.6, max: 0.7 },
  { label: '70–80%', min: 0.7, max: 0.8 },
  { label: '80–90%', min: 0.8, max: 0.9 },
  { label: '90–100%', min: 0.9, max: 1.01 },
];

export function AuditOverviewCharts({
  attributeAudits,
  samplePoints,
}: AuditOverviewChartsProps) {
  const availableAttributes = useMemo(
    () => attributeAudits.map((a) => a.attribute),
    [attributeAudits],
  );
  const [selectedAttr, setSelectedAttr] = useState<string>(
    availableAttributes[0] || 'Gender',
  );
  const [scatterMode, setScatterMode] = useState<'subgroups' | 'confidence_bins'>(
    'subgroups',
  );

  const effectiveAttr = availableAttributes.includes(selectedAttr)
    ? selectedAttr
    : availableAttributes[0] || 'Gender';

  const activeAudit = attributeAudits.find((a) => a.attribute === effectiveAttr);

  // 1. Plotly Scatter Traces
  const scatterTraces: Data[] = useMemo(() => {
    if (scatterMode === 'subgroups') {
      return attributeAudits.map((audit) => ({
        type: 'scatter',
        mode: 'text+markers',
        name: audit.attribute,
        x: audit.groups.map((g) => Number((g.avgModelConfidence * 100).toFixed(2))),
        y: audit.groups.map((g) => Number((g.selectionRate * 100).toFixed(2))),
        text: audit.groups.map((g) => g.group),
        textposition: 'top center',
        textfont: { size: 11, color: '#0F172A' },
        marker: {
          size: audit.groups.map((g) => Math.max(12, Math.min(34, Math.sqrt(g.count) * 1.45))),
          color: ATTR_COLORS[audit.attribute] || '#1D4ED8',
          opacity: 0.88,
          line: { width: 1.5, color: '#FFFFFF' },
        },
        customdata: audit.groups.map((g) => [
          audit.attribute,
          g.count,
          (g.ciLow * 100).toFixed(1),
          (g.ciHigh * 100).toFixed(1),
          (g.avgPredictedProb * 100).toFixed(1),
          audit.status,
        ]),
        hovertemplate:
          '<b>%{text}</b> (%{customdata[0]})<br>' +
          'Selection Rate: %{y:.1f}% (95% CI: %{customdata[2]}%–%{customdata[3]}%)<br>' +
          'Mean Confidence: %{x:.1f}%<br>' +
          'Mean P(Positive): %{customdata[4]}%<br>' +
          'Sample Size: n=%{customdata[1]} · Status: %{customdata[5]}<extra></extra>',
      }));
    }

    // Confidence-binned scatter curves for the selected demographic attribute
    const groupKey =
      effectiveAttr === 'Gender'
        ? 'gender'
        : effectiveAttr === 'Age Group'
          ? 'ageGroup'
          : 'race';

    const groups = activeAudit ? activeAudit.groups.map((g) => g.group) : [];
    return groups.map((grp, idx) => {
      const xVals: string[] = [];
      const yVals: (number | null)[] = [];
      const counts: number[] = [];

      for (const bin of CONF_BINS) {
        const subset = samplePoints.filter(
          (pt) =>
            (pt[groupKey] || 'Missing') === grp &&
            pt.modelConfidence >= bin.min &&
            pt.modelConfidence < bin.max,
        );
        xVals.push(bin.label);
        counts.push(subset.length);
        if (subset.length > 0) {
          const sel =
            (subset.filter((s) => s.predictedLabel === 1).length / subset.length) *
            100;
          yVals.push(Number(sel.toFixed(2)));
        } else {
          yVals.push(null);
        }
      }

      return {
        type: 'scatter',
        mode: 'lines+markers',
        name: grp,
        x: xVals,
        y: yVals,
        connectgaps: false,
        line: {
          width: 2.5,
          color: GROUP_PALETTE[idx % GROUP_PALETTE.length],
        },
        marker: {
          size: 8,
          color: GROUP_PALETTE[idx % GROUP_PALETTE.length],
        },
        customdata: counts,
        hovertemplate:
          `<b>${grp}</b> (${effectiveAttr})<br>` +
          'Confidence Band: %{x}<br>' +
          'Selection Rate: %{y:.1f}%<br>' +
          'Records in Band: n=%{customdata}<extra></extra>',
      };
    });
  }, [scatterMode, attributeAudits, effectiveAttr, activeAudit, samplePoints]);

  const scatterLayout: Partial<Layout> = useMemo(() => {
    if (scatterMode === 'subgroups') {
      return {
        margin: { l: 56, r: 24, t: 24, b: 52 },
        xaxis: {
          title: { text: 'Mean Model Confidence Score (%)' },
          range: [50, 100],
          gridcolor: '#E2E8F0',
          zeroline: false,
        },
        yaxis: {
          title: { text: 'Selection Rate (%)' },
          range: [0, 108],
          gridcolor: '#E2E8F0',
          zeroline: false,
        },
        legend: {
          orientation: 'h',
          yanchor: 'bottom',
          y: 1.02,
          xanchor: 'left',
          x: 0,
        },
        hovermode: 'closest',
      };
    }

    return {
      margin: { l: 56, r: 24, t: 24, b: 52 },
      xaxis: {
        title: { text: `Model Confidence Score Band (${effectiveAttr})` },
        gridcolor: '#E2E8F0',
      },
      yaxis: {
        title: { text: 'Selection Rate (%)' },
        range: [0, 105],
        gridcolor: '#E2E8F0',
      },
      legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: 1.02,
        xanchor: 'left',
        x: 0,
      },
      hovermode: 'closest',
    };
  }, [scatterMode, effectiveAttr]);

  // 2. Plotly Grouped Bar Chart Traces (Selection Rate + 95% Wilson CI vs. Mean Confidence Score)
  const barTraces: Data[] = useMemo(() => {
    if (!activeAudit) return [];
    const xLabels = activeAudit.groups.map((g) => g.group);
    const selValues = activeAudit.groups.map((g) => Number((g.selectionRate * 100).toFixed(2)));
    const errPlus = activeAudit.groups.map((g) =>
      Math.max(0, Number(((g.ciHigh - g.selectionRate) * 100).toFixed(2))),
    );
    const errMinus = activeAudit.groups.map((g) =>
      Math.max(0, Number(((g.selectionRate - g.ciLow) * 100).toFixed(2))),
    );
    const confValues = activeAudit.groups.map((g) =>
      Number((g.avgModelConfidence * 100).toFixed(2)),
    );

    return [
      {
        type: 'bar',
        name: 'Selection Rate (%)',
        x: xLabels,
        y: selValues,
        marker: { color: '#1D4ED8' },
        error_y: {
          type: 'data',
          symmetric: false,
          array: errPlus,
          arrayminus: errMinus,
          color: '#0F172A',
          thickness: 1.5,
          width: 4,
        },
        customdata: activeAudit.groups.map((g) => [
          g.count,
          (g.ciLow * 100).toFixed(1),
          (g.ciHigh * 100).toFixed(1),
        ]),
        hovertemplate:
          '<b>%{x}</b><br>' +
          'Selection Rate: %{y:.1f}% (95% CI: %{customdata[1]}%–%{customdata[2]}%)<br>' +
          'Sample Size: n=%{customdata[0]}<extra></extra>',
      },
      {
        type: 'bar',
        name: 'Mean Model Confidence (%)',
        x: xLabels,
        y: confValues,
        marker: { color: '#0F172A' },
        customdata: activeAudit.groups.map((g) => [
          g.count,
          (g.avgPredictedProb * 100).toFixed(1),
        ]),
        hovertemplate:
          '<b>%{x}</b><br>' +
          'Mean Model Confidence: %{y:.1f}%<br>' +
          'Mean P(Positive): %{customdata[1]}%<br>' +
          'Sample Size: n=%{customdata[0]}<extra></extra>',
      },
    ];
  }, [activeAudit]);

  const barLayout: Partial<Layout> = useMemo(
    () => ({
      barmode: 'group',
      margin: { l: 56, r: 24, t: 24, b: 52 },
      xaxis: {
        title: { text: effectiveAttr },
        gridcolor: '#E2E8F0',
      },
      yaxis: {
        title: { text: 'Percentage (%)' },
        range: [0, 108],
        gridcolor: '#E2E8F0',
      },
      legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: 1.02,
        xanchor: 'left',
        x: 0,
      },
    }),
    [effectiveAttr],
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Selection Rate vs. Model Confidence Score Across Demographic Groups
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Interactive Plotly visualizations linking positive selection rates with model confidence scores and 95% Wilson intervals.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-200/70 rounded-md">
          {availableAttributes.map((attr) => (
            <button
              key={attr}
              type="button"
              onClick={() => setSelectedAttr(attr)}
              className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors whitespace-nowrap shrink-0 ${
                effectiveAttr === attr
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {attr}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {/* Interactive Plotly Scatter Chart */}
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                {scatterMode === 'subgroups'
                  ? 'Subgroup Selection Rate vs. Mean Confidence'
                  : `Selection Rate by Confidence Band (${effectiveAttr})`}
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                {scatterMode === 'subgroups'
                  ? 'Bubble size reflects subgroup sample count (n); hover for Wilson CI and P(Positive).'
                  : 'Compares selection rates across demographic groups at matched model confidence intervals.'}
              </p>
            </div>
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
              <button
                type="button"
                onClick={() => setScatterMode('subgroups')}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  scatterMode === 'subgroups'
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                All Subgroups
              </button>
              <button
                type="button"
                onClick={() => setScatterMode('confidence_bins')}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  scatterMode === 'confidence_bins'
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Confidence Bands
              </button>
            </div>
          </div>

          <PlotlyChart
            data={scatterTraces}
            layout={scatterLayout}
            className="w-full h-[350px]"
          />
        </div>

        {/* Interactive Plotly Grouped Bar Chart */}
        <div className="bg-white border border-slate-200 rounded-lg p-5">
          <div className="flex items-baseline justify-between gap-3 mb-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                {effectiveAttr}: Selection Rate vs. Mean Confidence Score
              </h3>
              <p className="text-xs text-slate-500 mt-0.5 tabular-nums">
                {activeAudit
                  ? `DP Diff ${activeAudit.dpDiff.toFixed(3)} · DP Ratio ${activeAudit.dpRatio.toFixed(3)} · Status ${activeAudit.status}`
                  : ''}
              </p>
            </div>
          </div>

          <PlotlyChart
            data={barTraces}
            layout={barLayout}
            className="w-full h-[350px]"
          />

          {activeAudit && activeAudit.smallGroupWarnings.length > 0 && (
            <p className="mt-2 text-xs text-amber-700">
              {activeAudit.smallGroupWarnings[0]}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
