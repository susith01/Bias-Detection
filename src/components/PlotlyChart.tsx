import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { Config, Data, Layout } from 'plotly.js';

interface PlotlyChartProps {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
}

export function PlotlyChart({ data, layout, config, className }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const mergedLayout: Partial<Layout> = {
      autosize: true,
      paper_bgcolor: '#FFFFFF',
      plot_bgcolor: '#F8FAFC',
      font: {
        family: 'Plus Jakarta Sans, sans-serif',
        color: '#0F172A',
        size: 12,
      },
      ...layout,
    };

    const mergedConfig: Partial<Config> = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      ...config,
    };

    Plotly.react(el, data, mergedLayout, mergedConfig);

    const observer = new ResizeObserver(() => {
      if (el) {
        Plotly.Plots.resize(el);
      }
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
    };
  }, [data, layout, config]);

  useEffect(() => {
    const el = containerRef.current;
    return () => {
      if (el) {
        Plotly.purge(el);
      }
    };
  }, []);

  return <div ref={containerRef} className={className || 'w-full h-[360px]'} />;
}
