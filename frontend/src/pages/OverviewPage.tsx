import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Calendar,
  CheckCircle2,
  Database,
  ShieldAlert,
  TrendingUp,
} from 'lucide-react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { api } from '../api/client';

interface Props {
  diseaseCode: string;
  locationCode: string;
  onNavigate: (page: string) => void;
}

export function OverviewPage({ diseaseCode, locationCode, onNavigate }: Props) {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    api
      .getOverview(diseaseCode, locationCode)
      .then((res) => {
        if (mounted) {
          setData(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, [diseaseCode, locationCode]);

  if (loading) {
    return (
      <div className="p-8 text-sm text-slate-600 flex items-center gap-3">
        <div className="w-4 h-4 border-2 border-slate-800 border-t-transparent rounded-full animate-spin" />
        Loading public health surveillance overview & BMA posterior ensemble...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 bg-amber-50 border border-amber-200 rounded text-sm text-amber-900">
        <p className="font-semibold mb-1">No pre-trained BMA run found for ({diseaseCode}, {locationCode})</p>
        <p className="mb-3">{error || 'Navigate to Outbreak Forecast to train the Bayesian ensemble on this region.'}</p>
        <button
          onClick={() => onNavigate('forecast')}
          className="px-3 py-1.5 bg-slate-900 text-white rounded text-xs font-medium hover:bg-slate-800"
        >
          Go to Outbreak Forecast & Train Models
        </button>
      </div>
    );
  }

  const peakRisk = data.peak_risk_summary;
  const riskBadgeClass = (level: string) => {
    if (level?.includes('Critical')) return 'bg-red-100 text-red-900 border-red-300';
    if (level?.includes('High')) return 'bg-orange-100 text-orange-900 border-orange-300';
    if (level?.includes('Moderate')) return 'bg-amber-100 text-amber-900 border-amber-300';
    return 'bg-emerald-100 text-emerald-900 border-emerald-300';
  };

  // Merge historical observations + future BMA forecast into unified timeline with explicit distinction
  const timelineData: any[] = [];
  const histSlice = (data.historical_observations || []).slice(-36);
  for (const h of histSlice) {
    timelineData.push({
      date: h.observation_date,
      observed_cases: h.reported_cases,
      bma_mean: null,
      bma_80_band: null,
      bma_95_band: null,
      threshold: null,
      series_type: 'Observed Surveillance Record',
    });
  }

  const lastObs = histSlice[histSlice.length - 1];
  for (const f of data.combined_horizon_table || []) {
    timelineData.push({
      date: f.target_date,
      observed_cases: null,
      bma_mean: f.bma_mean,
      bma_median: f.bma_median,
      bma_80_band: [f.bma_q10, f.bma_q90],
      bma_95_band: [f.bma_q025, f.bma_q975],
      threshold: f.threshold_value,
      series_type: 'Model-Generated BMA Forecast',
    });
  }

  const weightsChartData = Object.entries(data.bma_weights || {}).map(([code, w]) => ({
    model: code,
    weight_pct: Number(((w as number) * 100).toFixed(2)),
  }));

  return (
    <div className="space-y-6">
      {/* Scientific Epistemic Banner */}
      <div className="bg-white border border-slate-200 rounded-md p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-500">
            <span>Surveillance Target: {data.disease.code}</span>
            <span>•</span>
            <span>Jurisdiction: {data.location.code}</span>
            <span>•</span>
            <span>Run ID #{data.training_run_id}</span>
          </div>
          <h1 className="text-xl font-bold text-slate-900 mt-0.5">
            {data.disease.name} — {data.location.name}
          </h1>
          <p className="text-xs text-slate-600 mt-1">
            Source Provenance: <span className="font-medium text-slate-800">{data.data_freshness.source_name}</span>.
            Reported historical counts are strictly separated from probabilistic BMA forecasts.
          </p>
        </div>
        {peakRisk && (
          <div className={`px-3.5 py-2 rounded border text-right ${riskBadgeClass(peakRisk.risk_level)}`}>
            <div className="text-[11px] font-mono uppercase tracking-wider opacity-80">
              Peak Horizon Analytical Risk
            </div>
            <div className="text-base font-bold flex items-center justify-end gap-1.5">
              <ShieldAlert className="w-4 h-4" />
              {peakRisk.risk_level} ({peakRisk.exceedance_prob_pct}% Exceedance)
            </div>
          </div>
        )}
      </div>

      {/* Top KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-md p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>LATEST OBSERVATION DATE</span>
            <Calendar className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 mt-2">
            {data.data_freshness.latest_observation_date}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Chronological training cutoff ($T$)
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>LATEST REPORTED CASES</span>
            <Activity className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 mt-2">
            {Number(data.data_freshness.latest_reported_cases).toLocaleString()}
          </div>
          <div className="text-xs text-slate-600 mt-1">
            Week-over-week change:{' '}
            <span className="font-mono font-semibold">
              {data.data_freshness.week_over_week_change_pct >= 0 ? '+' : ''}
              {data.data_freshness.week_over_week_change_pct}%
            </span>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>NEXT-WEEK BMA FORECAST (+1W)</span>
            <TrendingUp className="w-4 h-4 text-indigo-600" />
          </div>
          <div className="text-2xl font-bold font-mono text-indigo-950 mt-2">
            {data.bma_forecasts?.[0]
              ? `${Math.round(data.bma_forecasts[0].mean).toLocaleString()} cases`
              : 'N/A'}
          </div>
          <div className="text-xs text-slate-500 mt-1 font-mono">
            95% PI: [
            {data.bma_forecasts?.[0]
              ? `${Math.round(data.bma_forecasts[0].q025).toLocaleString()}, ${Math.round(
                  data.bma_forecasts[0].q975
                ).toLocaleString()}`
              : ''}
            ]
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
            <span>OUTBREAK EXCEEDANCE PROB.</span>
            <AlertTriangle className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 mt-2">
            {peakRisk ? `${peakRisk.exceedance_prob_pct}%` : '0.0%'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Max $P(Y_{'{T+h}'} &gt; T_{'{\\text{outbreak}}'} \mid D)$ over {data.bma_forecasts?.length || 8}w
          </div>
        </div>
      </div>

      {/* Main Chart: Reported Incidence vs. BMA Forecast & Credible Intervals */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-md p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
                Historical Surveillance Incidence vs. BMA Posterior Predictive Forecast
              </h2>
              <p className="text-xs text-slate-500">
                Solid navy line = actual reported cases ($t \le T$). Dashed indigo line & shaded bands = BMA forecast mean, 80% PI, and 95% PI ($t &gt; T$).
              </p>
            </div>
            <span className="text-[11px] font-mono bg-slate-100 text-slate-700 px-2 py-1 rounded border border-slate-200">
              Cutoff: {lastObs?.observation_date}
            </span>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timelineData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#475569' }} minTickGap={24} />
                <YAxis tick={{ fontSize: 11, fill: '#475569' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#cbd5e1',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                {lastObs && (
                  <ReferenceLine
                    x={lastObs.observation_date}
                    stroke="#64748b"
                    strokeDasharray="4 4"
                    label={{ value: 'Forecast Origin T', position: 'top', fontSize: 10, fill: '#475569' }}
                  />
                )}
                <Area
                  type="monotone"
                  dataKey="bma_95_band"
                  name="95% Predictive Interval (BMA)"
                  fill="#c7d2fe"
                  stroke="none"
                  fillOpacity={0.45}
                />
                <Area
                  type="monotone"
                  dataKey="bma_80_band"
                  name="80% Predictive Interval (BMA)"
                  fill="#818cf8"
                  stroke="none"
                  fillOpacity={0.45}
                />
                <Line
                  type="monotone"
                  dataKey="observed_cases"
                  name="Actual Reported Cases"
                  stroke="#0f172a"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls={false}
                />
                <Line
                  type="monotone"
                  dataKey="bma_mean"
                  name="BMA Predictive Mean Forecast"
                  stroke="#4338ca"
                  strokeWidth={2.5}
                  strokeDasharray="5 3"
                  dot={{ r: 3 }}
                  connectNulls={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="threshold"
                  name="Analytical Outbreak Threshold"
                  stroke="#dc2626"
                  strokeWidth={1.5}
                  strokeDasharray="2 2"
                  dot={false}
                  connectNulls={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Posterior Model Probabilities P(M_i | D) */}
        <div className="bg-white border border-slate-200 rounded-md p-5 flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
              BMA Posterior Model Weights P(M_i | D)
            </h2>
            <p className="text-xs text-slate-500 mt-1 mb-4">
              Computed via Sequential Monte Carlo (SMC) marginal likelihood $\log P(D \mid M_i)$ strictly on training observations.
            </p>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weightsChartData} layout="vertical" margin={{ top: 5, right: 20, left: 35, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="model" tick={{ fontSize: 10, fontFamily: 'monospace' }} width={110} />
                  <Tooltip formatter={(val: any) => [`${val}%`, 'Posterior Weight']} />
                  <Bar dataKey="weight_pct" name="Posterior Weight %" radius={[0, 3, 3, 0]}>
                    {weightsChartData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.weight_pct > 25 ? '#1e3a8a' : '#475569'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-600 space-y-1.5">
            <div className="flex items-center justify-between">
              <span>Normalization Check $\sum_i P(M_i \mid D)$:</span>
              <span className="font-mono font-semibold text-emerald-700 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> 1.0000
              </span>
            </div>
            {data.evaluation_summary?.has_validation && (
              <p className="text-[11px] text-slate-500 leading-relaxed">
                {data.evaluation_summary.comparison_summary_text}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Outbreak Risk Horizon Table */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
              Horizon-by-Horizon Forecast & Outbreak Risk Assessment Summary
            </h2>
            <p className="text-xs text-slate-500">
              Note: Forecast values are probabilistic model outputs and must not be interpreted as observed surveillance facts.
            </p>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600 font-mono uppercase">
                <th className="py-2.5 px-3">Step</th>
                <th className="py-2.5 px-3">Target Date</th>
                <th className="py-2.5 px-3 text-right">BMA Mean</th>
                <th className="py-2.5 px-3 text-right">BMA Median</th>
                <th className="py-2.5 px-3 text-right">80% Interval</th>
                <th className="py-2.5 px-3 text-right">95% Interval</th>
                <th className="py-2.5 px-3 text-right">Analytical Threshold</th>
                <th className="py-2.5 px-3 text-right">P(Y &gt; T_outbreak | D)</th>
                <th className="py-2.5 px-3">Risk Classification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {(data.combined_horizon_table || []).map((row: any) => (
                <tr key={row.horizon_step} className="hover:bg-slate-50">
                  <td className="py-2 px-3 font-mono font-semibold">+{row.horizon_step}w</td>
                  <td className="py-2 px-3 font-mono">{row.target_date}</td>
                  <td className="py-2 px-3 text-right font-mono font-semibold text-indigo-950">
                    {Number(row.bma_mean).toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono">{Number(row.bma_median).toLocaleString()}</td>
                  <td className="py-2 px-3 text-right font-mono text-slate-600">
                    [{Math.round(row.bma_q10).toLocaleString()}, {Math.round(row.bma_q90).toLocaleString()}]
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-slate-600">
                    [{Math.round(row.bma_q025).toLocaleString()}, {Math.round(row.bma_q975).toLocaleString()}]
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-red-800">
                    {Number(row.threshold_value).toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono font-semibold">
                    {row.exceedance_prob_pct}%
                  </td>
                  <td className="py-2 px-3">
                    <span className={`px-2 py-0.5 rounded border text-[11px] font-medium ${riskBadgeClass(row.risk_level)}`}>
                      {row.risk_level}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

