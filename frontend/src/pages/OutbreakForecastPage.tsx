import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Play,
  RefreshCw,
  ShieldAlert,
  Sliders,
} from 'lucide-react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';

interface Props {
  diseaseCode: string;
  locationCode: string;
}

const MODEL_COLORS: Record<string, string> = {
  M1_POISSON: '#94a3b8',
  M2_NEGBIN: '#0284c7',
  M3_AR_COUNT: '#d97706',
  M4_SEASONAL_FOURIER: '#059669',
  M5_RENEWAL_GROWTH: '#7c3aed',
};

export function OutbreakForecastPage({ diseaseCode, locationCode }: Props) {
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Interactive Forecast Configuration
  const [horizon, setHorizon] = useState<number>(8);
  const [smcDraws, setSmcDraws] = useState<number>(250);
  const [smcChains, setSmcChains] = useState<number>(2);
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [zMultiplier, setZMultiplier] = useState<number>(1.28);
  const [runBacktest, setRunBacktest] = useState<boolean>(true);
  const [visibleModels, setVisibleModels] = useState<Record<string, boolean>>({
    M1_POISSON: false,
    M2_NEGBIN: true,
    M3_AR_COUNT: true,
    M4_SEASONAL_FOURIER: true,
    M5_RENEWAL_GROWTH: true,
  });

  // Live training job status
  const [isTraining, setIsTraining] = useState(false);
  const [jobProgress, setJobProgress] = useState<number>(0);
  const [jobMessage, setJobMessage] = useState<string>('');

  const loadForecasts = (runId?: number) => {
    setLoading(true);
    setError(null);
    api
      .getForecasts(diseaseCode, locationCode, runId, 36)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadForecasts();
  }, [diseaseCode, locationCode]);

  const handleGenerateForecast = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsTraining(true);
    setJobProgress(5);
    setJobMessage('Initializing PyMC Sequential Monte Carlo (SMC) chains...');
    setError(null);

    try {
      const startRes = await api.triggerTrainingRun(
        {
          disease_code: diseaseCode,
          location_code: locationCode,
          forecast_horizon: horizon,
          smc_draws: smcDraws,
          smc_chains: smcChains,
          random_seed: randomSeed,
          run_backtest: runBacktest,
          backtest_folds: 2,
          backtest_horizon: Math.min(4, horizon),
          outbreak_z_multiplier: zMultiplier,
        },
        false
      );

      const runId = startRes.training_run_id;
      const pollInterval = setInterval(async () => {
        try {
          const st = await api.getTrainingRunStatus(runId);
          setJobProgress(st.progress_pct || 10);
          setJobMessage(st.status_message || 'Sampling posterior distributions...');
          if (st.status === 'COMPLETED') {
            clearInterval(pollInterval);
            setIsTraining(false);
            loadForecasts(runId);
          } else if (st.status === 'FAILED') {
            clearInterval(pollInterval);
            setIsTraining(false);
            setError(st.status_message || 'Training run failed.');
          }
        } catch (pollErr: any) {
          clearInterval(pollInterval);
          setIsTraining(false);
          setError(pollErr.message);
        }
      }, 1500);
    } catch (err: any) {
      setIsTraining(false);
      setError(err.message);
    }
  };

  const toggleModelVisibility = (code: string) => {
    setVisibleModels((prev) => ({ ...prev, [code]: !prev[code] }));
  };

  const riskBadgeClass = (level: string) => {
    if (level?.includes('Critical')) return 'bg-red-100 text-red-900 border-red-300';
    if (level?.includes('High')) return 'bg-orange-100 text-orange-900 border-orange-300';
    if (level?.includes('Moderate')) return 'bg-amber-100 text-amber-900 border-amber-300';
    return 'bg-emerald-100 text-emerald-900 border-emerald-300';
  };

  // Build unified chart series
  const chartData: any[] = [];
  if (data) {
    const hist = (data.historical_observations || []).slice(-28);
    for (const h of hist) {
      chartData.push({
        date: h.observation_date,
        reported_cases: h.reported_cases,
        bma_mean: null,
        bma_80: null,
        bma_95: null,
        threshold: null,
      });
    }
    for (const row of data.combined_horizon_table || []) {
      chartData.push({
        date: row.target_date,
        reported_cases: null,
        bma_mean: row.bma_mean,
        bma_median: row.bma_median,
        bma_80: [row.bma_q10, row.bma_q90],
        bma_95: [row.bma_q025, row.bma_q975],
        threshold: row.threshold_value,
        M1_POISSON: row.M1_POISSON_mean,
        M2_NEGBIN: row.M2_NEGBIN_mean,
        M3_AR_COUNT: row.M3_AR_COUNT_mean,
        M4_SEASONAL_FOURIER: row.M4_SEASONAL_FOURIER_mean,
        M5_RENEWAL_GROWTH: row.M5_RENEWAL_GROWTH_mean,
      });
    }
  }

  return (
    <div className="space-y-6">
      {/* Forecast Generation Control Panel */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide">
            <Sliders className="w-4 h-4 text-slate-700" />
            <span>Interactive Bayesian Model Averaging Forecast Generator ({diseaseCode} • {locationCode})</span>
          </div>
          {data && (
            <span className="text-xs font-mono text-slate-500">
              Active Run #{data.training_run_id} • Horizon: {data.forecast_horizon} weeks
            </span>
          )}
        </div>

        <form onSubmit={handleGenerateForecast} className="grid grid-cols-2 sm:grid-cols-6 gap-3 items-end text-xs">
          <div>
            <label className="block font-medium text-slate-600 mb-1">Forecast Horizon (Weeks)</label>
            <input
              type="number"
              min={1}
              max={24}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="w-full border border-slate-300 rounded px-2.5 py-1.5 font-mono"
            />
          </div>
          <div>
            <label className="block font-medium text-slate-600 mb-1">SMC Draws / Chain</label>
            <input
              type="number"
              min={100}
              max={1500}
              step={50}
              value={smcDraws}
              onChange={(e) => setSmcDraws(Number(e.target.value))}
              className="w-full border border-slate-300 rounded px-2.5 py-1.5 font-mono"
            />
          </div>
          <div>
            <label className="block font-medium text-slate-600 mb-1">SMC Chains</label>
            <select
              value={smcChains}
              onChange={(e) => setSmcChains(Number(e.target.value))}
              className="w-full border border-slate-300 rounded px-2.5 py-1.5 font-mono bg-white"
            >
              <option value={2}>2 Chains</option>
              <option value={3}>3 Chains</option>
              <option value={4}>4 Chains</option>
            </select>
          </div>
          <div>
            <label className="block font-medium text-slate-600 mb-1">Threshold Z-Multiplier</label>
            <input
              type="number"
              step={0.1}
              min={0.5}
              max={3.5}
              value={zMultiplier}
              onChange={(e) => setZMultiplier(Number(e.target.value))}
              className="w-full border border-slate-300 rounded px-2.5 py-1.5 font-mono"
            />
          </div>
          <div>
            <label className="block font-medium text-slate-600 mb-1">Random Seed</label>
            <input
              type="number"
              value={randomSeed}
              onChange={(e) => setRandomSeed(Number(e.target.value))}
              className="w-full border border-slate-300 rounded px-2.5 py-1.5 font-mono"
            />
          </div>
          <div>
            <button
              type="submit"
              disabled={isTraining}
              className="w-full py-1.5 px-3 bg-indigo-950 text-white rounded font-medium hover:bg-indigo-900 disabled:opacity-50 flex items-center justify-center gap-1.5"
            >
              {isTraining ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Sampling...
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" /> Train & Forecast
                </>
              )}
            </button>
          </div>
        </form>

        {/* Progress Bar when background PyMC SMC training is running */}
        {isTraining && (
          <div className="mt-4 p-3 bg-indigo-50/80 border border-indigo-200 rounded">
            <div className="flex items-center justify-between text-xs font-mono text-indigo-950 mb-1.5">
              <span>{jobMessage}</span>
              <span className="font-bold">{jobProgress.toFixed(0)}%</span>
            </div>
            <div className="w-full h-2 bg-indigo-100 rounded overflow-hidden">
              <div
                className="h-full bg-indigo-800 transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(5, jobProgress))}%` }}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded text-xs text-red-900">
            {error}
          </div>
        )}
      </div>

      {data && (
        <>
          {/* Interactive Multi-Model & BMA Forecast Chart */}
          <div className="bg-white border border-slate-200 rounded-md p-5">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
              <div>
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
                  BMA Ensemble Forecast vs. Individual Candidate Model Trajectories
                </h2>
                <p className="text-xs text-slate-500">
                  Toggle individual candidate models below to compare their structural trajectories against the BMA posterior mixture.
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {Object.keys(MODEL_COLORS).map((code) => {
                  const active = visibleModels[code];
                  const weight = ((data.bma_weights?.[code] || 0) * 100).toFixed(1);
                  return (
                    <button
                      key={code}
                      type="button"
                      onClick={() => toggleModelVisibility(code)}
                      className={`px-2 py-1 rounded text-[11px] font-mono border transition-all ${
                        active
                          ? 'bg-slate-900 text-white border-slate-900'
                          : 'bg-white text-slate-500 border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      {code} ({weight}%)
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={20} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <ReferenceLine
                    x={data.train_end_date}
                    stroke="#64748b"
                    strokeDasharray="4 4"
                    label={{ value: 'Forecast Origin T', position: 'top', fontSize: 10 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="bma_95"
                    name="BMA 95% Predictive Interval"
                    fill="#c7d2fe"
                    stroke="none"
                    fillOpacity={0.4}
                  />
                  <Area
                    type="monotone"
                    dataKey="bma_80"
                    name="BMA 80% Predictive Interval"
                    fill="#818cf8"
                    stroke="none"
                    fillOpacity={0.45}
                  />
                  <Line
                    type="monotone"
                    dataKey="reported_cases"
                    name="Reported Surveillance Cases"
                    stroke="#0f172a"
                    strokeWidth={2.2}
                    dot={{ r: 2 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="bma_mean"
                    name="BMA Ensemble Mean Forecast"
                    stroke="#1e1b4b"
                    strokeWidth={3}
                    dot={{ r: 3.5 }}
                  />
                  <Line
                    type="stepAfter"
                    dataKey="threshold"
                    name="Analytical Outbreak Threshold"
                    stroke="#dc2626"
                    strokeWidth={1.6}
                    strokeDasharray="3 3"
                    dot={false}
                  />
                  {Object.entries(MODEL_COLORS).map(([code, color]) =>
                    visibleModels[code] ? (
                      <Line
                        key={code}
                        type="monotone"
                        dataKey={code}
                        name={`${code} Mean`}
                        stroke={color}
                        strokeWidth={1.6}
                        strokeDasharray="4 2"
                        dot={false}
                      />
                    ) : null
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Outbreak Exceedance Probability Chart & Scientific Explanations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white border border-slate-200 rounded-md p-5">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
                Outbreak Exceedance Probability P(Y_future &gt; T_outbreak | D)
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Proportion of discrete BMA posterior predictive trajectories exceeding the seasonal endemic channel threshold at each horizon step.
              </p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.outbreak_risk_assessments || []} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="target_date" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                    <Tooltip formatter={(val: any) => [`${val}%`, 'Exceedance Probability']} />
                    <ReferenceLine y={25} stroke="#d97706" strokeDasharray="3 3" label={{ value: 'Moderate (25%)', fontSize: 9 }} />
                    <ReferenceLine y={55} stroke="#ea580c" strokeDasharray="3 3" label={{ value: 'High (55%)', fontSize: 9 }} />
                    <ReferenceLine y={80} stroke="#dc2626" strokeDasharray="3 3" label={{ value: 'Critical (80%)', fontSize: 9 }} />
                    <Bar dataKey="exceedance_prob_pct" name="P(Y > T_outbreak | D) %" radius={[3, 3, 0, 0]}>
                      {(data.outbreak_risk_assessments || []).map((entry: any, idx: number) => {
                        const p = entry.exceedance_prob_pct;
                        const fill =
                          p >= 80 ? '#b91c1c' : p >= 55 ? '#ea580c' : p >= 25 ? '#d97706' : '#059669';
                        return <Cell key={idx} fill={fill} />;
                      })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-md p-5">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
                Horizon-by-Horizon Outbreak Risk Classification & Diagnostic Justifications
              </h3>
              <div className="max-h-72 overflow-y-auto space-y-2.5 pr-1">
                {(data.outbreak_risk_assessments || []).map((r: any) => (
                  <div key={r.horizon_step} className="p-3 bg-slate-50 border border-slate-200 rounded text-xs">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono font-bold text-slate-900">
                        Step +{r.horizon_step}w ({r.target_date})
                      </span>
                      <span className={`px-2 py-0.5 rounded border text-[11px] font-semibold ${riskBadgeClass(r.risk_level)}`}>
                        {r.risk_level} ({r.exceedance_prob_pct}%)
                      </span>
                    </div>
                    <p className="text-slate-600 leading-relaxed">{r.explanation_text}</p>
                    <div className="text-[10px] font-mono text-slate-400 mt-1">{r.threshold_method}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

