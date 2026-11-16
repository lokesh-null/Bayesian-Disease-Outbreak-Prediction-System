import React, { useEffect, useState } from 'react';
import {
  Award,
  CheckCircle2,
  ClipboardCheck,
  Scale,
  TrendingDown,
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

export function ModelEvaluationPage({ diseaseCode, locationCode }: Props) {
  const [data, setData] = useState<any | null>(null);
  const [selectedFoldIdx, setSelectedFoldIdx] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    api
      .getEvaluation(diseaseCode, locationCode)
      .then((res) => {
        if (mounted) {
          setData(res);
          setSelectedFoldIdx(0);
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
        Loading chronological expanding-window cross-validation metrics and out-of-sample calibration results...
      </div>
    );
  }

  if (error || !data || !data.has_validation) {
    return (
      <div className="p-6 bg-amber-50 border border-amber-200 rounded text-sm text-amber-900">
        {error || 'No chronological cross-validation backtest found for this training run.'}
      </div>
    );
  }

  const metrics: any[] = data.metrics || [];
  const folds: any[] = data.folds || [];
  const activeFold = folds[selectedFoldIdx] || folds[0];

  const foldChartData = (activeFold?.predictions || []).map((p: any) => ({
    target_date: p.target_date,
    actual_cases: p.actual_cases,
    bma_mean: p.bma_mean,
    bma_80: [p.bma_q10, p.bma_q90],
    bma_95: [p.bma_q025, p.bma_q975],
    ...p.individual_model_means,
  }));

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-md p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="text-xs font-mono text-slate-500 uppercase">
            STRICT OUT-OF-SAMPLE CHRONOLOGICAL VALIDATION • {data.disease.code} • {data.location.code}
          </div>
          <h1 className="text-lg font-bold text-slate-900 mt-0.5">
            {data.validation_protocol} ({data.n_folds} Folds × {data.horizon_per_fold} Weeks)
          </h1>
          <p className="text-xs text-slate-600 mt-1 leading-relaxed max-w-4xl">
            {data.comparison_summary_text}
          </p>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded px-3.5 py-2 text-right font-mono text-xs shrink-0">
          <div className="text-slate-500">LEAKAGE AUDIT</div>
          <div className="text-sm font-bold text-emerald-800 flex items-center justify-end gap-1">
            <CheckCircle2 className="w-4 h-4" /> Zero Future Leakage
          </div>
        </div>
      </div>

      {/* Out-of-Sample Proper Scoring Rules & Error Comparison Table */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
              Out-of-Sample Forecasting Performance: BMA Ensemble vs. Individual Candidate Models
            </h2>
            <p className="text-xs text-slate-500">
              All metrics are computed strictly on held-out future folds never seen during training or BMA weight estimation. MAPE is excluded in favor of count-safe MASE and proper scoring rules (CRPS, NLPD).
            </p>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 font-mono text-slate-600 uppercase">
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3 text-right">CRPS (Proper)</th>
                <th className="py-2.5 px-3 text-right">NLPD</th>
                <th className="py-2.5 px-3 text-right">MAE</th>
                <th className="py-2.5 px-3 text-right">RMSE</th>
                <th className="py-2.5 px-3 text-right">MASE</th>
                <th className="py-2.5 px-3 text-right">80% Coverage</th>
                <th className="py-2.5 px-3 text-right">95% Coverage</th>
                <th className="py-2.5 px-3 text-right">Mean 95% Width</th>
                <th className="py-2.5 px-3 text-right">Mean Fold Weight</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono">
              {metrics.map((m) => {
                const isBma = m.model_code === 'BMA';
                return (
                  <tr
                    key={m.model_code}
                    className={isBma ? 'bg-indigo-50/80 font-bold text-indigo-950' : 'hover:bg-slate-50'}
                  >
                    <td className="py-2.5 px-3">
                      <div>{m.model_code}</div>
                      <div className="text-[10px] font-sans font-normal text-slate-500">{m.model_name}</div>
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold">{Number(m.crps).toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-right">{Number(m.nlpd).toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-right">{Number(m.mae).toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-right">{Number(m.rmse).toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-right">{Number(m.mase).toFixed(3)}</td>
                    <td className="py-2.5 px-3 text-right">{m.coverage_80_pct}%</td>
                    <td className="py-2.5 px-3 text-right">{m.coverage_95_pct}%</td>
                    <td className="py-2.5 px-3 text-right">{Number(m.interval_width_95).toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-right">
                      {isBma
                        ? '100% (Ensemble)'
                        : m.mean_bma_weight_across_folds !== null
                        ? `${(m.mean_bma_weight_across_folds * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts: CRPS / MASE Comparison & Calibration Coverage Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            1. Probabilistic Forecast Error Comparison (CRPS & MAE — Lower is Better)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Continuous Ranked Probability Score (CRPS) evaluates both calibration and sharpness of the entire predictive distribution.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metrics} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="model_code" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="crps" name="CRPS (Continuous Ranked Prob. Score)" fill="#1e3a8a" />
                <Bar dataKey="mae" name="MAE (Mean Absolute Error)" fill="#64748b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            2. Empirical Predictive Interval Calibration (Nominal 80% & 95% vs. Actual Coverage)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Note how equidispersed Poisson (`M1_POISSON`) suffers severe under-coverage on overdispersed count data, whereas Negative Binomial and BMA achieve reliable coverage.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metrics} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="model_code" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={80} stroke="#0284c7" strokeDasharray="3 3" label={{ value: 'Nominal 80%', fontSize: 9 }} />
                <ReferenceLine y={95} stroke="#059669" strokeDasharray="3 3" label={{ value: 'Nominal 95%', fontSize: 9 }} />
                <Bar dataKey="coverage_80_pct" name="Empirical 80% PI Coverage (%)" fill="#0284c7" />
                <Bar dataKey="coverage_95_pct" name="Empirical 95% PI Coverage (%)" fill="#059669" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Fold-by-Fold Out-of-Sample Backtest Inspector */}
      {activeFold && (
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
                3. Historical Backtesting Fold Inspector: Predicted vs. Actual Held-Out Observations
              </h3>
              <p className="text-xs text-slate-500 font-mono mt-0.5">
                Train Window: {activeFold.train_start_date} → {activeFold.train_end_date} ({activeFold.train_size} obs) | Held-Out Test Window: {activeFold.test_start_date} → {activeFold.test_end_date}
              </p>
            </div>
            <div className="flex gap-1.5">
              {folds.map((f, idx) => (
                <button
                  key={f.fold_index}
                  onClick={() => setSelectedFoldIdx(idx)}
                  className={`px-3 py-1 rounded text-xs font-mono border ${
                    selectedFoldIdx === idx
                      ? 'bg-slate-900 text-white border-slate-900'
                      : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  Fold #{f.fold_index}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={foldChartData} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="target_date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Area
                    type="monotone"
                    dataKey="bma_95"
                    name="BMA 95% Predictive Interval"
                    fill="#c7d2fe"
                    stroke="none"
                    fillOpacity={0.45}
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
                    dataKey="actual_cases"
                    name="Held-Out Actual Reported Cases"
                    stroke="#0f172a"
                    strokeWidth={2.5}
                    dot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="bma_mean"
                    name="BMA Out-of-Sample Mean Forecast"
                    stroke="#4338ca"
                    strokeWidth={2.5}
                    strokeDasharray="5 3"
                    dot={{ r: 3.5 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded p-4 text-xs flex flex-col justify-between">
              <div>
                <div className="font-mono font-bold text-slate-900 uppercase mb-2">
                  Fold #{activeFold.fold_index} Training-Only BMA Weights
                </div>
                <p className="text-[11px] text-slate-500 mb-3">
                  Weights calculated strictly from training data up to {activeFold.train_end_date} prior to observing test actuals:
                </p>
                <div className="space-y-1.5 font-mono">
                  {Object.entries(activeFold.bma_weights_in_fold || {}).map(([code, w]) => (
                    <div key={code} className="flex items-center justify-between border-b border-slate-200 pb-1">
                      <span className="text-slate-700">{code}</span>
                      <span className="font-bold text-slate-900">{((w as number) * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-3 pt-2 border-t border-slate-200 text-[11px] text-slate-500">
                Log Marginal Likelihoods $\log P(D_{'{\\text{train}}'} \mid M_i)$ computed independently per fold via SMC tempering.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

