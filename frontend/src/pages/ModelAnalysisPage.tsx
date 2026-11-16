import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cpu,
  Info,
  Layers,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
} from 'recharts';
import { api } from '../api/client';

interface Props {
  diseaseCode: string;
  locationCode: string;
}

export function ModelAnalysisPage({ diseaseCode, locationCode }: Props) {
  const [data, setData] = useState<any | null>(null);
  const [selectedModelCode, setSelectedModelCode] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    api
      .getModelAnalysis(diseaseCode, locationCode)
      .then((res) => {
        if (mounted) {
          setData(res);
          if (res.candidate_models?.length) {
            setSelectedModelCode(res.candidate_models[0].model_code);
          }
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
        Loading candidate Bayesian models, SMC log marginal likelihoods, and ArviZ convergence diagnostics...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 bg-amber-50 border border-amber-200 rounded text-sm text-amber-900">
        {error || 'No model analysis available.'}
      </div>
    );
  }

  const models: any[] = data.candidate_models || [];
  const activeModel = models.find((m) => m.model_code === selectedModelCode) || models[0];

  const evidenceComparisonData = models.map((m) => ({
    model: m.model_code,
    bma_weight_pct: m.bma_weight_pct,
    log_ml: m.log_marginal_likelihood,
    waic: m.waic,
    loo: m.loo,
  }));

  const fittedBandData = (activeModel?.in_sample_fitted || []).map((pt: any) => ({
    date: pt.date,
    actual: pt.actual,
    mean: pt.mean,
    band_80: [pt.q10, pt.q90],
  }));

  return (
    <div className="space-y-6">
      {/* Header Summary */}
      <div className="bg-white border border-slate-200 rounded-md p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="text-xs font-mono text-slate-500 uppercase">
            BAYESIAN CANDIDATE MODEL SUITE • RUN #{data.training_run_id} • SEED={data.random_seed}
          </div>
          <h1 className="text-lg font-bold text-slate-900 mt-0.5">
            Marginal Likelihood Estimation, Convergence Gating & Posterior Model Weights
          </h1>
          <p className="text-xs text-slate-600 mt-1">
            Training Window: <span className="font-mono">{data.train_start_date}</span> to{' '}
            <span className="font-mono">{data.train_end_date}</span> ({data.train_observations_count} chronological weeks).
            Marginal likelihoods $\log P(D \mid M_i)$ estimated via PyMC Sequential Monte Carlo (`pm.sample_smc`).
          </p>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded px-3.5 py-2 text-right font-mono text-xs">
          <div className="text-slate-500">WEIGHT NORMALIZATION</div>
          <div className="text-sm font-bold text-emerald-800">
            Σ P(M_i | D) = {Number(data.bma_weight_sum).toFixed(4)}
          </div>
        </div>
      </div>

      {/* Candidate Models Comparison Table */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-3">
          Candidate Models, SMC Marginal Likelihoods, Information Criteria & ArviZ Convergence Diagnostics
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 font-mono text-slate-600 uppercase">
                <th className="py-2.5 px-3">Model Code</th>
                <th className="py-2.5 px-3">Model Specification</th>
                <th className="py-2.5 px-3">Convergence Gate</th>
                <th className="py-2.5 px-3 text-right">Max R-hat</th>
                <th className="py-2.5 px-3 text-right">Min ESS (Bulk)</th>
                <th className="py-2.5 px-3 text-right">log P(D | M_i) ± SE</th>
                <th className="py-2.5 px-3 text-right">ELPD-LOO</th>
                <th className="py-2.5 px-3 text-right">ELPD-WAIC</th>
                <th className="py-2.5 px-3 text-right">Prior P(M_i)</th>
                <th className="py-2.5 px-3 text-right">Posterior P(M_i | D)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {models.map((m) => {
                const isSelected = m.model_code === activeModel?.model_code;
                return (
                  <tr
                    key={m.model_code}
                    onClick={() => setSelectedModelCode(m.model_code)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-indigo-50/70 font-medium' : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-mono font-bold text-slate-900">{m.model_code}</td>
                    <td className="py-2.5 px-3">
                      <div className="text-slate-900 font-medium">{m.model_name}</div>
                      <div className="text-[11px] text-slate-500">{m.model_family}</div>
                    </td>
                    <td className="py-2.5 px-3">
                      {m.converged ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 font-mono text-[11px]">
                          <CheckCircle2 className="w-3 h-3" /> CONVERGED
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-red-50 text-red-800 border border-red-200 font-mono text-[11px]">
                          <XCircle className="w-3 h-3" /> EXCLUDED
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      {m.diagnostics?.max_rhat?.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      {m.diagnostics?.min_ess_bulk?.toFixed(0)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono font-semibold text-slate-900">
                      {m.log_marginal_likelihood !== null
                        ? `${m.log_marginal_likelihood.toFixed(2)} ± ${(m.log_ml_std_error || 0).toFixed(2)}`
                        : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                      {m.loo !== null ? m.loo.toFixed(1) : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                      {m.waic !== null ? m.waic.toFixed(1) : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                      {(m.prior_prob * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono font-bold text-indigo-950">
                      {m.bma_weight_pct.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Model Deep-Dive: Posterior Parameters & In-Sample Fit */}
      {activeModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-md p-5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
                Posterior Parameter Summary — {activeModel.model_code}
              </h3>
              <span className="text-xs font-mono text-indigo-900 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                BMA Weight: {activeModel.bma_weight_pct}%
              </span>
            </div>
            <p className="text-xs text-slate-600 mb-4 leading-relaxed">
              {activeModel.assumptions_text}
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 font-mono text-slate-600">
                    <th className="py-2 px-2.5">Parameter</th>
                    <th className="py-2 px-2.5 text-right">Post. Mean</th>
                    <th className="py-2 px-2.5 text-right">Post. SD</th>
                    <th className="py-2 px-2.5 text-right">Credible Interval</th>
                    <th className="py-2 px-2.5 text-right">R-hat</th>
                    <th className="py-2 px-2.5 text-right">ESS (Bulk)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 font-mono">
                  {(activeModel.parameters || []).map((p: any) => (
                    <tr key={p.parameter} className="hover:bg-slate-50">
                      <td className="py-1.5 px-2.5 font-semibold text-slate-900">{p.parameter}</td>
                      <td className="py-1.5 px-2.5 text-right">{p.mean}</td>
                      <td className="py-1.5 px-2.5 text-right text-slate-600">{p.sd}</td>
                      <td className="py-1.5 px-2.5 text-right text-slate-600">
                        [{p.ci_lower ?? '—'}, {p.ci_upper ?? '—'}]
                      </td>
                      <td className="py-1.5 px-2.5 text-right">
                        <span className={p.r_hat > 1.05 ? 'text-amber-700 font-bold' : 'text-emerald-800'}>
                          {p.r_hat}
                        </span>
                      </td>
                      <td className="py-1.5 px-2.5 text-right">{p.ess_bulk}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {activeModel.diagnostics?.chain_log_marginal_likelihoods?.length > 0 && (
              <div className="mt-4 pt-3 border-t border-slate-100 text-xs font-mono text-slate-600">
                SMC Per-Chain Log Marginal Likelihoods:{' '}
                {activeModel.diagnostics.chain_log_marginal_likelihoods
                  .map((v: number) => v.toFixed(2))
                  .join(' , ')}
              </div>
            )}
          </div>

          <div className="bg-white border border-slate-200 rounded-md p-5">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
              In-Sample Posterior Fit vs. Reported Cases ({activeModel.model_code})
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Click any model row in the table above to inspect its structural fit and 80% posterior trajectory band.
            </p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={fittedBandData} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={24} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Area
                    type="monotone"
                    dataKey="band_80"
                    name="80% In-Sample Credible Band"
                    fill="#c7d2fe"
                    stroke="none"
                    fillOpacity={0.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Actual Reported Cases"
                    stroke="#0f172a"
                    strokeWidth={1.8}
                    dot={{ r: 1.5 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="mean"
                    name={`${activeModel.model_code} Posterior Mean`}
                    stroke="#4338ca"
                    strokeWidth={2}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Scientific Explanation: Why Different Models Produce Different Predictions */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide mb-3">
          <Info className="w-4 h-4 text-indigo-700" />
          <span>Why Candidate Models Produce Distinct Epidemiological Forecasts</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-xs text-slate-700">
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <div className="font-mono font-bold text-slate-900 mb-1">M1_POISSON</div>
            <p className="leading-relaxed">
              Enforces $\text{'Var'}(Y_t) = \mu_t$. On overdispersed surveillance data, its likelihood sharply penalizes outliers, resulting in substantially lower marginal likelihood $\log P(D \mid M_1)$ and near-zero BMA weight.
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <div className="font-mono font-bold text-slate-900 mb-1">M2_NEGBIN</div>
            <p className="leading-relaxed">
              Estimates dispersion $\alpha$ so $\text{'Var'}(Y_t) = \mu_t + \mu_t^2/\alpha$. Captures broad baseline seasonal waves without relying on recent week-to-week autocorrelation.
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <div className="font-mono font-bold text-slate-900 mb-1">M3_AR_COUNT</div>
            <p className="leading-relaxed">
              Conditioned on lagged log-counts $\log(y_{'{t-1}'}+1)$ and $\log(y_{'{t-2}'}+1)$. Responds rapidly to active transmission momentum and unseasonal surges via recursive trajectory rollout.
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <div className="font-mono font-bold text-slate-900 mb-1">M4_SEASONAL_FOURIER</div>
            <p className="leading-relaxed">
              Uses $K=3$ Fourier harmonics ($\omega, 2\omega, 3\omega$) with frequency-shrinking priors to represent asymmetric winter peaks and multi-modal seasonal profiles.
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <div className="font-mono font-bold text-slate-900 mb-1">M5_RENEWAL_GROWTH</div>
            <p className="leading-relaxed">
              Models $\mathbb{'E'}[Y_t] = I_{'{t-1}'} \cdot \mathcal{'R'}_t$ with serial-interval infectious load $I_{'{t-1}'}$ and density-dependent damping $r_{'{\\text{damp}}'} \log(I_{'{t-1}'})$, capturing self-limiting epidemic peaks.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

