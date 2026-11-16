import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  Layers,
  PieChart,
  Sigma,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
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

export function UncertaintyAnalysisPage({ diseaseCode, locationCode }: Props) {
  const [data, setData] = useState<any | null>(null);
  const [selectedStep, setSelectedStep] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    api
      .getUncertainty(diseaseCode, locationCode)
      .then((res) => {
        if (mounted) {
          setData(res);
          setSelectedStep(1);
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
        Decomposing predictive variance into Parameter, Aleatoric Observation, and Structural Model Uncertainty...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 bg-amber-50 border border-amber-200 rounded text-sm text-amber-900">
        {error || 'No uncertainty analysis available.'}
      </div>
    );
  }

  const horizons: any[] = data.horizon_uncertainty_decomposition || [];
  const activeStepObj = horizons.find((h) => h.horizon_step === selectedStep) || horizons[0];

  const intervalBandChartData = horizons.map((h) => ({
    target_date: h.target_date,
    step: `+${h.horizon_step}w`,
    mean: h.mean,
    median: h.median,
    band_80: h.interval_80,
    band_95: h.interval_95,
    width_80: h.width_80,
    width_95: h.width_95,
  }));

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-md p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="text-xs font-mono text-slate-500 uppercase">
            LAW OF TOTAL VARIANCE DECOMPOSITION • {data.disease.code} • {data.location.code}
          </div>
          <h1 className="text-lg font-bold text-slate-900 mt-0.5">
            Predictive Uncertainty Quantification: Parameter, Observation & Structural Model Uncertainty
          </h1>
          <p className="text-xs text-slate-600 mt-1">
            Total BMA predictive variance $\text{'Var'}(Y_{'{T+h}'} \mid D)$ is decomposed into Within-Model Parameter Variance, Within-Model Aleatoric Count Noise, and Between-Model Structural Disagreement.
          </p>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded px-3.5 py-2 text-right font-mono text-xs">
          <div className="text-slate-500">MODEL SPACE SHANNON ENTROPY</div>
          <div className="text-sm font-bold text-slate-900">
            H(M | D) = {data.model_space_entropy} nats ({Math.round(data.normalized_model_entropy * 100)}% max)
          </div>
        </div>
      </div>

      {/* Row 1: 3-Way Variance Share Stacked Bar Chart & 80%/95% Predictive Interval Fan Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            1. Three-Way Predictive Variance Decomposition Across Forecast Horizons (%)
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Relative contribution of Parameter Uncertainty, Observation Count Noise, and Between-Model Disagreement at each step $T+h$.
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={horizons} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="target_date" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar
                  dataKey="parameter_share_pct"
                  name="Parameter Uncertainty Share (%)"
                  stackId="var"
                  fill="#1e3a8a"
                />
                <Bar
                  dataKey="observation_share_pct"
                  name="Aleatoric Observation Noise Share (%)"
                  stackId="var"
                  fill="#0284c7"
                />
                <Bar
                  dataKey="model_share_pct"
                  name="Between-Model Structural Share (%)"
                  stackId="var"
                  fill="#d97706"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            2. 80% & 95% BMA Predictive Interval Expansion Fan Chart
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Visualizes how predictive uncertainty widens as forecast horizon $h$ increases.
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={intervalBandChartData} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="target_date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area
                  type="monotone"
                  dataKey="band_95"
                  name="95% Predictive Interval [Q2.5, Q97.5]"
                  fill="#c7d2fe"
                  stroke="#818cf8"
                  fillOpacity={0.45}
                />
                <Area
                  type="monotone"
                  dataKey="band_80"
                  name="80% Predictive Interval [Q10, Q90]"
                  fill="#6366f1"
                  stroke="#4f46e5"
                  fillOpacity={0.45}
                />
                <Line
                  type="monotone"
                  dataKey="mean"
                  name="BMA Predictive Mean"
                  stroke="#1e1b4b"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="median"
                  name="BMA Predictive Median"
                  stroke="#047857"
                  strokeWidth={2}
                  strokeDasharray="4 2"
                  dot={{ r: 2.5 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 2: Posterior Predictive Density Histogram for Selected Step & Prior vs Posterior Model Probabilities */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide">
              3. Full Discrete Posterior Predictive Distribution P(Y_T+h | D)
            </h3>
            <select
              value={selectedStep}
              onChange={(e) => setSelectedStep(Number(e.target.value))}
              className="border border-slate-300 rounded px-2 py-1 text-xs font-mono bg-white"
            >
              {horizons.map((h) => (
                <option key={h.horizon_step} value={h.horizon_step}>
                  Horizon +{h.horizon_step}w ({h.target_date})
                </option>
              ))}
            </select>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Empirical probability mass over 2,000 discrete count draws from the BMA posterior mixture at +{activeStepObj?.horizon_step}w.
          </p>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activeStepObj?.predictive_distribution || []} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="bin_center" tick={{ fontSize: 10 }} label={{ value: 'Predicted Weekly Case Count', position: 'insideBottom', offset: -3, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Bar dataKey="probability" name="Posterior Predictive Probability Mass" fill="#312e81" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            4. Prior P(M_i) vs. Posterior P(M_i | D) Model Probability Distribution
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Illustrates how uniform prior model probabilities ($1/K$) concentrate onto high-evidence models after conditioning on surveillance data $D$.
          </p>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.posterior_model_distribution || []} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="model_code" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="prior_prob" name="Prior Model Probability P(M_i)" fill="#94a3b8" />
                <Bar dataKey="posterior_prob" name="Posterior Model Probability P(M_i | D)" fill="#1e3a8a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Detailed Numerical Table of Variance Decomposition */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-3">
          Numerical Variance Decomposition & Credible Interval Width Table
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 font-mono text-slate-600 uppercase">
                <th className="py-2 px-3">Step</th>
                <th className="py-2 px-3">Target Date</th>
                <th className="py-2 px-3 text-right">80% Interval (Width)</th>
                <th className="py-2 px-3 text-right">95% Interval (Width)</th>
                <th className="py-2 px-3 text-right">Parameter Var (%)</th>
                <th className="py-2 px-3 text-right">Observation Var (%)</th>
                <th className="py-2 px-3 text-right">Model Structural Var (%)</th>
                <th className="py-2 px-3 text-right">Total Pred. Variance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono">
              {horizons.map((h) => (
                <tr key={h.horizon_step} className="hover:bg-slate-50">
                  <td className="py-2 px-3 font-semibold">+{h.horizon_step}w</td>
                  <td className="py-2 px-3">{h.target_date}</td>
                  <td className="py-2 px-3 text-right">
                    [{Math.round(h.interval_80[0])}, {Math.round(h.interval_80[1])}] ({Math.round(h.width_80)})
                  </td>
                  <td className="py-2 px-3 text-right">
                    [{Math.round(h.interval_95[0])}, {Math.round(h.interval_95[1])}] ({Math.round(h.width_95)})
                  </td>
                  <td className="py-2 px-3 text-right text-blue-900">
                    {Number(h.parameter_variance).toLocaleString()} ({h.parameter_share_pct}%)
                  </td>
                  <td className="py-2 px-3 text-right text-sky-800">
                    {Number(h.observation_variance).toLocaleString()} ({h.observation_share_pct}%)
                  </td>
                  <td className="py-2 px-3 text-right text-amber-800 font-semibold">
                    {Number(h.structural_model_variance).toLocaleString()} ({h.model_share_pct}%)
                  </td>
                  <td className="py-2 px-3 text-right font-bold text-slate-900">
                    {Number(h.total_variance).toLocaleString()}
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

