import React from 'react';
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  GitMerge,
  Sigma,
} from 'lucide-react';

export function AboutModelPage() {
  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-md p-6">
        <div className="flex items-center gap-2 text-xs font-mono text-slate-500 uppercase mb-1">
          <BookOpen className="w-4 h-4 text-indigo-800" />
          <span>Mathematical Foundation & Methodological Reference</span>
        </div>
        <h1 className="text-xl font-bold text-slate-900">
          Disease Outbreak Forecasting via Bayesian Model Averaging (BMA)
        </h1>
        <p className="text-sm text-slate-600 mt-2 leading-relaxed">
          Standard epidemiological forecasting pipelines typically select a single model M* and proceed as if M* generated the true transmission process. Conditioning on a single model ignores <strong>structural model uncertainty</strong>, leading to overconfident predictive intervals and brittle outbreak risk assessments when epidemic dynamics shift between seasonal endemicity and rapid autoregressive surges.
        </p>
      </div>

      {/* Mathematical Equations Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-6 space-y-4">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide">
            <Sigma className="w-4 h-4 text-indigo-800" />
            <span>1. Posterior Model Probabilities & Marginal Likelihood</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Let M = {'{M_1, M_2, ..., M_K}'} denote our ensemble of candidate Bayesian count models, and let D = y_1:T denote strictly historical surveillance observations. By Bayes&apos; theorem, the posterior probability that model M_i represents the data-generating structure is:
          </p>
          <div className="p-3.5 bg-slate-900 text-slate-100 rounded font-mono text-xs overflow-x-auto">
            P(M_i | D) = [ P(D | M_i) · P(M_i) ] / Σ_j [ P(D | M_j) · P(M_j) ] ,   where Σ_i P(M_i | D) = 1
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Here P(D | M_i) is the <strong>marginal likelihood (model evidence)</strong> obtained by integrating the likelihood over the entire prior parameter space Θ_i:
          </p>
          <div className="p-3.5 bg-slate-900 text-slate-100 rounded font-mono text-xs overflow-x-auto">
            P(D | M_i) = ∫ P(D | θ_i, M_i) · P(θ_i | M_i) dθ_i
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            We estimate log P(D | M_i) via <strong>Sequential Monte Carlo (SMC) tempering</strong> (pm.sample_smc), which bridges the prior (β_0 = 0) to the posterior (β_M = 1) across adaptive inverse-temperature stages and computes unbiased importance-weight normalizing constants. To prevent numerical underflow when log P(D | M_i) ≪ -1000, weights are normalized in log-space via the Log-Sum-Exp identity:
          </p>
          <div className="p-3 bg-slate-100 border border-slate-200 text-slate-800 rounded font-mono text-xs">
            log P(M_i | D) = (ℓ_i + log π_i) - [ max_k(ℓ_k + log π_k) + log Σ_j exp((ℓ_j + log π_j) - max_k) ]
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-6 space-y-4">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide">
            <GitMerge className="w-4 h-4 text-indigo-800" />
            <span>2. BMA Posterior Predictive Mixture & Variance Decomposition</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Genuine Bayesian Model Averaging is <strong>not</strong> a simple average of point predictions. Instead, it forms the complete mixture of posterior predictive distributions across all converged candidate models:
          </p>
          <div className="p-3.5 bg-slate-900 text-slate-100 rounded font-mono text-xs overflow-x-auto">
            P(y_future | D) = Σ_{'{i=1..K}'} P(y_future | D, M_i) · P(M_i | D)
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            By the <strong>Law of Total Variance</strong>, the total BMA predictive variance at forecast horizon T+h decomposes into three interpretable epidemiological components:
          </p>
          <div className="p-3.5 bg-slate-900 text-slate-100 rounded font-mono text-xs overflow-x-auto space-y-1">
            <div>Var(Y_future | D) = Σ_i w_i · Var(Y_future | D, M_i)</div>
            <div>                  + Σ_i w_i · ( E[Y_future | D, M_i] - E[Y_future | D] )²</div>
          </div>
          <ul className="text-xs text-slate-600 space-y-1.5 list-disc pl-4">
            <li>
              <strong>Within-Model Parameter Uncertainty</strong>: Σ_i w_i Var_θ(μ_T+h(θ_i) | D, M_i), arising from finite historical sample size when estimating regression, lag, and seasonal parameters.
            </li>
            <li>
              <strong>Within-Model Aleatoric Observation Noise</strong>: Σ_i w_i E_θ[Var(Y_T+h | θ_i, M_i)], capturing irreducible Poisson/Negative-Binomial count variability (μ + μ²/α).
            </li>
            <li>
              <strong>Between-Model Structural Uncertainty</strong>: Σ_i w_i (μ_i - μ_BMA)², measuring divergence across competing transmission hypotheses.
            </li>
          </ul>
        </div>
      </div>

      {/* Explicit Five-Way Conceptual Distinction */}
      <div className="bg-white border border-slate-200 rounded-md p-6">
        <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-4">
          3. Core Statistical Distinctions in Bayesian Epidemiological Forecasting
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-xs">
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
            <div className="font-mono font-bold text-slate-900 mb-1">1. Individual Model Prediction</div>
            <p className="text-slate-600 leading-relaxed">
              P(y_future | D, M_i): The forecast distribution conditional on a single mathematical structure M_i being 100% true.
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
            <div className="font-mono font-bold text-slate-900 mb-1">2. Posterior Model Probability</div>
            <p className="text-slate-600 leading-relaxed">
              P(M_i | D): The Bayesian probability weight assigned to model M_i by its marginal likelihood P(D | M_i), which naturally penalizes overparameterization (Occam&apos;s razor).
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
            <div className="font-mono font-bold text-slate-900 mb-1">3. Model Uncertainty</div>
            <p className="text-slate-600 leading-relaxed">
              Epistemic uncertainty regarding which mathematical formulation (Poisson, NegBin, Autoregressive, Fourier, or Renewal) best reflects the true transmission process.
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
            <div className="font-mono font-bold text-slate-900 mb-1">4. Parameter Uncertainty</div>
            <p className="text-slate-600 leading-relaxed">
              Posterior uncertainty P(θ_i | D, M_i) in the coefficients (β, φ, α, r_0) within a fixed model family M_i.
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
            <div className="font-mono font-bold text-slate-900 mb-1">5. Predictive Uncertainty</div>
            <p className="text-slate-600 leading-relaxed">
              The complete uncertainty over future unobserved counts Y_T+h, integrating Parameter Uncertainty, Aleatoric Count Variance, and Structural Model Uncertainty.
            </p>
          </div>
        </div>
      </div>

      {/* Scientific Scope & Limitations: What the System Can and Cannot Predict */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-emerald-50/60 border border-emerald-200 rounded-md p-5">
          <div className="flex items-center gap-2 text-sm font-bold text-emerald-950 uppercase tracking-wide mb-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-700" />
            <span>What This System Can Reliably Quantify</span>
          </div>
          <ul className="text-xs text-emerald-900 space-y-2 list-disc pl-4 leading-relaxed">
            <li>
              Short-to-medium horizon (1–12 week) probabilistic count trajectories conditioned on historical surveillance patterns, seasonal harmonics, and recent transmission momentum.
            </li>
            <li>
              Rigorous Bayesian model comparison via SMC marginal likelihoods log P(D | M_i) and out-of-sample expanding-window proper scoring rules (CRPS, NLPD, MASE).
            </li>
            <li>
              Calibrated 80% and 95% discrete predictive intervals and exceedance probabilities P(Y_T+h &gt; T_outbreak | D) relative to analytical seasonal endemic channels.
            </li>
          </ul>
        </div>

        <div className="bg-amber-50/70 border border-amber-200 rounded-md p-5">
          <div className="flex items-center gap-2 text-sm font-bold text-amber-950 uppercase tracking-wide mb-2">
            <AlertTriangle className="w-4 h-4 text-amber-700" />
            <span>What This System Cannot Definitively Predict (Epistemic Limits)</span>
          </div>
          <ul className="text-xs text-amber-900 space-y-2 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Novel Pathogen Emergence & Antigenic Shift</strong>: Statistical surveillance models cannot anticipate zoonotic spillover or sudden viral mutations prior to their appearance in surveillance counts.
            </li>
            <li>
              <strong>Exogenous Policy or Behavioral Shocks</strong>: Abrupt public health interventions, school closures, or changes in clinical laboratory testing protocols alter reporting denominators in ways historical time series cannot foresee.
            </li>
            <li>
              <strong>Reporting Backfill & Surveillance Delays</strong>: Provisional weekly surveillance counts (such as CDC NNDSS and ILINet) are subject to administrative reporting lags; forecasts reflect conditional risk given reported observations, never deterministic certainty.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
