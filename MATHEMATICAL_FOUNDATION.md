# Mathematical Foundation: Bayesian Model Averaging for Epidemiological Forecasting

## 1. Problem Formulation

Let $y_1, y_2, \dots, y_T \in \mathbb{N}_0$ denote a chronologically ordered sequence of observed disease surveillance counts at discrete reporting intervals $t \in \{1, \dots, T\}$. Let $D = y_{1:T}$ denote the observed historical dataset up to forecast origin $T$, and let $\mathbf{y}_{\text{future}} = (Y_{T+1}, \dots, Y_{T+H})$ denote future unobserved disease incidence over forecast horizon $H$.

We consider a finite model space of $K = 5$ competing Bayesian statistical and mechanistic-inspired models:
$$\mathcal{M} = \{M_1, M_2, M_3, M_4, M_5\}$$
where each candidate model $M_i$ is characterized by a parameter vector $\theta_i \in \Theta_i$, a prior distribution $P(\theta_i \mid M_i)$, and a count likelihood function $P(D \mid \theta_i, M_i)$.

---

## 2. Candidate Bayesian Epidemiological Models

All candidate models use standardized temporal predictors learned strictly on the training window $[1, T]$:
$$\tilde{t} = \frac{t - \mu_t}{\sigma_t}, \qquad \mu_t = \frac{1}{T}\sum_{t=1}^T t, \qquad \sigma_t = \sqrt{\frac{1}{T}\sum_{t=1}^T (t - \mu_t)^2}$$

### 2.1 Model $M_1$: Bayesian Poisson Generalized Linear Model (`M1_POISSON`)
Models weekly count incidence under the equidispersion hypothesis $\mathbb{E}[Y_t \mid \lambda_t] = \text{Var}(Y_t \mid \lambda_t) = \lambda_t$:
$$y_t \mid \lambda_t \sim \text{Poisson}(\lambda_t), \qquad P(y_t \mid \lambda_t) = \frac{\lambda_t^{y_t} e^{-\lambda_t}}{y_t!}$$
$$\log(\lambda_t) = \beta_0 + \beta_{\text{trend}} \tilde{t} + \beta_{\text{sin1}} \sin\left(\frac{2\pi t}{P}\right) + \beta_{\text{cos1}} \cos\left(\frac{2\pi t}{P}\right)$$
with priors $\beta_0 \sim \mathcal{N}(\log(\bar{y}_{\text{train}}), 1.5^2)$, $\beta_{\text{trend}} \sim \mathcal{N}(0, 0.8^2)$, and $\beta_{\text{sin1}}, \beta_{\text{cos1}} \sim \mathcal{N}(0, 1.0^2)$.

### 2.2 Model $M_2$: Bayesian Negative Binomial Regression (`M2_NEGBIN`)
Accounts for overdispersion ($\text{Var}(Y_t) > \mathbb{E}[Y_t]$) caused by superspreading events, spatial heterogeneity, and surveillance reporting batching:
$$y_t \mid \mu_t, \alpha \sim \text{NegativeBinomial}(\mu_t, \alpha)$$
$$P(y_t \mid \mu_t, \alpha) = \frac{\Gamma(y_t + \alpha)}{\Gamma(\alpha) \, y_t!} \left(\frac{\alpha}{\alpha + \mu_t}\right)^\alpha \left(\frac{\mu_t}{\alpha + \mu_t}\right)^{y_t}$$
Under this $(\mu, \alpha)$ parameterization:
$$\mathbb{E}[Y_t \mid \mu_t, \alpha] = \mu_t, \qquad \text{Var}(Y_t \mid \mu_t, \alpha) = \mu_t + \frac{\mu_t^2}{\alpha}$$
with dispersion prior $\alpha \sim \text{Gamma}(2.0, 0.5)$.

### 2.3 Model $M_3$: Bayesian Autoregressive Count Model (`M3_AR_COUNT`)
Captures short-term epidemic momentum via an order-2 log-linear autoregressive (INGARCH) structure:
$$y_t \mid \mu_t, \alpha \sim \text{NegativeBinomial}(\mu_t, \alpha)$$
$$\log(\mu_t) = \beta_0 + \beta_{\text{trend}} \tilde{t} + \phi_1 \tilde{\ell}_{t-1} + \phi_2 \tilde{\ell}_{t-2} + \beta_{\text{sin1}} \sin\left(\frac{2\pi t}{P}\right) + \beta_{\text{cos1}} \cos\left(\frac{2\pi t}{P}\right)$$
where $\tilde{\ell}_{t-j} = \log(y_{t-j} + 1) - \overline{\log(y_{\text{train}} + 1)}$.
During $H$-step forecasting ($h = 1, \dots, H$), future actuals are **never** accessed: each posterior draw $s \in \{1, \dots, S\}$ recursively feeds its own sampled trajectory $\tilde{y}^{(s)}_{T+h-1}$ into the lag terms for step $T+h$.

### 2.4 Model $M_4$: Bayesian Multi-Harmonic Seasonal Model (`M4_SEASONAL_FOURIER`)
Captures asymmetric annual and semi-annual epidemic waves via $K = 3$ Fourier harmonic pairs:
$$\log(\mu_t) = \beta_0 + \beta_{\text{trend}} \tilde{t} + \sum_{k=1}^{3} \left[ a_k \sin\left(\frac{2\pi k t}{P}\right) + b_k \cos\left(\frac{2\pi k t}{P}\right) \right]$$
with frequency-shrinking priors $a_k, b_k \sim \mathcal{N}(0, (1/k)^2)$.

### 2.5 Model $M_5$: Bayesian Renewal / Density-Dependent Growth Model (`M5_RENEWAL_GROWTH`)
Grounded in the discrete renewal equation $\mathbb{E}[Y_t] = I_{t-1} \cdot \mathcal{R}_t$, where $I_{t-1} = \sum_{\tau=1}^3 w_\tau y_{t-\tau} + 1$ ($w = [0.5, 0.3, 0.2]$) is the serial-interval weighted infectious load:
$$\log(\mu_t) = \log(I_{t-1}) + r_0 + r_{\text{trend}} \tilde{t} + r_{\text{damp}} \left(\log(I_{t-1}) - \overline{\log(y_{\text{train}}+1)}\right) + s_1 \sin\left(\frac{2\pi t}{P}\right) + c_1 \cos\left(\frac{2\pi t}{P}\right)$$
where $r_{\text{damp}} \sim \mathcal{N}(-0.15, 0.25^2)$ captures density-dependent damping (susceptible depletion at epidemic peaks).

---

## 3. Marginal Likelihood Estimation via Sequential Monte Carlo (SMC)

The posterior probability of model $M_i$ given training data $D$ is:
$$P(M_i \mid D) = \frac{P(D \mid M_i) P(M_i)}{\sum_{j \in \mathcal{V}} P(D \mid M_j) P(M_j)}$$
where $\mathcal{V} \subseteq \mathcal{M}$ is the subset of candidate models that pass convergence diagnostics ($\hat{R} \le 1.10$, $\text{ESS} \ge 40$, finite evidence) and $P(D \mid M_i)$ is the marginal likelihood:
$$P(D \mid M_i) = \int_{\Theta_i} P(D \mid \theta_i, M_i) P(\theta_i \mid M_i) \, d\theta_i$$

We use **Sequential Monte Carlo (SMC) tempering** (`pm.sample_smc`), which constructs a sequence of annealed intermediate distributions along inverse temperatures $0 = \beta_0 < \beta_1 < \dots < \beta_M = 1$:
$$\pi_m(\theta_i) \propto \left[ P(D \mid \theta_i, M_i) \right]^{\beta_m} P(\theta_i \mid M_i)$$
At each stage $m$, the ratio of normalizing constants $Z_m / Z_{m-1}$ is estimated unbiasedly by the particle importance weights:
$$\log \hat{P}(D \mid M_i) = \sum_{m=1}^{M} \log \left( \frac{1}{N_{\text{particles}}} \sum_{n=1}^{N_{\text{particles}}} \left[ P(D \mid \theta_i^{(n, m-1)}, M_i) \right]^{\beta_m - \beta_{m-1}} \right)$$
Across $C$ independent SMC chains with log-evidence estimates $\ell_{i,1}, \dots, \ell_{i,C}$, the combined estimate uses the log-mean-exp identity:
$$\ell_i = \text{logsumexp}_{c=1}^C(\ell_{i,c}) - \log C$$

---

## 4. Posterior Predictive Mixture & Law of Total Variance Decomposition

The BMA posterior predictive distribution is the discrete count mixture:
$$P(\mathbf{y}_{\text{future}} \mid D) = \sum_{i \in \mathcal{V}} P(\mathbf{y}_{\text{future}} \mid D, M_i) P(M_i \mid D)$$

Let $w_i = P(M_i \mid D)$. By the **Law of Total Variance**, total predictive variance at horizon step $T+h$ decomposes exactly into:
$$\text{Var}(Y_{T+h} \mid D) = \underbrace{\sum_{i \in \mathcal{V}} w_i \, \text{Var}(Y_{T+h} \mid D, M_i)}_{\text{Within-Model Variance}} + \underbrace{\sum_{i \in \mathcal{V}} w_i \left( \mathbb{E}[Y_{T+h} \mid D, M_i] - \mathbb{E}[Y_{T+h} \mid D] \right)^2}_{\text{Between-Model Structural Variance}}$$
Furthermore, within each model $M_i$, $\text{Var}(Y_{T+h} \mid D, M_i)$ decomposes into:
1. **Parameter Uncertainty**: $\text{Var}_{\theta_i \mid D}(\mathbb{E}[Y_{T+h} \mid \theta_i, M_i])$
2. **Aleatoric Observation Variance**: $\mathbb{E}_{\theta_i \mid D}[\text{Var}(Y_{T+h} \mid \theta_i, M_i)]$

