/**
 * API Client for the Disease Outbreak Prediction System Using Bayesian Model Averaging.
 * Connects strictly to actual FastAPI backend endpoints without any mock or hardcoded data.
 */

const API_BASE = '/api/v1';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${url}`, options);
  if (!resp.ok) {
    let errDetail = `HTTP ${resp.status}`;
    try {
      const errObj = await resp.json();
      if (errObj.detail) errDetail = errObj.detail;
    } catch {
      // ignore json parse failure
    }
    throw new Error(errDetail);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  getDatasets: () => fetchJson<any[]>('/datasets'),

  getDiseases: () => fetchJson<any[]>('/diseases'),

  getLocations: (diseaseCode?: string) =>
    fetchJson<any[]>(`/locations${diseaseCode ? `?disease_code=${encodeURIComponent(diseaseCode)}` : ''}`),

  getOverview: (diseaseCode: string, locationCode: string) =>
    fetchJson<any>(
      `/overview?disease_code=${encodeURIComponent(diseaseCode)}&location_code=${encodeURIComponent(locationCode)}`
    ),

  getObservations: (
    diseaseCode: string,
    locationCode: string,
    startDate?: string,
    endDate?: string,
    limit = 600
  ) => {
    const params = new URLSearchParams({
      disease_code: diseaseCode,
      location_code: locationCode,
      limit: String(limit),
    });
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    return fetchJson<any>(`/observations?${params.toString()}`);
  },

  getEDA: (diseaseCode: string, locationCode: string, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams({
      disease_code: diseaseCode,
      location_code: locationCode,
    });
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    return fetchJson<any>(`/eda?${params.toString()}`);
  },

  getModelAnalysis: (diseaseCode: string, locationCode: string, runId?: number) => {
    const params = new URLSearchParams({
      disease_code: diseaseCode,
      location_code: locationCode,
    });
    if (runId) params.set('run_id', String(runId));
    return fetchJson<any>(`/models/analysis?${params.toString()}`);
  },

  getForecasts: (diseaseCode: string, locationCode: string, runId?: number, historyWeeks = 52) => {
    const params = new URLSearchParams({
      disease_code: diseaseCode,
      location_code: locationCode,
      history_weeks: String(historyWeeks),
    });
    if (runId) params.set('run_id', String(runId));
    return fetchJson<any>(`/forecasts?${params.toString()}`);
  },

  getUncertainty: (diseaseCode: string, locationCode: string, runId?: number) => {
    const params = new URLSearchParams({
      disease_code: diseaseCode,
      location_code: locationCode,
    });
    if (runId) params.set('run_id', String(runId));
    return fetchJson<any>(`/uncertainty?${params.toString()}`);
  },

  getEvaluation: (diseaseCode: string, locationCode: string, runId?: number) => {
    const params = new URLSearchParams({
      disease_code: diseaseCode,
      location_code: locationCode,
    });
    if (runId) params.set('run_id', String(runId));
    return fetchJson<any>(`/evaluation?${params.toString()}`);
  },

  triggerTrainingRun: (payload: {
    disease_code: string;
    location_code: string;
    forecast_horizon: number;
    smc_draws: number;
    smc_chains: number;
    random_seed: number;
    run_backtest: boolean;
    backtest_folds: number;
    backtest_horizon: number;
    outbreak_z_multiplier: number;
  }, sync = false) =>
    fetchJson<any>(`/training/run?sync=${sync}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  getTrainingRunStatus: (runId: number) => fetchJson<any>(`/training/runs/${runId}/status`),

  uploadDataset: (file: File, datasetName: string, sourceName: string, attribution: string) => {
    const form = new FormData();
    form.append('file', file);
    if (datasetName) form.append('dataset_name', datasetName);
    if (sourceName) form.append('source_name', sourceName);
    if (attribution) form.append('attribution', attribution);
    return fetchJson<any>('/datasets/upload', {
      method: 'POST',
      body: form,
    });
  },
};

