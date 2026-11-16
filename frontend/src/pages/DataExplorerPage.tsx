import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  BarChart2,
  CheckCircle,
  Database,
  FileSpreadsheet,
  Filter,
  Upload,
} from 'lucide-react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';

interface Props {
  diseaseCode: string;
  locationCode: string;
  onSelectTarget: (disease: string, location: string) => void;
  onDatasetsChanged: () => void;
}

export function DataExplorerPage({
  diseaseCode,
  locationCode,
  onSelectTarget,
  onDatasetsChanged,
}: Props) {
  const [eda, setEda] = useState<any | null>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [sourceName, setSourceName] = useState('Public Health Surveillance Upload');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const loadExplorerData = () => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.getEDA(diseaseCode, locationCode, startDate || undefined, endDate || undefined),
      api.getDatasets(),
    ])
      .then(([edaRes, dsRes]) => {
        setEda(edaRes);
        setDatasets(dsRes);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadExplorerData();
  }, [diseaseCode, locationCode]);

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const res = await api.uploadDataset(
        uploadFile,
        datasetName || uploadFile.name,
        sourceName,
        'Uploaded via Data Explorer'
      );
      setUploadResult(res);
      setUploadFile(null);
      setDatasetName('');
      onDatasetsChanged();
      loadExplorerData();
      // If new disease/location detected, allow switching to it
      if (
        res.ingestion_report?.diseases_detected?.length &&
        res.ingestion_report?.locations_detected?.length
      ) {
        onSelectTarget(
          res.ingestion_report.diseases_detected[0],
          res.ingestion_report.locations_detected[0]
        );
      }
    } catch (err: any) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  if (loading && !eda) {
    return (
      <div className="p-8 text-sm text-slate-600 flex items-center gap-3">
        <div className="w-4 h-4 border-2 border-slate-800 border-t-transparent rounded-full animate-spin" />
        Running Exploratory Data Analysis (EDA), seasonal decomposition, and missingness audit...
      </div>
    );
  }

  const stats = eda?.summary_statistics || {};
  const tsSlice = (eda?.time_series || []).slice(-156); // Last 3 years for crisp rendering

  return (
    <div className="space-y-6">
      {/* Top Bar: Date Range Filter & CSV/Excel Upload */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-md p-5">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide mb-3">
            <Filter className="w-4 h-4 text-slate-600" />
            <span>Historical Date Range & Surveillance Audit ({diseaseCode} • {locationCode})</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-end">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Start Date (ISO)</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full border border-slate-300 rounded px-2.5 py-1.5 text-xs font-mono"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">End Date (ISO)</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full border border-slate-300 rounded px-2.5 py-1.5 text-xs font-mono"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={loadExplorerData}
                className="px-3.5 py-1.5 bg-slate-900 text-white rounded text-xs font-medium hover:bg-slate-800"
              >
                Apply Date Filter
              </button>
              <button
                onClick={() => {
                  setStartDate('');
                  setEndDate('');
                  setTimeout(loadExplorerData, 10);
                }}
                className="px-3 py-1.5 border border-slate-300 text-slate-700 rounded text-xs hover:bg-slate-50"
              >
                Reset
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-3 p-2.5 bg-red-50 border border-red-200 rounded text-xs text-red-800">
              {error}
            </div>
          )}

          {/* Statistical & Dispersion Summary Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 pt-4 border-t border-slate-100">
            <div>
              <div className="text-[11px] font-mono text-slate-500">VALID / MISSING OBS.</div>
              <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                {stats.valid_observations} /{' '}
                <span className={stats.missing_observations > 0 ? 'text-amber-700' : 'text-emerald-700'}>
                  {stats.missing_observations} ({stats.missing_pct}%)
                </span>
              </div>
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-500">MEAN ± STD CASES</div>
              <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                {Number(stats.mean_cases || 0).toLocaleString()} ± {Number(stats.std_cases || 0).toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-500">VARIANCE-TO-MEAN (VMR)</div>
              <div className="text-base font-bold font-mono text-indigo-950 mt-0.5">
                {Number(stats.variance_to_mean_ratio || 0).toLocaleString()}x{' '}
                <span className="text-[10px] font-sans px-1.5 py-0.5 bg-indigo-50 text-indigo-800 rounded border border-indigo-200">
                  {stats.overdispersed ? 'Overdispersed' : 'Equidispersed'}
                </span>
              </div>
            </div>
            <div>
              <div className="text-[11px] font-mono text-slate-500">CUMULATIVE / ZERO WEEKS</div>
              <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                {Number(stats.total_cumulative_cases || 0).toLocaleString()} / {stats.zero_count} ({stats.zero_pct}%)
              </div>
            </div>
          </div>
        </div>

        {/* CSV / Excel Upload Card */}
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
            <Upload className="w-4 h-4 text-slate-700" />
            <span>Ingest Custom CSV / Excel Dataset</span>
          </div>
          <p className="text-xs text-slate-500 mb-3">
            Normalizes columns automatically, removes duplicates, and flags chronological missing gaps.
          </p>
          <form onSubmit={handleUploadSubmit} className="space-y-2.5 text-xs">
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              className="block w-full text-xs text-slate-600 file:mr-3 file:py-1 file:px-2.5 file:rounded file:border file:border-slate-300 file:text-xs file:font-medium file:bg-slate-50 file:text-slate-800 hover:file:bg-slate-100"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                type="text"
                placeholder="Dataset Title (optional)"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="border border-slate-300 rounded px-2 py-1 text-xs"
              />
              <input
                type="text"
                placeholder="Source Agency"
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                className="border border-slate-300 rounded px-2 py-1 text-xs"
              />
            </div>
            <button
              type="submit"
              disabled={!uploadFile || uploading}
              className="w-full py-1.5 bg-slate-900 text-white rounded font-medium hover:bg-slate-800 disabled:opacity-50"
            >
              {uploading ? 'Validating & Normalizing Schema...' : 'Upload & Normalize Dataset'}
            </button>
          </form>

          {uploadError && (
            <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-[11px] text-red-800">
              {uploadError}
            </div>
          )}
          {uploadResult && (
            <div className="mt-2 p-2 bg-emerald-50 border border-emerald-200 rounded text-[11px] text-emerald-900">
              <div className="font-semibold flex items-center gap-1">
                <CheckCircle className="w-3.5 h-3.5" /> Ingested {uploadResult.dataset.row_count} rows
              </div>
              <div>
                Duplicates removed: {uploadResult.dataset.duplicate_count} • Missing tracked:{' '}
                {uploadResult.dataset.missing_count}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chart Row 1: Historical Incidence + 4w/12w Moving Averages + Seasonal Decomposition */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            1. Weekly Surveillance Incidence & Rolling Averages (4W / 12W MA)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Reported weekly counts alongside causal 4-week and 12-week moving averages.
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={tsSlice} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="observation_date" tick={{ fontSize: 10 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="reported_cases" name="Reported Weekly Cases" fill="#cbd5e1" />
                <Line type="monotone" dataKey="ma_4w" name="4-Week Moving Average" stroke="#1d4ed8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="ma_12w" name="12-Week Moving Average" stroke="#0f172a" strokeWidth={2} strokeDasharray="4 2" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            2. Seasonal Profile by Epidemiological Week (ISO Week 1–52)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Multi-year weekly endemic profile showing mean, median, and 25th–90th percentile bands.
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={eda?.seasonal_profile || []} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="week_of_year" tick={{ fontSize: 10 }} label={{ value: 'Epidemiological Week', position: 'insideBottom', offset: -3, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area type="monotone" dataKey="q90_cases" name="90th Percentile Envelope" fill="#e0e7ff" stroke="#818cf8" fillOpacity={0.5} />
                <Line type="monotone" dataKey="mean_cases" name="Historical Mean Cases" stroke="#1e3a8a" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="median_cases" name="Historical Median Cases" stroke="#047857" strokeWidth={1.8} strokeDasharray="3 3" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Chart Row 2: Classical Seasonal Decomposition & Monthly Trend Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            3. Additive Time-Series Decomposition (Trend + Seasonal + Residual)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Statsmodels decomposition separating secular trend from recurring annual seasonality and residual shocks.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={tsSlice} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="observation_date" tick={{ fontSize: 10 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="trend_component" name="Secular Trend" stroke="#0f172a" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="seasonal_component" name="Seasonal Component" stroke="#2563eb" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="residual_component" name="Residual Shock" stroke="#dc2626" strokeWidth={1} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-1">
            4. Monthly Aggregated Incidence & Growth Rate Analysis
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Monthly mean weekly cases and peak weekly incidence across the surveillance horizon.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={(eda?.monthly_trends || []).slice(-36)} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="year_month" tick={{ fontSize: 10 }} minTickGap={18} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="mean_weekly_cases" name="Mean Weekly Cases" fill="#1e3a8a" />
                <Bar dataKey="max_weekly_cases" name="Peak Weekly Cases in Month" fill="#93c5fd" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 3: Geographic Comparison, Explanatory Correlations & Sudden Surge Anomalies */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
            5. Geographic Comparison ({diseaseCode})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 font-mono text-slate-600">
                  <th className="py-2 px-2">Region</th>
                  <th className="py-2 px-2 text-right">Mean/Wk</th>
                  <th className="py-2 px-2 text-right">Peak</th>
                  <th className="py-2 px-2 text-right">/100k</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {(eda?.geographic_comparison || []).map((g: any) => (
                  <tr
                    key={g.location_code}
                    onClick={() => onSelectTarget(diseaseCode, g.location_code)}
                    className={`cursor-pointer hover:bg-slate-50 ${
                      g.location_code === locationCode ? 'bg-indigo-50/60 font-semibold' : ''
                    }`}
                  >
                    <td className="py-1.5 px-2 font-mono">{g.location_code}</td>
                    <td className="py-1.5 px-2 text-right font-mono">{Number(g.mean_weekly_cases).toLocaleString()}</td>
                    <td className="py-1.5 px-2 text-right font-mono">{Number(g.max_weekly_cases).toLocaleString()}</td>
                    <td className="py-1.5 px-2 text-right font-mono">
                      {g.mean_incidence_per_100k ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
            6. Explanatory Variable Correlations & Distribution
          </h3>
          <div className="space-y-2 mb-4">
            {(eda?.correlations || []).map((c: any) => (
              <div key={c.variable} className="flex items-center justify-between text-xs border-b border-slate-100 pb-1">
                <span className="font-mono text-slate-700">{c.variable}</span>
                <span className="font-mono font-semibold text-slate-900">
                  r = {c.pearson_r_with_cases >= 0 ? '+' : ''}
                  {c.pearson_r_with_cases.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={eda?.distribution_bins || []}>
                <XAxis dataKey="bin_label" tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} />
                <Tooltip contentStyle={{ fontSize: '11px' }} />
                <Bar dataKey="frequency" name="Weeks in Case Count Bin" fill="#334155" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-md p-5">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
            7. Detected Outliers & Sudden Surges ({eda?.total_anomalies_detected || 0})
          </h3>
          <div className="max-h-60 overflow-y-auto divide-y divide-slate-100 text-xs">
            {(eda?.anomalies_and_surges || []).slice().reverse().map((a: any, i: number) => (
              <div key={i} className="py-2 flex items-center justify-between">
                <div>
                  <div className="font-mono font-semibold text-slate-900">{a.observation_date}</div>
                  <div className="text-[11px] text-slate-500">{a.type}</div>
                </div>
                <div className="text-right font-mono">
                  <div className="font-bold text-red-800">{Number(a.reported_cases).toLocaleString()} cases</div>
                  <div className="text-[11px] text-slate-500">Z = +{a.z_score}σ (4w MA: {a.ma_4w})</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Dataset Provenance Registry */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-900 uppercase tracking-wide mb-3">
          <Database className="w-4 h-4 text-slate-700" />
          <span>Registered Epidemiological Datasets & Provenance Checksums</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 font-mono text-slate-600">
                <th className="py-2 px-3">ID</th>
                <th className="py-2 px-3">Dataset Name</th>
                <th className="py-2 px-3">Official Source & Attribution</th>
                <th className="py-2 px-3">Date Span</th>
                <th className="py-2 px-3 text-right">Rows</th>
                <th className="py-2 px-3 text-right">Missing</th>
                <th className="py-2 px-3">SHA-256 Checksum</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {datasets.map((ds) => (
                <tr key={ds.id} className="hover:bg-slate-50">
                  <td className="py-2 px-3 font-mono">#{ds.id}</td>
                  <td className="py-2 px-3 font-medium text-slate-900">{ds.name}</td>
                  <td className="py-2 px-3 text-slate-600">{ds.source_name}</td>
                  <td className="py-2 px-3 font-mono">
                    {ds.date_start} → {ds.date_end}
                  </td>
                  <td className="py-2 px-3 text-right font-mono">{ds.row_count.toLocaleString()}</td>
                  <td className="py-2 px-3 text-right font-mono">{ds.missing_count}</td>
                  <td className="py-2 px-3 font-mono text-[11px] text-slate-500">
                    {ds.file_hash?.slice(0, 16)}…
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

