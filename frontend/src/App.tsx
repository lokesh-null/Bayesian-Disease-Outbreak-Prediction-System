import React, { useEffect, useState } from 'react';
import {
  Activity,
  BarChart2,
  BookOpen,
  Cpu,
  Database,
  Layers,
  ShieldAlert,
  TrendingUp,
} from 'lucide-react';
import { api } from './api/client';
import { AboutModelPage } from './pages/AboutModelPage';
import { DataExplorerPage } from './pages/DataExplorerPage';
import { ModelAnalysisPage } from './pages/ModelAnalysisPage';
import { ModelEvaluationPage } from './pages/ModelEvaluationPage';
import { OutbreakForecastPage } from './pages/OutbreakForecastPage';
import { OverviewPage } from './pages/OverviewPage';
import { UncertaintyAnalysisPage } from './pages/UncertaintyAnalysisPage';

const NAV_ITEMS = [
  { id: 'overview', label: '1. Overview', icon: Activity },
  { id: 'explorer', label: '2. Data Explorer', icon: Database },
  { id: 'models', label: '3. Model Analysis', icon: Cpu },
  { id: 'forecast', label: '4. Outbreak Forecast', icon: TrendingUp },
  { id: 'uncertainty', label: '5. Uncertainty Analysis', icon: Layers },
  { id: 'evaluation', label: '6. Model Evaluation', icon: BarChart2 },
  { id: 'about', label: '7. About the Model', icon: BookOpen },
];

export function App() {
  const [activePage, setActivePage] = useState<string>('overview');
  const [diseases, setDiseases] = useState<any[]>([]);
  const [selectedDisease, setSelectedDisease] = useState<string>('ILI');
  const [selectedLocation, setSelectedLocation] = useState<string>('US-NAT');

  const fetchMetadata = () => {
    api
      .getDiseases()
      .then((res) => {
        setDiseases(res);
      })
      .catch(() => {
        // fallback if backend is starting
      });
  };

  useEffect(() => {
    fetchMetadata();
  }, []);

  const currentDiseaseObj = diseases.find((d) => d.code === selectedDisease);
  const availableLocations = currentDiseaseObj?.available_locations || [
    { code: 'US-NAT', name: 'United States (National)' },
  ];

  const handleDiseaseChange = (newDiseaseCode: string) => {
    setSelectedDisease(newDiseaseCode);
    const dObj = diseases.find((d) => d.code === newDiseaseCode);
    if (dObj?.available_locations?.length) {
      const hasCurrent = dObj.available_locations.some((l: any) => l.code === selectedLocation);
      if (!hasCurrent) {
        setSelectedLocation(dObj.available_locations[0].code);
      }
    }
  };

  const handleSelectTarget = (dCode: string, lCode: string) => {
    setSelectedDisease(dCode);
    setSelectedLocation(lCode);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Institutional Navigation Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-3 flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded bg-slate-900 text-white flex items-center justify-center font-mono font-bold text-sm shrink-0">
              BMA
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-sm tracking-tight text-slate-900">
                  EPI-BMA SURVEILLANCE INTELLIGENCE
                </span>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 bg-slate-100 text-slate-700 border border-slate-200 rounded">
                  PyMC v6 SMC + ArviZ
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Disease Outbreak Prediction & Probabilistic Risk Assessment via Bayesian Model Averaging
              </p>
            </div>
          </div>

          {/* Global Disease & Jurisdiction Selector */}
          <div className="flex flex-wrap items-center gap-2.5 text-xs">
            <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded px-2.5 py-1">
              <span className="font-mono text-[11px] text-slate-500 uppercase">Disease:</span>
              <select
                value={selectedDisease}
                onChange={(e) => handleDiseaseChange(e.target.value)}
                className="bg-transparent font-semibold text-slate-900 focus:outline-none"
              >
                {diseases.map((d) => (
                  <option key={d.code} value={d.code}>
                    {d.code} — {d.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded px-2.5 py-1">
              <span className="font-mono text-[11px] text-slate-500 uppercase">Region:</span>
              <select
                value={selectedLocation}
                onChange={(e) => setSelectedLocation(e.target.value)}
                className="bg-transparent font-semibold text-slate-900 focus:outline-none"
              >
                {availableLocations.map((l: any) => (
                  <option key={l.code} value={l.code}>
                    {l.code} — {l.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* 7-Page Tab Bar */}
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 border-t border-slate-100 overflow-x-auto">
          <nav className="flex space-x-1 py-1.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap transition-colors ${
                    isActive
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6">
        {activePage === 'overview' && (
          <OverviewPage
            diseaseCode={selectedDisease}
            locationCode={selectedLocation}
            onNavigate={setActivePage}
          />
        )}
        {activePage === 'explorer' && (
          <DataExplorerPage
            diseaseCode={selectedDisease}
            locationCode={selectedLocation}
            onSelectTarget={handleSelectTarget}
            onDatasetsChanged={fetchMetadata}
          />
        )}
        {activePage === 'models' && (
          <ModelAnalysisPage
            diseaseCode={selectedDisease}
            locationCode={selectedLocation}
          />
        )}
        {activePage === 'forecast' && (
          <OutbreakForecastPage
            diseaseCode={selectedDisease}
            locationCode={selectedLocation}
          />
        )}
        {activePage === 'uncertainty' && (
          <UncertaintyAnalysisPage
            diseaseCode={selectedDisease}
            locationCode={selectedLocation}
          />
        )}
        {activePage === 'evaluation' && (
          <ModelEvaluationPage
            diseaseCode={selectedDisease}
            locationCode={selectedLocation}
          />
        )}
        {activePage === 'about' && <AboutModelPage />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-3 px-4 sm:px-6 text-xs text-slate-500">
        <div className="max-w-[1440px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            Disease Outbreak Prediction System Using Bayesian Model Averaging (BMA) • Genuine CDC ILINet & CDC NNDSS Surveillance Data
          </div>
          <div className="font-mono text-[11px]">
            P(M_i | D) ∝ P(D | M_i)P(M_i) • Sequential Monte Carlo (SMC) Evidence Estimation
          </div>
        </div>
      </footer>
    </div>
  );
}
export default App;

