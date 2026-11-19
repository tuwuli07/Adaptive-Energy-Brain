import React, { lazy, Suspense } from 'react';
import { DataSourceProvider } from './context/DataSourceContext';
import { Skeleton } from './components/SourceBoundary';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
const OverviewDashboard = lazy(() => import('./pages/OverviewDashboard'));
const EnergyMonitoring = lazy(() => import('./pages/EnergyMonitoring'));
const ForecastingPage = lazy(() => import('./pages/ForecastingPage'));
const BatteryManagement = lazy(() => import('./pages/BatteryManagement'));
const ControllerComparison = lazy(() => import('./pages/ControllerComparison'));
const RiskAnalytics = lazy(() => import('./pages/RiskAnalytics'));
const AdaptiveControl = lazy(() => import('./pages/AdaptiveControl'));
const SimulationExplorer = lazy(() => import('./pages/SimulationExplorer'));
const ResearchResults = lazy(() => import('./pages/ResearchResults'));
const AboutPage = lazy(() => import('./pages/AboutPage'));

const MyEnergyData = lazy(() => import('./pages/MyEnergyData'));

export default function App() {
  return (
    <DataSourceProvider><BrowserRouter><Suspense fallback={<div className="p-8"><Skeleton /></div>}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<OverviewDashboard />} />
          <Route path="data" element={<MyEnergyData />} />
          <Route path="energy" element={<EnergyMonitoring />} />
          <Route path="forecasting" element={<ForecastingPage />} />
          <Route path="battery" element={<BatteryManagement />} />
          <Route path="comparison" element={<ControllerComparison />} />
          <Route path="risk" element={<RiskAnalytics />} />
          <Route path="adaptive" element={<AdaptiveControl />} />
          <Route path="assistant" element={<Navigate to="/?assistant=open" replace />} />
          <Route path="explorer" element={<SimulationExplorer />} />
          <Route path="research" element={<ResearchResults />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense></BrowserRouter></DataSourceProvider>
  );
}
