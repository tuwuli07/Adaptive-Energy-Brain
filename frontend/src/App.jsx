import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import OverviewDashboard from './pages/OverviewDashboard';
import EnergyMonitoring from './pages/EnergyMonitoring';
import ForecastingPage from './pages/ForecastingPage';
import BatteryManagement from './pages/BatteryManagement';
import ControllerComparison from './pages/ControllerComparison';
import RiskAnalytics from './pages/RiskAnalytics';
import AdaptiveControl from './pages/AdaptiveControl';
import SimulationExplorer from './pages/SimulationExplorer';
import ResearchResults from './pages/ResearchResults';
import AboutPage from './pages/AboutPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<OverviewDashboard />} />
          <Route path="energy" element={<EnergyMonitoring />} />
          <Route path="forecasting" element={<ForecastingPage />} />
          <Route path="battery" element={<BatteryManagement />} />
          <Route path="comparison" element={<ControllerComparison />} />
          <Route path="risk" element={<RiskAnalytics />} />
          <Route path="adaptive" element={<AdaptiveControl />} />
          <Route path="explorer" element={<SimulationExplorer />} />
          <Route path="research" element={<ResearchResults />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
