import React, { useState, useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import { Database, CheckCircle2, AlertCircle } from 'lucide-react';
import { api } from '../api/client';

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });
  const [apiOnline, setApiOnline] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  useEffect(() => {
    api.getHealth()
      .then(res => setApiOnline(res.status === 'online'))
      .catch(() => setApiOnline(false));
  }, [location.pathname]);

  const pageTitles = {
    '/': 'Overview Dashboard',
    '/energy': 'Energy Monitoring (Historical Simulation Data)',
    '/forecasting': 'AI Energy Forecasting (GNN-Transformer)',
    '/battery': 'Battery Storage Management & Dispatch',
    '/comparison': 'Controller Comparative Benchmark',
    '/explorer': 'Simulation Run Explorer',
    '/research': 'Scientific Research Results & Audit',
    '/about': 'About Adaptive Energy Brain',
  };

  const currentTitle = pageTitles[location.pathname] || 'Dashboard';

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex transition-colors duration-200">
      {/* Sidebar Navigation */}
      <Sidebar
        collapsed={collapsed}
        setCollapsed={setCollapsed}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />

      {/* Main Content Area */}
      <div
        className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ease-in-out ${
          collapsed ? 'ml-20' : 'ml-64'
        }`}
      >
        {/* Top Bar */}
        <header className="h-16 sticky top-0 z-30 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800/80 px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              {currentTitle}
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs">
            {/* Simulation Status Badge */}
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200/60 dark:border-slate-700/60">
              <Database className="w-3.5 h-3.5 text-slate-500" />
              <span>Winter Test Partition (1,313 hrs)</span>
            </div>

            {/* API Connection Indicator */}
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full font-medium border ${
                apiOnline
                  ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800'
                  : 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border-amber-200 dark:border-amber-800'
              }`}
              title={apiOnline ? 'Connected to local Python REST API (Port 8000)' : 'Standalone mode (using verified Phase 8.5 fallback data)'}
            >
              {apiOnline ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  <span>REST API Connected</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                  <span>Standalone Cache Active</span>
                </>
              )}
            </div>
          </div>
        </header>

        {/* Page View Body */}
        <main className="p-6 md:p-8 flex-1 max-w-7xl w-full mx-auto space-y-6">
          <Outlet />
        </main>

        {/* Footer */}
        <footer className="px-8 py-4 border-t border-slate-200/60 dark:border-slate-800/60 text-xs text-slate-400 dark:text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Adaptive Energy Brain • Phase 9 Research & Analytics Frontend</span>
          <span>Authoritative Phase 8.5 Results Reconciled • Read-Only Interface</span>
        </footer>
      </div>
    </div>
  );
}
