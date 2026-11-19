import React, { useState, useEffect } from 'react';
import { Outlet, useLocation, Link } from 'react-router-dom';
import SourceBoundary from './SourceBoundary';
import { useDataSource } from '../context/DataSourceContext';
import FloatingAssistant, { AssistantContext } from './FloatingAssistant';
import Sidebar from './Sidebar';
import { Database, CheckCircle2, AlertCircle, Sparkles, Bot, ShieldCheck } from 'lucide-react';
import { api } from '../api/client';

export default function Layout() {
  const { source, setSource, datasets, activeMeta } = useDataSource();
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('energy-appearance') !== 'light';
  });
  const [apiOnline, setApiOnline] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('energy-appearance', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('energy-appearance', 'light');
    }
  }, [darkMode]);

  useEffect(() => {
    api.getHealth()
      .then(res => setApiOnline(res.status === 'online'))
      .catch(() => setApiOnline(false));
  }, [location.pathname]);

  useEffect(() => {
    if (new URLSearchParams(location.search).get('assistant') === 'open') setAssistantOpen(true);
  }, [location.search]);

  const pageMeta = {
    '/': ['Overview', 'Your energy workspace at a glance'],
    '/data': ['My Energy Data', 'Upload, validate, and explore your measurements'],
    '/energy': ['Energy use', 'See where your power comes from and where it goes'],
    '/forecasting': ['Energy forecast', 'Explore expected demand and renewable generation'],
    '/battery': ['Battery storage', 'Understand stored energy, charging, and battery wear'],
    '/adaptive': ['Smart control', 'See how energy strategies adapt to changing conditions'],
    '/comparison': ['Compare strategies', 'Explore the costs and savings of each approach'],
    '/risk': ['Risk & reliability', 'Understand how the system responds to uncertainty'],
    '/explorer': ['Scenario explorer', 'Explore energy decisions, one hour at a time'],
    '/research': ['Study results', 'Evidence and limitations of the reference study'],
    '/assistant': ['Energy assistant', 'Get help understanding your energy insights'],
    '/about': ['About the project', 'A closer look at Adaptive Energy Brain'],
  };
  const [title, subtitle] = pageMeta[location.pathname] || pageMeta['/'];
  const currentMeta = { title, subtitle };

  return (
    <AssistantContext.Provider value={() => setAssistantOpen(true)}><div className="app-shell min-h-screen flex transition-colors duration-200">
      {/* Sidebar Navigation */}
      <Sidebar
        collapsed={collapsed}
        setCollapsed={setCollapsed}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />

      {/* Main Content Area */}
      <div
        className={`app-content min-w-0 flex-1 flex flex-col min-h-screen transition-all duration-300 ease-in-out ${
          collapsed ? 'ml-20' : 'ml-64'
        }`}
      >
        {/* Top Header Bar */}
        <header className="app-header h-20 sticky top-0 z-30 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800/80 px-6 flex items-center justify-between gap-4">
          <div className="flex flex-col">
            <span className="text-sm font-bold text-slate-900 dark:text-slate-100 tracking-tight">
              {currentMeta.title}
            </span>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:block truncate max-w-md">
              {currentMeta.subtitle}
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <label className="source-select">
              <span className="sr-only">Active data source</span>
              <select aria-label="Active data source" value={source} onChange={e => setSource(e.target.value)}>
                <option value="none">Select energy data</option>
                {source !== 'none' && !datasets.some(d => d.id === source) && <option value={source}>My Uploaded Data (unavailable)</option>}
                {datasets.filter(d => d.status === 'ready').map(d => <option key={d.id} value={d.id}>My Uploaded Data / {d.name}</option>)}
              </select>
            </label>
            <Link to="/data" className="data-link">My data</Link>

            {/* API Connection Indicator */}
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full font-medium border ${
                apiOnline
                  ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border-blue-200 dark:border-blue-800'
                  : 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border-amber-200 dark:border-amber-800'
              }`}
              title={apiOnline ? 'Connected to study results' : 'Showing saved results and illustrative sample charts'}
            >
              {apiOnline ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-500" />
                  <span>Connected</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                  <span>Demo data</span>
                </>
              )}
            </div>
          </div>
        </header>

        {/* Page View Body */}
        <main className="p-4 sm:p-6 md:p-8 flex-1 max-w-[1600px] w-full mx-auto space-y-6">
          <SourceBoundary />
        </main>

        {/* Cohesive Footer */}
        <footer className="px-6 py-4 border-t border-slate-200/60 dark:border-slate-800/60 text-xs text-slate-500 dark:text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
            <ShieldCheck className="w-4 h-4 text-blue-500" />
            <span>Adaptive Energy Brain</span>
          </div>
          <span className="text-[11px] text-slate-400 dark:text-slate-500 text-center sm:text-right">
            {source === 'none' ? 'Your energy workspace' : activeMeta?.name || 'Uploaded data'} / No live equipment connected
          </span>
        </footer>
      </div>
      <FloatingAssistant open={assistantOpen} setOpen={setAssistantOpen} />
    </div></AssistantContext.Provider>
  );
}
