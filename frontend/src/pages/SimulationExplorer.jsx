import React, { useState, useEffect } from 'react';
import {
  Compass,
  Download,
  Filter,
  Layers,
  Zap,
  ArrowRight,
  Database,
  Calendar,
  FileCode,
  FileSpreadsheet
} from 'lucide-react';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import CausalityBadge from '../components/CausalityBadge';
import { formatCurrency, formatNumber, formatDate } from '../utils/formatters';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';

export default function SimulationExplorer() {
  const [controllerId, setControllerId] = useState('mpc_gnn_transformer');
  const [windowHours, setWindowHours] = useState(72);
  const [startStep, setStartStep] = useState(480); // Step 480 starts right before the Nov 28 weather event!
  const [downsample, setDownsample] = useState(1);
  const [timeseries, setTimeseries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const res = await api.getTimeseries({
          controllerId,
          partition: 'test',
          limit: windowHours,
          downsample,
        });
        setTimeseries(res.data || []);
      } catch (err) {
        console.error('Failed to load explorer timeseries:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [controllerId, windowHours, downsample]);

  // Export handlers
  const handleDownloadCsv = () => {
    window.open(api.getExportUrl(controllerId, 'test', 'csv'), '_blank');
  };

  const handleDownloadJson = () => {
    window.open(api.getExportUrl(controllerId, 'test', 'json'), '_blank');
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Simulation Run Explorer"
        subtitle="Follow hourly energy use, battery decisions, and costs."
        action={
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadCsv}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-500 hover:bg-blue-600 text-white text-xs font-semibold shadow-xs transition-colors"
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span>Download CSV</span>
            </button>
            <button
              onClick={handleDownloadJson}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-semibold shadow-xs transition-colors"
            >
              <FileCode className="w-3.5 h-3.5" />
              <span>Download JSON</span>
            </button>
          </div>
        }
      />

      {/* Control Configuration Bar */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-4 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div>
            <label className="block text-slate-400 font-semibold mb-1 uppercase tracking-wider text-[10px]">
              Controller Trajectory
            </label>
            <select
              value={controllerId}
              onChange={(e) => setControllerId(e.target.value)}
              className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-1.5 font-semibold text-slate-800 dark:text-slate-200"
            >
              <option value="mpc_gnn_transformer">GNN-Transformer MPC (Causal)</option>
              <option value="mpc_seasonal_persistence">Seasonal Persistence MPC (Causal)</option>
              <option value="price_aware_continuous">Continuous Price-Aware Baseline</option>
              <option value="price_aware_discrete">Discrete Price-Aware Baseline</option>
              <option value="ppo_continuous_control">PPO Continuous Control (16k)</option>
              <option value="zero_action">Zero-Action Baseline</option>
              <option value="mpc_oracle_rolling_24h">Oracle MPC (24h Diagnostic)</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-semibold mb-1 uppercase tracking-wider text-[10px]">
              Inspection Window
            </label>
            <select
              value={windowHours}
              onChange={(e) => setWindowHours(Number(e.target.value))}
              className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-1.5 font-semibold text-slate-800 dark:text-slate-200"
            >
              <option value={24}>24 Hours (1 Day)</option>
              <option value={48}>48 Hours (2 Days)</option>
              <option value={72}>72 Hours (3 Days)</option>
              <option value={168}>168 Hours (1 Week)</option>
              <option value={336}>336 Hours (2 Weeks)</option>
              <option value={1313}>Full Episode (1,313 Hours)</option>
            </select>
          </div>
        </div>

        <CausalityBadge causality={controllerId === 'mpc_oracle_rolling_24h' ? 'noncausal_lookahead' : 'strictly_causal'} />
      </div>

      {/* Primary Chart: Campus Load, Renewables & Grid Flow */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-4">
        <div>
          <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
            Physical Microgrid Dispatch & Power Balance
          </h3>
          <p className="text-xs text-slate-400">
            Positive battery power indicates discharging; negative power indicates charging. Grid import satisfies net unserved campus load.
          </p>
        </div>

        <div className="h-72 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeseries} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis
                  dataKey="timestamp"
                  stroke="#94a3b8"
                  fontSize={10}
                  tickFormatter={(val) => formatDate(val)}
                  tickLine={false}
                />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" kW" />
                <Tooltip formatter={(val, name) => [`${val} kW`, name]} labelFormatter={(lbl) => formatDate(lbl)} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
                <Line type="monotone" dataKey="demand_kw" name="Site Demand (kW)" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="solar_kw" name="Solar Available (kW)" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="wind_kw" name="Wind Available (kW)" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="grid_import_kw" name="Grid Import (kW)" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
                <Line type="stepAfter" dataKey="battery_power_kw" name="Battery Power (kW)" stroke="#1051b9" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Secondary Chart: Step-by-Step Hourly Cost */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-4">
        <div>
          <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
            Hourly Net Operating Cost Breakdown ($ / Hour)
          </h3>
          <p className="text-xs text-slate-400">
            Includes time-of-use electricity import costs, battery cycling degradation ($0.02/kWh), minus export feed-in revenues.
          </p>
        </div>

        <div className="h-56 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeseries} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1051b9" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#1051b9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis
                  dataKey="timestamp"
                  stroke="#94a3b8"
                  fontSize={10}
                  tickFormatter={(val) => formatDate(val)}
                  tickLine={false}
                />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" $" />
                <Tooltip formatter={(val) => [formatCurrency(val), 'Hourly Net Cost']} labelFormatter={(lbl) => formatDate(lbl)} />
                <Area type="monotone" dataKey="cost_usd" name="Net Operating Cost ($)" stroke="#1051b9" fill="url(#costGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
