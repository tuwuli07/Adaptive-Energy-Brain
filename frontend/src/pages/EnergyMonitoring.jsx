import React, { useState, useEffect } from 'react';
import {
  Activity,
  Calendar,
  Layers,
  Zap,
  Sun,
  Wind,
  Battery,
  ArrowUpDown,
  Filter,
  AlertTriangle
} from 'lucide-react';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import StatCard from '../components/StatCard';
import CausalityBadge from '../components/CausalityBadge';
import { formatNumber, formatCurrency, formatDate } from '../utils/formatters';
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
  Brush
} from 'recharts';

export default function EnergyMonitoring() {
  const [controllerId, setControllerId] = useState('mpc_gnn_transformer');
  const [preset, setPreset] = useState('7d'); // '24h', '7d', '14d', 'all'
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Time preset configurations
  const presetConfig = {
    '24h': { limit: 24, downsample: 1 },
    '7d': { limit: 168, downsample: 1 },
    '14d': { limit: 336, downsample: 2 },
    'all': { limit: 1313, downsample: 4 },
  };

  useEffect(() => {
    async function loadTimeseries() {
      setLoading(true);
      try {
        const conf = presetConfig[preset] || presetConfig['7d'];
        const res = await api.getTimeseries({
          controllerId,
          partition: 'test',
          limit: conf.limit,
          downsample: conf.downsample,
        });
        setData(res.data || []);
      } catch (err) {
        console.error('Failed to load telemetry:', err);
      } finally {
        setLoading(false);
      }
    }
    loadTimeseries();
  }, [controllerId, preset]);

  // Aggregate stats over visible window
  const totalDemand = data.reduce((acc, d) => acc + d.demand_kw, 0);
  const totalSolar = data.reduce((acc, d) => acc + d.solar_kw, 0);
  const totalWind = data.reduce((acc, d) => acc + d.wind_kw, 0);
  const totalImport = data.reduce((acc, d) => acc + d.grid_import_kw, 0);
  const peakDemand = data.reduce((max, d) => Math.max(max, d.demand_kw), 0);

  return (
    <div className="space-y-6">
      {/* Header with Historical Simulation Label */}
      <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-4 text-xs text-amber-800 dark:text-amber-300 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
          <span>
            <strong>Data Classification: Synthetic Simulation Dataset</strong> — Replaying evaluated 1,313-hour winter test partition (Nov 7 – Dec 31, 2026) on the demonstration microgrid case study. Not from live physical utility measurements.
          </span>
        </div>
      </div>

      <SectionHeader
        title="Energy Monitoring & Microgrid Telemetry"
        subtitle="Explore high-resolution power flows across facility loads, local renewables, battery storage, and utility grid interconnections."
        action={
          <div className="flex flex-wrap items-center gap-2">
            {/* Controller Selector */}
            <select
              value={controllerId}
              onChange={(e) => setControllerId(e.target.value)}
              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-1.5 text-xs font-medium text-slate-800 dark:text-slate-200 shadow-sm focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
            >
              <option value="mpc_gnn_transformer">GNN-Transformer MPC (Causal)</option>
              <option value="mpc_seasonal_persistence">Seasonal Persistence MPC (Causal)</option>
              <option value="price_aware_continuous">Continuous Price-Aware Baseline</option>
              <option value="price_aware_discrete">Discrete Price-Aware Baseline</option>
              <option value="ppo_continuous_control">PPO Continuous Control (16k)</option>
              <option value="zero_action">Zero-Action Baseline</option>
              <option value="mpc_oracle_rolling_24h">Oracle MPC (24h Diagnostic)</option>
            </select>

            {/* Time Window Presets */}
            <div className="flex items-center bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-1 shadow-sm text-xs">
              {['24h', '7d', '14d', 'all'].map((p) => (
                <button
                  key={p}
                  onClick={() => setPreset(p)}
                  className={`px-3 py-1 rounded-lg font-medium transition-colors ${
                    preset === p
                      ? 'bg-emerald-500 text-white shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                  }`}
                >
                  {p === 'all' ? 'All (1,313h)' : p.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        }
      />

      {/* Overview Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard
          title="Peak Demand"
          value={`${formatNumber(peakDemand, 1)} kW`}
          subtext="Highest single-hour load"
          icon={Zap}
          color="red"
        />
        <StatCard
          title="Solar Generation"
          value={`${formatNumber(totalSolar, 0)} kWh`}
          subtext="300 kW PV Array"
          icon={Sun}
          color="amber"
        />
        <StatCard
          title="Wind Generation"
          value={`${formatNumber(totalWind, 0)} kWh`}
          subtext="150 kW Wind Turbine"
          icon={Wind}
          color="blue"
        />
        <StatCard
          title="Grid Import"
          value={`${formatNumber(totalImport, 0)} kWh`}
          subtext="Max Limit: 2,000 kW"
          icon={ArrowUpDown}
          color="purple"
        />
      </div>

      {/* Main Microgrid Power Balance Time-Series */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
              Campus Power Balance Telemetry ({preset.toUpperCase()} Window)
            </h3>
            <p className="text-xs text-slate-400">
              Campus Demand (kW) vs Local Renewable Generation (Solar + Wind) and Utility Grid Imports
            </p>
          </div>
          <CausalityBadge causality={controllerId === 'mpc_oracle_rolling_24h' ? 'noncausal_lookahead' : 'strictly_causal'} />
        </div>

        <div className="h-80 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="areaDemand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="areaSolar" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="areaWind" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="timestamp"
                  stroke="#94a3b8"
                  fontSize={11}
                  tickFormatter={(val) => formatDate(val)}
                  tickLine={false}
                />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" kW" />
                <Tooltip
                  formatter={(val, name) => [`${val} kW`, name]}
                  labelFormatter={(lbl) => formatDate(lbl)}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Area type="monotone" dataKey="demand_kw" name="Campus Demand" stroke="#ef4444" fill="url(#areaDemand)" strokeWidth={2} />
                <Area type="monotone" dataKey="solar_kw" name="Solar Power" stroke="#f59e0b" fill="url(#areaSolar)" strokeWidth={1.5} />
                <Area type="monotone" dataKey="wind_kw" name="Wind Power" stroke="#06b6d4" fill="url(#areaWind)" strokeWidth={1.5} />
                <Area type="monotone" dataKey="grid_import_kw" name="Grid Import" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.1} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Secondary Chart: Battery Dispatch & State of Charge */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-4">
        <div>
          <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
            Battery Dispatch Power & State of Charge (SOC)
          </h3>
          <p className="text-xs text-slate-400">
            Negative power indicates charging from local renewables/grid; positive power indicates discharging to campus load.
          </p>
        </div>

        <div className="h-64 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <XAxis
                  dataKey="timestamp"
                  stroke="#94a3b8"
                  fontSize={11}
                  tickFormatter={(val) => formatDate(val)}
                  tickLine={false}
                />
                <YAxis yAxisId="power" stroke="#10b981" fontSize={11} tickLine={false} unit=" kW" />
                <YAxis yAxisId="soc" orientation="right" domain={[0, 1]} stroke="#6366f1" fontSize={11} tickLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip
                  formatter={(val, name) => {
                    if (name.includes('SOC')) return [`${(val * 100).toFixed(1)}%`, name];
                    return [`${val} kW`, name];
                  }}
                  labelFormatter={(lbl) => formatDate(lbl)}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Line yAxisId="power" type="stepAfter" dataKey="battery_power_kw" name="Battery Power (kW)" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line yAxisId="soc" type="monotone" dataKey="battery_soc" name="Battery SOC (%)" stroke="#6366f1" strokeWidth={2} strokeDasharray="3 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
