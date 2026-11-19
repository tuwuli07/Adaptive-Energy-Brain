import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import {
  Sun,
  Wind,
  Zap,
  Battery,
  DollarSign,
  TrendingUp,
  ShieldCheck,
  ArrowRight,
  Sparkles,
  Database
} from 'lucide-react';
import { api } from '../api/client';
import StatCard from '../components/StatCard';
import BatteryGauge from '../components/BatteryGauge';
import SectionHeader from '../components/SectionHeader';
import { formatCurrency, formatEnergy, formatPercent } from '../utils/formatters';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell
} from 'recharts';

export default function OverviewDashboard() {
  const [overview, setOverview] = useState(null);
  const [timeseries, setTimeseries] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [ov, ts, comp] = await Promise.all([
          api.getOverview(),
          api.getTimeseries({ controllerId: 'mpc_gnn_transformer', limit: 48 }),
          api.getControllersComparison('test'),
        ]);
        setOverview(ov);
        setTimeseries(ts.data || []);
        setComparison(comp);
      } catch (err) {
        console.error('Failed to load dashboard overview data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading || !overview) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-slate-500">Loading energy telemetry...</span>
        </div>
      </div>
    );
  }

  const kpis = overview.economic_kpis;
  const env = overview.environment;
  const testPart = overview.test_partition;

  // Bar chart data for controllers
  const barData = (comparison?.controllers || []).map((c) => ({
    name: c.display_name.replace(' Baseline', '').replace(' (16k steps)', '').replace(' (24h Rolling Perfect-Foresight)', ' (Diagnostic)'),
    adjustedCost: c.test.adjusted_cost_usd,
    savings: c.test.savings_vs_zero_usd,
    isDeployable: c.is_deployable,
    isGnn: c.controller_id === 'mpc_gnn_transformer',
  }));

  return (
    <div className="space-y-6">
      {/* Banner / Header */}
      <div className="bg-gradient-to-r from-emerald-900 via-teal-900 to-slate-900 rounded-3xl p-6 sm:p-8 text-white relative overflow-hidden shadow-lg">
        <div className="relative z-10 max-w-2xl space-y-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold backdrop-blur-md border border-emerald-500/30">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Winter Test Partition Benchmark • Phase 8.5 Audited</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            AI-Driven Autonomous Energy Management System
          </h1>
          <p className="text-slate-300 text-sm leading-relaxed">
            General multi-asset energy optimization platform for renewable communities, microgrids, smart buildings, and commercial facilities. Demonstrated on a multi-building campus microgrid case study with GNN forecasting, Model Predictive Control, and robust risk management.
          </p>
          <div className="pt-2 flex flex-wrap items-center gap-3 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <Database className="w-3.5 h-3.5 text-emerald-400" />
              <span>Synthetic Simulation Dataset (1,313 Hours)</span>
            </span>
            <span>•</span>
            <span>500 kWh / 150 kW Battery</span>
            <span>•</span>
            <span>300 kW Solar PV</span>
            <span>•</span>
            <span>150 kW Wind Turbine</span>
          </div>
        </div>

        {/* Decorative Grid BG */}
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-10 bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:16px_16px] pointer-events-none" />
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Net Operating Cost"
          value={formatCurrency(kpis.best_causal_adjusted_cost_usd)}
          subtext={`Zero-Action: ${formatCurrency(kpis.baseline_operating_cost_usd)}`}
          icon={DollarSign}
          color="brand"
          tooltip="Inventory-adjusted operating cost over 1,313 test hours (winter regime)."
        />
        <StatCard
          title="Best Causal Savings"
          value={formatCurrency(kpis.net_savings_usd)}
          subtext={`+${kpis.savings_percentage}% vs Zero-Action`}
          icon={TrendingUp}
          trend={`+${formatCurrency(kpis.net_savings_usd)}`}
          trendPositive={true}
          color="blue"
          tooltip="Achieved by GNN-Transformer MPC without noncausal foresight."
        />
        <StatCard
          title="Campus Demand Served"
          value={`${testPart.campus_demand_mwh} MWh`}
          subtext="100% Demand Met (0 kWh unserved)"
          icon={Zap}
          color="amber"
          tooltip="Total building electricity demand across 5 campus facilities."
        />
        <StatCard
          title="Clean Generation"
          value={`${(testPart.solar_available_mwh + testPart.wind_available_mwh).toFixed(1)} MWh`}
          subtext={`Solar: ${testPart.solar_available_mwh} MWh • Wind: ${testPart.wind_available_mwh} MWh`}
          icon={Sun}
          color="purple"
          tooltip="Total local renewable generation available during test partition."
        />
      </div>

      {/* Grid: Live Energy Flow & Battery Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main 48-Hour Energy Profile Chart */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div>
              <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
                Microgrid Power Balance (Sample 48-Hour Trajectory)
              </h3>
              <p className="text-xs text-slate-400">
                Visualizing campus load, solar generation, wind power, and battery dispatch under GNN-MPC
              </p>
            </div>
            <NavLink
              to="/energy"
              className="inline-flex items-center gap-1 text-xs font-medium text-emerald-600 hover:text-emerald-700 dark:text-emerald-400"
            >
              <span>View Full 1,313h Telemetry</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </NavLink>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeseries}>
                <defs>
                  <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorSolar" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorGrid" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(val) => {
                    const d = new Date(val);
                    return `${d.getUTCHours()}:00`;
                  }}
                  stroke="#94a3b8"
                  fontSize={11}
                  tickLine={false}
                />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" kW" />
                <Tooltip
                  formatter={(val, name) => [`${val} kW`, name]}
                  labelFormatter={(lbl) => new Date(lbl).toLocaleString('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', hour: '2-digit' }) + ' UTC'}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                <Area type="monotone" dataKey="demand_kw" name="Campus Demand" stroke="#ef4444" fill="url(#colorDemand)" strokeWidth={2} />
                <Area type="monotone" dataKey="solar_kw" name="Solar PV" stroke="#f59e0b" fill="url(#colorSolar)" strokeWidth={2} />
                <Area type="monotone" dataKey="grid_import_kw" name="Grid Import" stroke="#8b5cf6" fill="url(#colorGrid)" strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Battery Widget & Quick Summary */}
        <div className="space-y-4">
          <BatteryGauge soc={0.67} powerKw={-49.9} capacityKwh={500} />

          <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-3">
            <h4 className="font-semibold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Leading Dispatch Strategy
            </h4>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-bold text-sm text-slate-900 dark:text-slate-100">
                  {kpis.best_causal_display_name}
                </div>
                <div className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                  Causal Neural Lookahead
                </div>
              </div>
              <ShieldCheck className="w-6 h-6 text-emerald-500" />
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800 text-xs">
              <div className="flex justify-between text-slate-600 dark:text-slate-400">
                <span>Oracle Gap Captured:</span>
                <span className="font-bold text-slate-900 dark:text-slate-100">{kpis.oracle_gap_captured_pct}%</span>
              </div>
              <div className="flex justify-between text-slate-600 dark:text-slate-400">
                <span>Advantage over Seasonal MPC:</span>
                <span className="font-bold text-emerald-600 dark:text-emerald-400">+$5.18 (0.014%)</span>
              </div>
              <div className="flex justify-between text-slate-600 dark:text-slate-400">
                <span>Battery Degradation Cost:</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200">$630.82 (31.5 FCE)</span>
              </div>
            </div>

            <NavLink
              to="/comparison"
              className="mt-2 w-full py-2 px-3 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-medium text-xs flex items-center justify-center gap-1.5 transition-colors"
            >
              <span>Explore All 7 Controllers</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </NavLink>
          </div>
        </div>
      </div>

      {/* Controller Performance Summary Bar Chart */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
              Controller Economic Savings on Held-Out Test Partition ($ vs Zero-Action)
            </h3>
            <p className="text-xs text-slate-400">
              Positive values indicate operating savings; negative values indicate net economic loss vs zero-action
            </p>
          </div>
          <NavLink
            to="/research"
            className="text-xs font-medium text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 inline-flex items-center gap-1"
          >
            <span>Read Scientific Audit</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </NavLink>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 10, right: 10, left: 10, bottom: 25 }}>
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} angle={-15} textAnchor="end" tickLine={false} />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" $" />
              <Tooltip formatter={(val) => [`${formatCurrency(val)}`, 'Net Savings vs Zero-Action']} />
              <Bar dataKey="savings" name="Net Savings ($)" radius={[4, 4, 0, 0]}>
                {barData.map((entry, index) => {
                  let fillColor = '#10b981'; // Green for positive savings
                  if (!entry.isDeployable) fillColor = '#8b5cf6'; // Purple for noncausal oracle
                  else if (entry.isGnn) fillColor = '#059669'; // Dark emerald for GNN
                  else if (entry.savings < 0) fillColor = '#f43f5e'; // Red for heuristics losing money
                  return <Cell key={`cell-${index}`} fill={fillColor} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-emerald-600" />
            <span>GNN-Transformer MPC (Best Causal: +$77.92)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-emerald-500" />
            <span>Seasonal Persistence MPC (+$72.74)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-rose-500" />
            <span>Heuristic Inversion (Price-Aware baselines lost -$15.87 and -$3.14)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-purple-500" />
            <span>Noncausal Diagnostic Oracle (+$295.75)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
