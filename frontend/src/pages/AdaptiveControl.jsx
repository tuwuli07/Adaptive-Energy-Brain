import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Cpu,
  Layers,
  Zap,
  Clock,
  ShieldCheck,
  CheckCircle2,
  Info,
  Sun,
  AlertTriangle,
  TrendingDown,
  Sparkles,
  ArrowRight,
  Battery
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import StatCard from '../components/StatCard';
import { formatCurrency, formatNumber } from '../utils/formatters';

const REGIME_COLORS = {
  SOLAR_SELF_CONSUMPTION: '#10b981', // emerald
  PEAK_TARIFF_TAIL_HEDGING: '#f59e0b', // amber
  COMPOUND_STRESS_ROBUST: '#8b5cf6', // purple
  NOMINAL_BASE_DISPATCH: '#3b82f6', // blue
};

const CONTROLLER_BADGES = {
  mpc_gnn_transformer: { name: 'GNN-Transformer MPC', color: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20' },
  mpc_cvar_balanced: { name: 'CVaR Risk-Aware MPC', color: 'bg-amber-500/10 text-amber-600 border-amber-500/20' },
  mpc_dro_robust: { name: 'DRO Robust MPC', color: 'bg-purple-500/10 text-purple-600 border-purple-500/20' },
  mpc_seasonal_persistence: { name: 'Seasonal Persistence MPC', color: 'bg-blue-500/10 text-blue-600 border-blue-500/20' },
};

export default function AdaptiveControl() {
  const [adaptiveData, setAdaptiveData] = useState(null);
  const [trajectory, setTrajectory] = useState([]);
  const [selectedStep, setSelectedStep] = useState(0);
  const [selectedCondition, setSelectedCondition] = useState('compound_adverse');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [results, traj] = await Promise.all([
          api.getAdaptiveResults(),
          api.getAdaptiveTrajectory(),
        ]);
        setAdaptiveData(results);
        setTrajectory(traj || []);
        if (traj && traj.length > 0) {
          setSelectedStep(Math.min(2, traj.length - 1)); // default to interesting step
        }
      } catch (err) {
        console.error('Failed to load adaptive control data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading || !adaptiveData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="w-8 h-8 border-3 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const { regimes = [], matrix = {}, candidate_controllers = [], key_insights = [] } = adaptiveData;
  const currentStepData = trajectory[selectedStep] || trajectory[0] || {};
  const activeConditionData = matrix[selectedCondition] || {};

  // Table rows for active condition
  const tableRows = Object.entries(activeConditionData).map(([id, item]) => ({
    id,
    name: item.name,
    ...item.metrics,
  }));

  // Regime distribution pie data for active condition
  const adaptiveMetrics = activeConditionData.adaptive_policy_selector?.metrics || {};
  const regimeDist = adaptiveMetrics.regime_distribution || {};
  const pieData = Object.entries(regimeDist).map(([regime, pct]) => ({
    name: regime.replace(/_/g, ' '),
    value: pct,
    color: REGIME_COLORS[regime] || '#64748b',
  }));

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Adaptive Policy Selection & Dynamic Regime Routing"
        subtitle="Context-aware control router that dynamically dispatches execution among specialized MPC paradigms (GNN-Transformer, CVaR, DRO, Seasonal) based on causal observable conditions at decision time."
        badge="Phase 12: Contextual Adaptive Control"
      />

      {/* Scientific Accuracy & Data Provenance Notice */}
      <div className="p-4 bg-slate-100 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 rounded-2xl text-xs space-y-1">
        <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
          <Info className="w-4 h-4 text-emerald-500 shrink-0" />
          <span>Scientific Classification & Provenance</span>
        </div>
        <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
          <strong>Adaptive Policy Selection</strong> is implemented via causal contextual regime routing calibrated on development validation data.
          This is distinct from true gradient-based meta-learning across task distributions (e.g., MAML or RL²), which is identified as future work.
          All evaluations are conducted on synthetic simulation datasets (72-hour development slice), not physical hardware measurements.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Active Candidate Models"
          value="4 Models"
          subtitle="GNN, CVaR, DRO & Seasonal MPC"
          icon={Layers}
          color="brand"
        />
        <StatCard
          title="Switching Latency"
          value="~65 ms"
          subtitle="Feature extraction + route + solve"
          icon={Clock}
          color="blue"
        />
        <StatCard
          title="Physical Safety Record"
          value="0 Violations"
          subtitle="150 kW inverter & SOC bounds strictly held"
          icon={ShieldCheck}
          color="emerald"
        />
        <StatCard
          title="Routing Method"
          value="Contextual Rule"
          subtitle="Strictly non-anticipative causal logic"
          icon={Sliders}
          color="purple"
        />
      </div>

      {/* Interactive Trajectory Inspection Section */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-emerald-500" />
              Hourly Policy Switching Timeline (Compound Adverse Trajectory)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Scrub the timeline to observe how the Adaptive Selector monitors observable features and routes dispatch to the optimal paradigm.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-500">Step {selectedStep + 1} / {trajectory.length}</span>
          </div>
        </div>

        {/* Step Slider */}
        <div className="space-y-1">
          <input
            type="range"
            min="0"
            max={trajectory.length > 0 ? trajectory.length - 1 : 0}
            value={selectedStep}
            onChange={(e) => setSelectedStep(Number(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer h-2 bg-slate-100 dark:bg-slate-800 rounded-lg"
          />
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>Hour 0 (09-13 13:00 UTC)</span>
            <span>Hour 24</span>
            <span>Hour 48</span>
            <span>Hour 71 (09-16 11:00 UTC)</span>
          </div>
        </div>

        {/* Selected Hour Telemetry & Rationale Card */}
        {currentStepData && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            {/* Observable State Box */}
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 space-y-2 text-xs">
              <div className="font-semibold text-slate-500 uppercase tracking-wider text-[10px]">
                Causal State (Step {currentStepData.step})
              </div>
              <div className="space-y-1.5 font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-500">Timestamp:</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-200">{currentStepData.timestamp_utc}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Total Demand:</span>
                  <span className="font-bold text-slate-800 dark:text-slate-200">{currentStepData.demand_kw} kW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Solar / Wind:</span>
                  <span className="text-emerald-600 dark:text-emerald-400">{currentStepData.solar_kw} / {currentStepData.wind_kw} kW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Import Price:</span>
                  <span className={`font-bold ${currentStepData.import_price >= 0.25 ? 'text-rose-600' : 'text-slate-800 dark:text-slate-200'}`}>
                    ${Number(currentStepData.import_price).toFixed(2)}/kWh
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Battery SOC:</span>
                  <span className="font-bold text-slate-800 dark:text-slate-200">{(currentStepData.battery_soc_after * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Selected Controller & Regime Box */}
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 space-y-3 md:col-span-2 text-xs">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-500 uppercase tracking-wider text-[10px]">
                  Active Decision & Rationale
                </div>
                <span
                  className="px-2.5 py-0.5 rounded-full text-[11px] font-bold"
                  style={{
                    backgroundColor: `${REGIME_COLORS[currentStepData.regime_name] || '#64748b'}20`,
                    color: REGIME_COLORS[currentStepData.regime_name] || '#64748b',
                  }}
                >
                  {currentStepData.regime_name}
                </span>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">Selected Controller:</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                    {CONTROLLER_BADGES[currentStepData.selected_controller_id]?.name || currentStepData.selected_controller_id}
                  </span>
                </div>
                <p className="text-slate-600 dark:text-slate-300 leading-relaxed bg-white dark:bg-slate-900/60 p-2.5 rounded-lg border border-slate-200/60 dark:border-slate-700/60">
                  {currentStepData.rationale}
                </p>
                <div className="flex items-center gap-4 pt-1 font-mono text-[11px]">
                  <span className="text-slate-500">Battery Dispatch: <strong className="text-slate-800 dark:text-slate-200">{currentStepData.battery_actual_kw} kW</strong></span>
                  <span className="text-slate-500">Grid Import: <strong className="text-slate-800 dark:text-slate-200">{currentStepData.grid_import_kw} kW</strong></span>
                  <span className="text-slate-500">Net Cost: <strong className="text-slate-800 dark:text-slate-200">${currentStepData.net_operating_cost}</strong></span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 72-Hour Composed Area & Line Chart */}
        <div className="h-64 pt-3">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={trajectory} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.5} />
              <XAxis dataKey="step" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={(s) => `H${s}`} />
              <YAxis yAxisId="left" unit=" kW" tick={{ fontSize: 10, fill: '#64748b' }} />
              <YAxis yAxisId="right" orientation="right" unit=" $" tick={{ fontSize: 10, fill: '#64748b' }} />
              <Tooltip
                formatter={(val, name) => [typeof val === 'number' ? val.toFixed(1) : val, name]}
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc', fontSize: '11px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar yAxisId="left" dataKey="demand_kw" name="Total Demand (kW)" fill="#ef4444" opacity={0.25} />
              <Bar yAxisId="left" dataKey="solar_kw" name="Solar Power (kW)" fill="#10b981" opacity={0.4} />
              <Line yAxisId="left" type="stepAfter" dataKey="battery_actual_kw" name="Battery Power (kW)" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="stepAfter" dataKey="import_price" name="Import Price ($/kWh)" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Operational Regimes Architecture Grid */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-500" />
          Specialized Control Regimes Architecture
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {regimes.map((reg) => (
            <div
              key={reg.id}
              className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40 space-y-2 text-xs"
            >
              <div className="flex items-center gap-2 font-bold" style={{ color: REGIME_COLORS[reg.id] || '#64748b' }}>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: REGIME_COLORS[reg.id] || '#64748b' }} />
                <span>{reg.name}</span>
              </div>
              <div className="text-[11px] font-mono text-slate-500">
                Active: <strong>{CONTROLLER_BADGES[reg.active_controller]?.name || reg.active_controller}</strong>
              </div>
              <div className="p-2 rounded-lg bg-white dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/60 text-slate-600 dark:text-slate-400 text-[11px]">
                <strong>Trigger:</strong> {reg.trigger_condition}
              </div>
              <p className="text-slate-500 dark:text-slate-400 text-[11px] leading-relaxed">
                {reg.rationale}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Comparative Evaluation Section */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-500" />
              Development Benchmark: Adaptive Policy vs. Fixed Baselines
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Evaluating the Adaptive Policy Selector alongside fixed single-model deployments across 5 stress scenarios.
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 dark:bg-slate-800/80 rounded-xl border border-slate-200 dark:border-slate-700/60">
            {['normal', 'high_demand', 'renewable_scarcity', 'price_spikes', 'compound_adverse'].map((cond) => (
              <button
                key={cond}
                onClick={() => setSelectedCondition(cond)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedCondition === cond
                    ? 'bg-white dark:bg-slate-900 text-emerald-700 dark:text-emerald-300 shadow-xs font-semibold'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {cond.replace(/_/g, ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Results Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                <th className="py-2.5 px-3 font-semibold">Controller Paradigm</th>
                <th className="py-2.5 px-3 font-semibold text-right">Adjusted Net Cost</th>
                <th className="py-2.5 px-3 font-semibold text-right">Worst-Case Peak ($/h)</th>
                <th className="py-2.5 px-3 font-semibold text-right">95% CVaR ($/h)</th>
                <th className="py-2.5 px-3 font-semibold text-right">Battery FCE</th>
                <th className="py-2.5 px-3 font-semibold text-right">Avg Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
              {tableRows.map((row) => {
                const isAdaptive = row.id === 'adaptive_policy_selector';
                return (
                  <tr
                    key={row.id}
                    className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors ${
                      isAdaptive ? 'bg-emerald-50/40 dark:bg-emerald-950/20 font-bold' : ''
                    }`}
                  >
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        {isAdaptive ? (
                          <span className="p-1 rounded-md bg-emerald-500/20 text-emerald-600 dark:text-emerald-400">
                            <Sliders className="w-3.5 h-3.5" />
                          </span>
                        ) : (
                          <span className="p-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-400">
                            <Cpu className="w-3.5 h-3.5" />
                          </span>
                        )}
                        <span className="text-slate-900 dark:text-slate-100">
                          {row.name}
                        </span>
                        {isAdaptive && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500 text-white font-semibold">
                            Adaptive
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-900 dark:text-slate-100">
                      {formatCurrency(row.adjusted_net_cost_usd)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-rose-600 dark:text-rose-400">
                      ${row.worst_case_hourly_cost_usd?.toFixed(2)}/h
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-amber-600 dark:text-amber-400">
                      ${row.hourly_cost_cvar_95_usd?.toFixed(2)}/h
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-600 dark:text-slate-400">
                      {row.battery_fce?.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-500 dark:text-slate-400">
                      {row.mean_latency_ms?.toFixed(1)} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key Insights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {key_insights.map((item, idx) => (
          <div
            key={idx}
            className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 shadow-xs space-y-1.5"
          >
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              <span>{item.title}</span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              {item.detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
