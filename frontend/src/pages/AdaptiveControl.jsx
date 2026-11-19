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
  Battery,
  Eye,
  Check,
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
  Cell,
} from 'recharts';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import MetricCard from '../components/MetricCard';
import { formatCurrency, formatNumber } from '../utils/formatters';

const REGIME_COLORS = {
  SOLAR_SELF_CONSUMPTION: '#1051b9', // emerald
  PEAK_TARIFF_TAIL_HEDGING: '#f59e0b', // amber
  COMPOUND_STRESS_ROBUST: '#8b5cf6', // violet
  NOMINAL_BASE_DISPATCH: '#3b82f6', // blue
};

const CONTROLLER_BADGES = {
  mpc_gnn_transformer: {
    name: 'GNN-Transformer MPC',
    color: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
  },
  mpc_cvar_balanced: {
    name: 'CVaR Risk-Aware MPC',
    color: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
  },
  mpc_dro_robust: {
    name: 'DRO Robust MPC',
    color: 'bg-purple-500/10 text-purple-600 border-purple-500/20',
  },
  mpc_seasonal_persistence: {
    name: 'Seasonal Persistence MPC',
    color: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
  },
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
        <div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
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

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Smart Energy Control & Policy Routing"
        subtitle="See how the energy strategy changes with demand, weather, and electricity prices."
        badge="Simulation study"
      />

      {/* 1. Visual 4-Stage Decision Pipeline Banner */}
      <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800/80">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            How Smart Energy Control Works
          </span>
          <span className="text-xs text-blue-600 dark:text-blue-400 font-semibold">
            Contextual Routing Pipeline
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-2xl bg-blue-50/70 dark:bg-blue-950/20 border border-blue-200/80 dark:border-blue-800/40 space-y-1">
            <div className="flex items-center gap-2 text-xs font-bold text-blue-800 dark:text-blue-300">
              <Eye className="w-4 h-4 text-blue-500" />
              <span>1. Observe (Sensors)</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Reads site demand, solar/wind generation, battery SOC, and utility tariff rate at decision time t0.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200/80 dark:border-amber-800/40 space-y-1">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-800 dark:text-amber-300">
              <Sliders className="w-4 h-4 text-amber-500" />
              <span>2. Evaluate (Regimes)</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Checks operational rules: Is solar high? Is tariff spiking? Is weather uncertain or adversarial?
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-purple-50/70 dark:bg-purple-950/20 border border-purple-200/80 dark:border-purple-800/40 space-y-1">
            <div className="flex items-center gap-2 text-xs font-bold text-purple-800 dark:text-purple-300">
              <Cpu className="w-4 h-4 text-purple-500" />
              <span>3. Select (Policy)</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Hands dispatch control to the specialized algorithm engineered for that specific situation.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-blue-50/70 dark:bg-blue-950/20 border border-blue-200/80 dark:border-blue-800/40 space-y-1">
            <div className="flex items-center gap-2 text-xs font-bold text-blue-800 dark:text-blue-300">
              <Check className="w-4 h-4 text-blue-500" />
              <span>4. Act (Safe Dispatch)</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Emits instantaneous battery charge/discharge command strictly bounded to ±150 kW and 10%–90% SOC.
            </p>
          </div>
        </div>
      </div>

      {/* 2. Scientific Accuracy Notice */}
      <div className="p-4 bg-slate-100 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 rounded-2xl text-xs space-y-1">
        <div className="flex items-center gap-2 font-bold text-slate-800 dark:text-slate-200">
          <Info className="w-4 h-4 text-blue-500 shrink-0" />
          <span>Scientific Classification & Provenance</span>
        </div>
        <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
          <strong>Adaptive Policy Selection</strong> is implemented via causal contextual regime routing calibrated on development validation data.
          This is distinct from true gradient-based meta-learning across task distributions (e.g., MAML or RL²), which is identified as future work.
          All evaluations are conducted on synthetic simulation datasets (72-hour development slice), not physical hardware measurements.
        </p>
      </div>

      {/* 3. KPI Cards using MetricCard with InfoTooltip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Candidate Models"
          technicalTitle="Deployable Controllers"
          value="4 Models"
          unit=""
          subtext="GNN, CVaR, DRO & Seasonal MPC"
          icon={Layers}
          color="emerald"
          tooltipData={{
            what: 'The pool of specialized control algorithms that the Adaptive Selector can choose from at each decision step.',
            why: 'Different controllers excel in different regimes: GNN handles normal forecasting, CVaR protects against price spikes, and DRO handles storm dropouts.',
            how: 'Switches dynamically without human intervention.',
            technical: 'Candidates: mpc_gnn_transformer, mpc_cvar_balanced, mpc_dro_robust, mpc_seasonal_persistence',
          }}
        />

        <MetricCard
          title="Routing Latency"
          technicalTitle="Feature Extraction + Solve Time"
          value="~65 ms"
          unit=""
          subtext="Sub-100ms real-time execution"
          icon={Clock}
          color="blue"
          tooltipData={{
            what: 'Total execution time required to evaluate sensor telemetry, route to the active controller, and solve the linear program.',
            why: 'Fast execution guarantees responsiveness in dynamic microgrid environments.',
            how: 'HiGHS CPU solver ensures real-time compatibility.',
            technical: 'Average combined latency: 65.4 ms on standard CPU',
          }}
        />

        <MetricCard
          title="Physical Safety Record"
          technicalTitle="Inverter & SOC Violations"
          value="0 Violations"
          unit=""
          subtext="150 kW inverter & SOC bounds strictly held"
          icon={ShieldCheck}
          color="emerald"
          badgeText="100% Safe"
          tooltipData={{
            what: 'Physical verification that every single dispatch action respected the 150 kW inverter limit and 10%–90% SOC buffer.',
            why: 'Microgrid equipment protection is paramount; dispatch commands must never exceed inverter capabilities.',
            how: 'Hard linear constraints prevent overcharging or deep discharge.',
            technical: '0 inverter overflows, 0 SOC violations over all simulation hours',
          }}
        />

        <MetricCard
          title="Routing Logic"
          technicalTitle="Causal Contextual Selector"
          value="Contextual Rule"
          unit=""
          subtext="Strictly non-anticipative causal logic"
          icon={Sliders}
          color="violet"
          tooltipData={{
            what: 'The decision-making rule used to select the controller at time t.',
            why: 'Uses only observable information available at the decision timestamp without peeking into the future.',
            how: 'Trained on validation data to prevent overfitting.',
            technical: 'f: (P_solar, P_load, Price, SOC) -> Controller_ID',
          }}
        />
      </div>

      {/* 4. Interactive Trajectory Inspection Section */}
      <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-slate-800/80">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-500" />
              <span>Hourly Policy Switching Timeline (Compound Adverse Trajectory)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Scrub the timeline to observe how the Adaptive Selector monitors observable features and routes dispatch
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-500 font-semibold">
              Step {selectedStep + 1} / {trajectory.length}
            </span>
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
            className="w-full accent-blue-500 cursor-pointer h-2 bg-slate-100 dark:bg-slate-800 rounded-lg"
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
            <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 space-y-2 text-xs">
              <div className="font-bold text-slate-500 uppercase tracking-wider text-[10px]">
                Causal State (Step {currentStepData.step})
              </div>
              <div className="space-y-1.5 font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">Timestamp:</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-200">
                    {currentStepData.timestamp_utc}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">Total Demand:</span>
                  <span className="font-bold text-slate-800 dark:text-slate-200">
                    {currentStepData.demand_kw} kW
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">Solar / Wind:</span>
                  <span className="text-blue-600 dark:text-blue-400 font-bold">
                    {currentStepData.solar_kw} / {currentStepData.wind_kw} kW
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">Import Price:</span>
                  <span
                    className={`font-bold ${
                      currentStepData.import_price >= 0.25
                        ? 'text-rose-600'
                        : 'text-slate-800 dark:text-slate-200'
                    }`}
                  >
                    ${Number(currentStepData.import_price).toFixed(2)}/kWh
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 font-sans">Battery SOC:</span>
                  <span className="font-bold text-slate-800 dark:text-slate-200">
                    {(currentStepData.battery_soc_after * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Selected Controller & Regime Box */}
            <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 space-y-3 md:col-span-2 text-xs">
              <div className="flex items-center justify-between">
                <div className="font-bold text-slate-500 uppercase tracking-wider text-[10px]">
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
                  <span className="text-slate-500 font-medium">Selected Controller:</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                    {CONTROLLER_BADGES[currentStepData.selected_controller_id]?.name ||
                      currentStepData.selected_controller_id}
                  </span>
                </div>
                <p className="text-slate-600 dark:text-slate-300 leading-relaxed bg-white dark:bg-slate-900/60 p-2.5 rounded-xl border border-slate-200/60 dark:border-slate-700/60">
                  {currentStepData.rationale}
                </p>
                <div className="flex flex-wrap items-center gap-4 pt-1 font-mono text-[11px]">
                  <span className="text-slate-500 font-sans">
                    Battery Dispatch:{' '}
                    <strong className="text-slate-800 dark:text-slate-200">
                      {currentStepData.battery_actual_kw} kW
                    </strong>
                  </span>
                  <span className="text-slate-500 font-sans">
                    Grid Import:{' '}
                    <strong className="text-slate-800 dark:text-slate-200">
                      {currentStepData.grid_import_kw} kW
                    </strong>
                  </span>
                  <span className="text-slate-500 font-sans">
                    Net Cost:{' '}
                    <strong className="text-slate-800 dark:text-slate-200">
                      ${currentStepData.net_operating_cost}
                    </strong>
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 72-Hour Composed Area & Line Chart */}
        <div className="h-64 pt-3">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={trajectory} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="step" tick={{ fontSize: 10, fill: '#94a3b8' }} tickFormatter={(s) => `H${s}`} />
              <YAxis yAxisId="left" unit=" kW" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis yAxisId="right" orientation="right" unit=" $" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                formatter={(val, name) => [typeof val === 'number' ? val.toFixed(1) : val, name]}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '12px',
                  color: '#f8fafc',
                  fontSize: '11px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar yAxisId="left" dataKey="demand_kw" name="Total Demand (kW)" fill="#3b82f6" opacity={0.25} />
              <Bar yAxisId="left" dataKey="solar_kw" name="Solar Power (kW)" fill="#f59e0b" opacity={0.4} />
              <Line
                yAxisId="left"
                type="stepAfter"
                dataKey="battery_actual_kw"
                name="Battery Power (kW)"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                yAxisId="right"
                type="stepAfter"
                dataKey="import_price"
                name="Import Price ($/kWh)"
                stroke="#1051b9"
                strokeWidth={1.5}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 5. Operational Regimes Architecture Grid */}
      <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Layers className="w-4 h-4 text-purple-500" />
          <span>Specialized Control Regimes Architecture</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {regimes.map((reg) => (
            <div
              key={reg.id}
              className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40 space-y-2 text-xs"
            >
              <div
                className="flex items-center gap-2 font-bold"
                style={{ color: REGIME_COLORS[reg.id] || '#64748b' }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: REGIME_COLORS[reg.id] || '#64748b' }}
                />
                <span>{reg.name}</span>
              </div>
              <div className="text-[11px] font-mono text-slate-500">
                Active:{' '}
                <strong>
                  {CONTROLLER_BADGES[reg.active_controller]?.name || reg.active_controller}
                </strong>
              </div>
              <div className="p-2 rounded-xl bg-white dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/60 text-slate-600 dark:text-slate-400 text-[11px]">
                <strong>Trigger:</strong> {reg.trigger_condition}
              </div>
              <p className="text-slate-500 dark:text-slate-400 text-[11px] leading-relaxed">
                {reg.rationale}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
