import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  TrendingDown,
  Clock,
  Layers,
  AlertTriangle,
  Info,
  CheckCircle2,
  Sliders,
  Sparkles,
  Compass,
  Scale,
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import MetricCard from '../components/MetricCard';
import { formatCurrency, formatNumber } from '../utils/formatters';

export default function RiskAnalytics() {
  const [riskData, setRiskData] = useState(null);
  const [activeTab, setActiveTab] = useState('cvar'); // 'cvar' | 'dro'
  const [selectedCondition, setSelectedCondition] = useState('compound_adverse');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadRisk() {
      try {
        const data = await api.getRiskResults();
        setRiskData(data);
      } catch (err) {
        console.error('Failed to load risk analytics data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadRisk();
  }, []);

  if (loading || !riskData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const {
    conditions,
    matrix = {},
    pareto_frontier = [],
    dro_matrix = {},
    dro_ambiguity_sweep = [],
    theoretical_foundation = {},
    key_findings = [],
  } = riskData;

  const currentConditionMeta = conditions.find((c) => c.id === selectedCondition) || {
    name: selectedCondition,
    description: '',
  };

  // CVaR condition data
  const currentConditionData = matrix[selectedCondition] || {};
  const cvarTableRows = Object.entries(currentConditionData).map(([id, item]) => ({
    id,
    name: item.name,
    ...item.metrics,
  }));

  // CVaR bar chart data
  const riskComparisonData = cvarTableRows.map((row) => ({
    name: row.name.replace(' Baseline', '').replace(' Deterministic', ''),
    cvar95: row.hourly_cost_cvar_95_usd,
    worstHour: row.worst_case_hourly_cost_usd,
    adjustedCost: row.adjusted_net_cost_usd,
  }));

  // CVaR Pareto frontier data
  const frontierChartData = pareto_frontier.map((pt) => ({
    lambda: `λ=${pt.risk_aversion_lambda}`,
    adjustedCost: pt.adjusted_net_cost_usd,
    cvar95: pt.hourly_cost_cvar_95_usd,
    worstHour: pt.worst_case_hourly_cost_usd,
    latencyMs: pt.mean_latency_ms,
  }));

  // DRO condition data
  const droConditionData = dro_matrix[selectedCondition] || {};
  const droTableRows = Object.entries(droConditionData).map(([id, item]) => ({
    id,
    name: item.name,
    ...item.metrics,
  }));

  // DRO ambiguity sweep chart data
  const droSweepChartData = dro_ambiguity_sweep.map((pt) => ({
    epsilon: `ε=${pt.ambiguity_radius_epsilon}`,
    adjustedCost: pt.adjusted_net_cost_usd,
    worstHour: pt.worst_case_hourly_cost_usd,
    fce: pt.battery_fce,
    latencyMs: pt.mean_latency_ms,
  }));

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Energy Risk & Reliability"
        subtitle="Explore protection against energy shortages and unexpected costs."
        badge="Simulation study"
      />

      {/* Formulation Selection Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-800 gap-4">
        <button
          onClick={() => setActiveTab('cvar')}
          className={`pb-3 px-3 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-colors ${
            activeTab === 'cvar'
              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
          }`}
        >
          <ShieldAlert className="w-4 h-4" />
          <span>High-Risk Cost Protection (CVaR-MPC)</span>
        </button>
        <button
          onClick={() => setActiveTab('dro')}
          className={`pb-3 px-3 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-colors ${
            activeTab === 'dro'
              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
          }`}
        >
          <Scale className="w-4 h-4" />
          <span>Uncertainty Protection (DRO-MPC)</span>
        </button>
      </div>

      {activeTab === 'cvar' ? (
        <>
          {/* CVaR Hero / Stat Cards using MetricCard with InfoTooltip */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Worst-Hour Peak Shaving"
              technicalTitle="Maximum Hourly Cost Reduction"
              value="-30.5%"
              unit=""
              subtext="Capped from $164.95 to $114.71/h"
              icon={ShieldCheck}
              color="emerald"
              trend="-30.5% Peak"
              trendPositive={true}
              tooltipData={{
                what: 'The percentage reduction in the single most expensive hour of electricity cost under compound stress.',
                why: 'Prevents budget-busting spikes when high building demand coincides with a sudden renewable dropout.',
                how: 'CVaR holds a reserve in the battery so it can discharge aggressively during the worst 5% of hours.',
                technical: 'Reduces max_t (Cost_t) from $164.95/h to $114.71/h under compound adverse conditions',
              }}
            />

            <MetricCard
              title="Stress Savings"
              technicalTitle="Compound Adverse Cost"
              value="$167.37"
              unit=""
              subtext="Operating savings under stress"
              icon={TrendingDown}
              color="blue"
              tooltipData={{
                what: 'Dollar savings delivered by CVaR-MPC compared to deterministic MPC when adverse weather hits.',
                why: 'Deterministic MPC assumes perfect forecasts and is caught flat-footed when weather turns bad.',
                how: 'CVaR proactively pre-charges before the storm, cushioning the microgrid.',
                technical: 'J_det ($5,387.02) - J_cvar ($5,219.65) = $167.37 net benefit under compound stress',
              }}
            />

            <MetricCard
              title="Solve Speed"
              technicalTitle="Multi-Scenario LP Latency"
              value="78.4 ms"
              unit=""
              subtext="5 scenarios x 24h LP on CPU"
              icon={Clock}
              color="violet"
              tooltipData={{
                what: 'Time required by the HiGHS solver to compute the optimal 24-hour battery schedule across all 5 scenarios.',
                why: 'Real-time microgrid controllers must solve in well under 1 second to dispatch reliably.',
                how: 'Rockafellar-Uryasev formulation linearizes the CVaR objective, making it extremely fast on standard CPUs.',
                technical: 'Mean solve latency over 1,313 steps: 78.4 ms using SciPy HiGHS simplex',
              }}
            />

            <MetricCard
              title="Causal Guarantee"
              technicalTitle="Physical Nonanticipativity"
              value="u_s[0] = u_0[0]"
              unit=""
              subtext="Strict single dispatch decision"
              icon={Layers}
              color="slate"
              tooltipData={{
                what: 'Mathematical guarantee that the battery action taken right now cannot peek at future scenario branches.',
                why: 'Without this constraint, the model would cheat by making different physical decisions before uncertainty resolves.',
                how: 'All 5 scenarios share the exact same first-step battery power variable u_0.',
                technical: 'u_s[0] == u_0[0] for all s in {1, ..., S}',
              }}
            />
          </div>

          {/* Scenario Condition Selector Tabs */}
          <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-slate-800/80">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span>Stress Scenario Evaluation Matrix</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Select an operational stress profile to inspect tail-risk containment
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 dark:bg-slate-800/80 rounded-xl border border-slate-200 dark:border-slate-700/60">
                {conditions.map((cond) => {
                  const active = selectedCondition === cond.id;
                  return (
                    <button
                      key={cond.id}
                      onClick={() => setSelectedCondition(cond.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        active
                          ? 'bg-white dark:bg-slate-900 text-blue-700 dark:text-blue-300 shadow-xs'
                          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                      }`}
                    >
                      {cond.name}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Selected Scenario Description Banner */}
            <div className="p-3.5 bg-blue-50/70 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800/60 rounded-2xl flex items-start gap-3">
              <Info className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
              <div className="text-xs text-blue-900 dark:text-blue-200 space-y-0.5">
                <span className="font-bold">{currentConditionMeta.name}:</span> {currentConditionMeta.description}
              </div>
            </div>

            {/* Comparative Performance Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-sans text-[11px] uppercase font-semibold">
                    <th className="py-2.5 px-3">Controller Formulation</th>
                    <th className="py-2.5 px-3 text-right">Adjusted Net Cost</th>
                    <th className="py-2.5 px-3 text-right">95% CVaR ($/h)</th>
                    <th className="py-2.5 px-3 text-right">Worst Hour Peak ($/h)</th>
                    <th className="py-2.5 px-3 text-right">Battery FCE</th>
                    <th className="py-2.5 px-3 text-right">Avg Latency (ms)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                  {cvarTableRows.map((row) => {
                    const isCVaR = row.id.startsWith('mpc_cvar');
                    return (
                      <tr
                        key={row.id}
                        className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors ${
                          isCVaR ? 'bg-blue-50/40 dark:bg-blue-950/20 font-semibold' : ''
                        }`}
                      >
                        <td className="py-3 px-3 font-sans">
                          <div className="flex items-center gap-2">
                            {isCVaR ? (
                              <span className="p-1 rounded-md bg-blue-500/10 text-blue-600 dark:text-blue-400">
                                <ShieldAlert className="w-3.5 h-3.5" />
                              </span>
                            ) : (
                              <span className="p-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-400">
                                <Zap className="w-3.5 h-3.5" />
                              </span>
                            )}
                            <span className="text-slate-900 dark:text-slate-100 font-medium">
                              {row.name}
                            </span>
                          </div>
                        </td>
                        <td className="py-3 px-3 text-right text-slate-900 dark:text-slate-100 font-bold">
                          {formatCurrency(row.adjusted_net_cost_usd)}
                        </td>
                        <td className="py-3 px-3 text-right text-amber-600 dark:text-amber-400 font-bold">
                          ${row.hourly_cost_cvar_95_usd?.toFixed(2)}/h
                        </td>
                        <td className="py-3 px-3 text-right text-rose-600 dark:text-rose-400 font-bold">
                          ${row.worst_case_hourly_cost_usd?.toFixed(2)}/h
                        </td>
                        <td className="py-3 px-3 text-right text-slate-600 dark:text-slate-400">
                          {row.battery_fce?.toFixed(2)}
                        </td>
                        <td className="py-3 px-3 text-right text-slate-500 dark:text-slate-400">
                          {row.mean_latency_ms?.toFixed(1)} ms
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visual Analytics Grid: Risk Comparison & Pareto Frontier */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Tail Risk Metrics Comparison ({currentConditionMeta.name})
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Comparing 95% CVaR and worst-case hourly cost peaks across control formulations
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskComparisonData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      angle={-15}
                      textAnchor="end"
                      interval={0}
                    />
                    <YAxis unit="$/h" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Tooltip
                      formatter={(val) => [`$${Number(val).toFixed(2)}/h`, '']}
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #334155',
                        borderRadius: '12px',
                        color: '#f8fafc',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="cvar95" name="95% CVaR ($/h)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="worstHour" name="Worst Hour Peak ($/h)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Risk-Aversion Pareto Trade-off Frontier (λ ∈ [0.0, 5.0])
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Compound adverse conditions: balancing expected cost against tail risk posture
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={frontierChartData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="lambda" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <YAxis domain={['dataMin - 10', 'dataMax + 10']} unit="$" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Tooltip
                      formatter={(val, name) => [
                        name === 'Adjusted Cost ($)' ? formatCurrency(val) : `$${Number(val).toFixed(2)}/h`,
                        name,
                      ]}
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #334155',
                        borderRadius: '12px',
                        color: '#f8fafc',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line
                      type="monotone"
                      dataKey="adjustedCost"
                      name="Adjusted Cost ($)"
                      stroke="#1051b9"
                      strokeWidth={2.5}
                      dot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="cvar95"
                      name="95% CVaR ($/h)"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      ) : (
        <>
          {/* DRO Hero / Stat Cards using MetricCard with InfoTooltip */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Ambiguity Budget"
              technicalTitle="L1 Total-Variation Ball Radius"
              value="ε = 0.5"
              unit=""
              subtext="Adversarial probability drift"
              icon={Scale}
              color="emerald"
              tooltipData={{
                what: 'The maximum allowable variation between the nominal weather forecast distribution and the worst-case adversary distribution.',
                why: 'Protects the microgrid against model misspecification when climate patterns shift unexpectedly.',
                how: 'ε = 0 recovers standard nominal expectation; larger ε introduces conservative hedging.',
                technical: 'Ambiguity set P = {q in Delta_S : ||q - p||_1 <= epsilon}',
              }}
            />

            <MetricCard
              title="Nominal Recovery"
              technicalTitle="Exact Expectation at ε = 0"
              value="$5,219.65"
              unit=""
              subtext="Identical to nominal stochastic MPC"
              icon={CheckCircle2}
              color="blue"
              tooltipData={{
                what: 'Theoretical proof that when the ambiguity radius is set to zero, DRO perfectly reproduces nominal stochastic MPC.',
                why: 'Confirms that the dual formulation does not introduce artificial distortion or numerical artifacts.',
                how: 'Verified within 1e-6 numerical tolerance against primal linear solver.',
                technical: 'lim_{epsilon -> 0} J_DRO(epsilon) == J_Nominal',
              }}
            />

            <MetricCard
              title="Robust Solve Speed"
              technicalTitle="Dual LP Solution Latency"
              value="82.4 ms"
              unit=""
              subtext="Exact LP dual with HiGHS on CPU"
              icon={Clock}
              color="violet"
              tooltipData={{
                what: 'Computation time to solve the distributionally robust optimization problem on a standard CPU.',
                why: 'The worst-case distribution over infinite probability vectors is collapsed into a single deterministic LP using L1 duality.',
                how: 'Solves in 82.4 milliseconds, comfortably inside the 1-hour dispatch horizon.',
                technical: 'Dual variable transformation introduces only 2S+1 linear constraints',
              }}
            />

            <MetricCard
              title="Dual Dimension"
              technicalTitle="Constraint Scaling Complexity"
              value="2S + 1 Constraints"
              unit=""
              subtext="Linear scaling with scenarios"
              icon={Layers}
              color="slate"
              tooltipData={{
                what: 'The number of auxiliary constraints added to the linear program to represent the adversary.',
                why: 'Linear scaling ensures fast execution even with many weather scenarios.',
                how: 'Avoids exponential curse of dimensionality common in robust optimization.',
                technical: 'Dualized with theta in R, gamma in R^S, mu in R_+',
              }}
            />
          </div>

          {/* DRO Ambiguity Radius Sweep Section */}
          <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
            <div className="pb-3 border-b border-slate-100 dark:border-slate-800/80">
              <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <Compass className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span>Distributionally Robust Ambiguity Sweep (ε ∈ [0.0, 2.0])</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Observing the price of robustness under compound stress as the adversary redistributes probability mass
              </p>
            </div>

            {/* DRO Sweep Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-sans text-[11px] uppercase font-semibold">
                    <th className="py-2.5 px-3">Ambiguity Radius (ε)</th>
                    <th className="py-2.5 px-3 text-right">Adjusted Net Cost</th>
                    <th className="py-2.5 px-3 text-right">Worst Hour Peak ($/h)</th>
                    <th className="py-2.5 px-3 text-right">Battery FCE</th>
                    <th className="py-2.5 px-3 text-right">Degradation Cost</th>
                    <th className="py-2.5 px-3 text-right">Solve Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                  {dro_ambiguity_sweep.map((row, idx) => (
                    <tr
                      key={idx}
                      className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors ${
                        row.ambiguity_radius_epsilon === 0.5
                          ? 'bg-blue-50/40 dark:bg-blue-950/20 font-semibold'
                          : ''
                      }`}
                    >
                      <td className="py-3 px-3">
                        <span className="font-mono text-blue-600 dark:text-blue-400 font-bold">
                          ε = {row.ambiguity_radius_epsilon.toFixed(2)}
                        </span>
                        {row.ambiguity_radius_epsilon === 0.0 && ' (Nominal Expected)'}
                        {row.ambiguity_radius_epsilon === 0.5 && ' (Default Robust)'}
                      </td>
                      <td className="py-3 px-3 text-right text-slate-900 dark:text-slate-100 font-bold">
                        {formatCurrency(row.adjusted_net_cost_usd)}
                      </td>
                      <td className="py-3 px-3 text-right text-rose-600 dark:text-rose-400 font-bold">
                        ${row.worst_case_hourly_cost_usd?.toFixed(2)}/h
                      </td>
                      <td className="py-3 px-3 text-right text-slate-600 dark:text-slate-400">
                        {row.battery_fce?.toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right text-slate-600 dark:text-slate-400">
                        {formatCurrency(row.battery_degradation_cost_usd)}
                      </td>
                      <td className="py-3 px-3 text-right text-slate-500 dark:text-slate-400">
                        {row.mean_latency_ms?.toFixed(1)} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visual Analytics: DRO Sweep Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Cost Impact vs Ambiguity Radius (ε)
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Evaluating the price of robustness across the total-variation ball size
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={droSweepChartData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="epsilon" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <YAxis domain={['dataMin - 50', 'dataMax + 50']} unit="$" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Tooltip
                      formatter={(val, name) => [formatCurrency(val), name]}
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #334155',
                        borderRadius: '12px',
                        color: '#f8fafc',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line
                      type="monotone"
                      dataKey="adjustedCost"
                      name="Adjusted Cost ($)"
                      stroke="#1051b9"
                      strokeWidth={2.5}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Worst-Case Cost Peak & Solve Latency
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Real-time CPU execution speed remains sub-100ms across all radii
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={droSweepChartData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="epsilon" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <YAxis unit="ms" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Tooltip
                      formatter={(val, name) => [`${Number(val).toFixed(1)} ms`, name]}
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #334155',
                        borderRadius: '12px',
                        color: '#f8fafc',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey="latencyMs" name="Solve Latency (ms)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Theoretical Foundation & Mathematical Contrast */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
          <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2 pb-2 border-b border-slate-100 dark:border-slate-800/80">
            <Sparkles className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <span>{activeTab === 'cvar' ? 'Mathematical Formulation (CVaR)' : 'Mathematical Formulation (DRO Dual)'}</span>
          </h4>
          <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
            {activeTab === 'cvar' ? (
              <>
                <p>
                  Converted into an exact linear program using Rockafellar-Uryasev (2000) equivalence:
                </p>
                <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl font-mono text-[11px] text-slate-800 dark:text-slate-200 overflow-x-auto">
                  min E[Cost(x, ξ)] + λ · [ η + (1 / (1 - α)) · ∑ p_s · z_s ]
                  <br />
                  s.t. Cost_s(x) - η ≤ z_s, &nbsp; z_s ≥ 0 &nbsp; ∀ s ∈ S
                </div>
              </>
            ) : (
              <>
                <p>
                  Formulated over an L1 total-variation probability ambiguity ball <code className="font-mono">||q - p||_1 ≤ ε</code> and dualized into a single joint linear program:
                </p>
                <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl font-mono text-[11px] text-slate-800 dark:text-slate-200 overflow-x-auto">
                  min θ + ∑ p_s · γ_s + ε · μ
                  <br />
                  s.t. θ + γ_s ≥ L_s(u), &nbsp; -μ ≤ γ_s ≤ μ, &nbsp; μ ≥ 0 &nbsp; ∀ s ∈ S
                </div>
              </>
            )}
            <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800/60">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                <span>
                  <strong>Strict Nonanticipativity:</strong> The immediate physical decision{' '}
                  <code className="font-mono bg-slate-100 dark:bg-slate-800 px-1 py-0.5 rounded">u_s[0] = u_0[0]</code> is single-valued across all scenarios, ensuring dispatch is non-anticipative and executable before uncertainty unfolds.
                </span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-blue-500 shrink-0 mt-0.5" />
                <span>
                  <strong>Conceptual Contrast:</strong> CVaR hedges against the tail of a <em>known</em> distribution. DRO protects against <em>distributional misspecification</em> where true scenario probabilities drift from nominal forecasts.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Empirical Findings Card */}
        <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
          <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2 pb-2 border-b border-slate-100 dark:border-slate-800/80">
            <CheckCircle2 className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <span>Key Research Insights</span>
          </h4>
          <div className="space-y-3">
            {key_findings.map((item, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800 space-y-1"
              >
                <div className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                  {item.title}
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  {item.detail}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
