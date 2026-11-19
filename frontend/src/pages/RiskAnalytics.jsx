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
  Scale
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
  ResponsiveContainer
} from 'recharts';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import StatCard from '../components/StatCard';
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
        <div className="w-8 h-8 border-3 border-emerald-500 border-t-transparent rounded-full animate-spin" />
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
    key_findings = []
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
        title="Risk-Aware & Robust Optimization Analytics"
        subtitle="Exploring tail-risk hedging via Rockafellar-Uryasev CVaR (Phase 10) and distributionally robust optimization against model misspecification via L1 duality (Phase 11)."
        badge="Phase 10 & 11: Risk-Aware Control"
      />

      {/* Formulation Selection Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-800 gap-4">
        <button
          onClick={() => setActiveTab('cvar')}
          className={`pb-3 px-2 text-sm font-semibold flex items-center gap-2 border-b-2 transition-colors ${
            activeTab === 'cvar'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
              : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
          }`}
        >
          <ShieldAlert className="w-4 h-4" />
          <span>CVaR Risk-Aware MPC (Phase 10)</span>
        </button>
        <button
          onClick={() => setActiveTab('dro')}
          className={`pb-3 px-2 text-sm font-semibold flex items-center gap-2 border-b-2 transition-colors ${
            activeTab === 'dro'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
              : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
          }`}
        >
          <Scale className="w-4 h-4" />
          <span>DRO Robust MPC (Phase 11)</span>
        </button>
      </div>

      {activeTab === 'cvar' ? (
        <>
          {/* CVaR Hero / Stat Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              title="Hourly Peak Cost Reduction"
              value="-30.5%"
              subtitle="Max hourly surge capped ($164.95 → $114.71/h)"
              icon={ShieldCheck}
              trend="down"
            />
            <StatCard
              title="Adverse Stress Net Savings"
              value="$167.37"
              subtitle="Adjusted savings under compound stress"
              icon={TrendingDown}
              trend="up"
            />
            <StatCard
              title="Multi-Scenario Solve Latency"
              value="78.4 ms"
              subtitle="5 scenarios x 24h LP on CPU (HiGHS)"
              icon={Clock}
            />
            <StatCard
              title="Causal Nonanticipativity"
              value="u_s[0] = u_0[0]"
              subtitle="Strict single immediate physical dispatch"
              icon={Layers}
            />
          </div>

          {/* Scenario Condition Selector Tabs */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                  CVaR Stress Scenario Evaluation Matrix
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  Select an exogenous stress profile to observe controller dispatch resilience and tail risk containment.
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 dark:bg-slate-800/80 rounded-xl border border-slate-200 dark:border-slate-700/60">
                {conditions.map((cond) => {
                  const active = selectedCondition === cond.id;
                  return (
                    <button
                      key={cond.id}
                      onClick={() => setSelectedCondition(cond.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        active
                          ? 'bg-white dark:bg-slate-900 text-emerald-700 dark:text-emerald-300 shadow-xs font-semibold'
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
            <div className="p-3.5 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 rounded-xl flex items-start gap-3">
              <Info className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
              <div className="text-xs text-emerald-900 dark:text-emerald-200 space-y-0.5">
                <span className="font-semibold">{currentConditionMeta.name}:</span> {currentConditionMeta.description}
              </div>
            </div>

            {/* Comparative Performance Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                    <th className="py-2.5 px-3 font-semibold">Controller Formulation</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Adjusted Net Cost</th>
                    <th className="py-2.5 px-3 font-semibold text-right">95% CVaR ($/h)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Max Hourly Peak ($/h)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Battery FCE</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Avg Latency (ms)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                  {cvarTableRows.map((row) => {
                    const isCVaR = row.id.startsWith('mpc_cvar');
                    return (
                      <tr
                        key={row.id}
                        className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors ${
                          isCVaR ? 'bg-emerald-50/30 dark:bg-emerald-950/20 font-medium' : ''
                        }`}
                      >
                        <td className="py-3 px-3">
                          <div className="flex items-center gap-2">
                            {isCVaR ? (
                              <span className="p-1 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
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
                        <td className="py-3 px-3 text-right font-mono text-slate-900 dark:text-slate-100 font-semibold">
                          {formatCurrency(row.adjusted_net_cost_usd)}
                        </td>
                        <td className="py-3 px-3 text-right font-mono text-amber-600 dark:text-amber-400 font-semibold">
                          ${row.hourly_cost_cvar_95_usd?.toFixed(2)}/h
                        </td>
                        <td className="py-3 px-3 text-right font-mono text-rose-600 dark:text-rose-400">
                          ${row.worst_case_hourly_cost_usd?.toFixed(2)}/h
                        </td>
                        <td className="py-3 px-3 text-right font-mono text-slate-600 dark:text-slate-400">
                          {row.battery_fce?.toFixed(2)}
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

          {/* Visual Analytics Grid: Risk Comparison & Pareto Frontier */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Tail Risk Metrics Comparison ({currentConditionMeta.name})
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Comparing 95% CVaR and worst-case hourly cost peaks across control formulations.
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskComparisonData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.5} />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 10, fill: '#64748b' }}
                      angle={-15}
                      textAnchor="end"
                      interval={0}
                    />
                    <YAxis
                      unit="$/h"
                      tick={{ fontSize: 10, fill: '#64748b' }}
                    />
                    <Tooltip
                      formatter={(val) => [`$${Number(val).toFixed(2)}/h`, '']}
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="cvar95" name="95% CVaR ($/h)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="worstHour" name="Worst Hour Peak ($/h)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Risk-Aversion Pareto Trade-off Frontier (λ ∈ [0.0, 5.0])
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Compound adverse conditions: mapping total operating cost against tail risk posture.
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={frontierChartData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.5} />
                    <XAxis
                      dataKey="lambda"
                      tick={{ fontSize: 11, fill: '#64748b' }}
                    />
                    <YAxis
                      domain={['dataMin - 10', 'dataMax + 10']}
                      unit="$"
                      tick={{ fontSize: 10, fill: '#64748b' }}
                    />
                    <Tooltip
                      formatter={(val, name) => [
                        name === 'Adjusted Cost ($)' ? formatCurrency(val) : `$${Number(val).toFixed(2)}/h`,
                        name,
                      ]}
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line
                      type="monotone"
                      dataKey="adjustedCost"
                      name="Adjusted Cost ($)"
                      stroke="#10b981"
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
          {/* DRO Hero / Stat Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              title="Ambiguity Budget (ε)"
              value="ε = 0.5"
              subtitle="L1 Total-Variation probability ball"
              icon={Scale}
              color="brand"
            />
            <StatCard
              title="Nominal Recovery (ε = 0)"
              value="$5,219.65"
              subtitle="Matches nominal expected cost exactly"
              icon={CheckCircle2}
              color="blue"
            />
            <StatCard
              title="Robust Solve Latency"
              value="82.4 ms"
              subtitle="Exact LP dual with HiGHS on CPU"
              icon={Clock}
              color="amber"
            />
            <StatCard
              title="Dual Variable Dimension"
              value="2S + 1 Constraints"
              subtitle="Linear scaling in number of scenarios"
              icon={Layers}
              color="purple"
            />
          </div>

          {/* DRO Ambiguity Radius Sweep Section */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
            <div>
              <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <Compass className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                Distributionally Robust Ambiguity Sweep (ε ∈ [0.0, 2.0])
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Observing the price of robustness under compound stress as the adversary redistributes probability mass within the L1 ball.
              </p>
            </div>

            {/* DRO Sweep Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                    <th className="py-2.5 px-3 font-semibold">Ambiguity Radius (ε)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Adjusted Net Cost</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Worst Hour Peak ($/h)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Battery FCE</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Degradation Cost</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Solve Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                  {dro_ambiguity_sweep.map((row, idx) => (
                    <tr
                      key={idx}
                      className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors ${
                        row.ambiguity_radius_epsilon === 0.5 ? 'bg-emerald-50/40 dark:bg-emerald-950/20 font-semibold' : ''
                      }`}
                    >
                      <td className="py-3 px-3">
                        <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                          ε = {row.ambiguity_radius_epsilon.toFixed(2)}
                        </span>
                        {row.ambiguity_radius_epsilon === 0.0 && ' (Nominal)'}
                        {row.ambiguity_radius_epsilon === 0.5 && ' (Default Robust)'}
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-slate-900 dark:text-slate-100">
                        {formatCurrency(row.adjusted_net_cost_usd)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-rose-600 dark:text-rose-400">
                        ${row.worst_case_hourly_cost_usd?.toFixed(2)}/h
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-slate-600 dark:text-slate-400">
                        {row.battery_fce?.toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-slate-600 dark:text-slate-400">
                        {formatCurrency(row.battery_degradation_cost_usd)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono text-slate-500 dark:text-slate-400">
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
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Cost Impact vs Ambiguity Radius (ε)
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Evaluating the price of robustness across the total-variation ball size.
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={droSweepChartData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.5} />
                    <XAxis dataKey="epsilon" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <YAxis domain={['dataMin - 50', 'dataMax + 50']} unit="$" tick={{ fontSize: 10, fill: '#64748b' }} />
                    <Tooltip
                      formatter={(val, name) => [formatCurrency(val), name]}
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line type="monotone" dataKey="adjustedCost" name="Adjusted Cost ($)" stroke="#10b981" strokeWidth={2.5} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Worst-Case Cost Peak & Solve Latency
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Real-time CPU execution speed remains sub-100ms across all radii.
                </p>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={droSweepChartData} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" strokeOpacity={0.5} />
                    <XAxis dataKey="epsilon" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <YAxis unit="ms" tick={{ fontSize: 10, fill: '#64748b' }} />
                    <Tooltip
                      formatter={(val, name) => [`${Number(val).toFixed(1)} ms`, name]}
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
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
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
          <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            {activeTab === 'cvar' ? 'Mathematical Formulation (CVaR)' : 'Mathematical Formulation (DRO Dual)'}
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
            <div className="space-y-1.5 pt-1">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                <span>
                  <strong>Strict Nonanticipativity:</strong> The immediate physical decision{' '}
                  <code className="font-mono">u_s[0] = u_0[0]</code> is single-valued across all scenarios, ensuring dispatch is non-anticipative and executable before uncertainty unfolds.
                </span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                <span>
                  <strong>Conceptual Contrast:</strong> CVaR hedges against the tail of a <em>known</em> distribution. DRO protects against <em>distributional misspecification</em> where true scenario probabilities drift from nominal forecasts.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Empirical Findings Card */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-xs space-y-4">
          <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            Key Research Insights
          </h4>
          <div className="space-y-3">
            {key_findings.map((item, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800 space-y-1"
              >
                <div className="text-xs font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
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
