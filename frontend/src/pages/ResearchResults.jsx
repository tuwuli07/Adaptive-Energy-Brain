import { Link } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import {
  FileText,
  CheckCircle2,
  AlertTriangle,
  Scale,
  Cpu,
  ShieldCheck,
  TrendingUp,
  Clock,
  Sparkles,
  Info
} from 'lucide-react';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import StatCard from '../components/StatCard';
import { formatCurrency, formatPercent } from '../utils/formatters';

export default function ResearchResults() {
  const [researchData, setResearchData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadResults() {
      try {
        const res = await api.getResearchResults();
        setResearchData(res);
      } catch (err) {
        console.error('Failed to load research results:', err);
      } finally {
        setLoading(false);
      }
    }
    loadResults();
  }, []);

  if (loading || !researchData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <nav aria-label="Study tools" className="study-tools surface">
        <div><span className="eyebrow">STUDY & METHODS</span><p>Explore the evidence behind the energy insights.</p></div>
        <div className="flex flex-wrap gap-2">
          <Link className="button-secondary" to="/comparison">Compare strategies</Link>
          <Link className="button-secondary" to="/risk">Risk & reliability</Link>
          <Link className="button-secondary" to="/explorer">Scenario explorer</Link>
          <Link className="button-secondary" to="/about">About the project</Link>
        </div>
      </nav>
      <SectionHeader
        title="Scientific Research Findings & Post-Evaluation Audit"
        subtitle="Explore the findings and limitations of the simulated campus study."
        badge="Simulation study"
      />

      {/* Audit Banner */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-xl bg-blue-500 text-white shrink-0">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm text-blue-900 dark:text-blue-200">
                Independent Post-Evaluation Audit Verdict: PASS
              </span>
              <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 text-[10px] font-bold">
                24/24 SHA-256 Verified
              </span>
            </div>
            <p className="text-xs text-blue-700 dark:text-blue-400 leading-relaxed">
              All 1,313-step hourly trajectories independently recomputed within floating-point tolerance ($&lt;10^{'{ -10 }' }$). Zero data leakage, zero retries, zero unhandled solver fallbacks, and zero physical violations.
            </p>
          </div>
        </div>

        <div className="text-right shrink-0">
          <span className="text-xs text-slate-500 dark:text-slate-400 block">Evaluated Partition:</span>
          <span className="text-xs font-bold text-slate-800 dark:text-slate-200">Winter Test (1,313 Hours)</span>
        </div>
      </div>

      {/* Primary Research Quantitative Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="GNN-MPC Adjusted Cost"
          value={formatCurrency(36532.73)}
          subtext="Lowest causal cost on test"
          icon={CheckCircle2}
          color="brand"
          tooltip="Includes $12.53 terminal inventory credit for ending at SOC 0.67."
        />
        <StatCard
          title="Seasonal MPC Adjusted Cost"
          value={formatCurrency(36537.91)}
          subtext="Benchmark persistence MPC"
          icon={Clock}
          color="blue"
          tooltip="24h seasonal persistence point forecast into identical HiGHS LP solver."
        />
        <StatCard
          title="GNN-MPC Advantage"
          value={formatCurrency(5.18)}
          subtext="0.0142% of total Seasonal cost"
          icon={TrendingUp}
          color="purple"
          tooltip="Net operating savings difference between GNN-MPC and Seasonal-MPC."
        />
        <StatCard
          title="Savings vs Zero-Action"
          value={formatCurrency(77.92)}
          subtext="+0.21% relative savings"
          icon={Scale}
          color="amber"
          tooltip="Net economic savings achieved compared to idling the battery."
        />
      </div>

      {/* In-Depth Scientific Explanations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Finding 1: The $5.18 Marginal Advantage & Throughput Identity */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
              GNN-MPC vs Seasonal MPC: The $5.18 Mechanism
            </h3>
          </div>

          <div className="space-y-2 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            <p>
              GNN-Transformer MPC achieved an inventory-adjusted operating cost of <strong>$36,532.73</strong>, outperforming Seasonal Persistence MPC (<strong>$36,537.91</strong>) by <strong>$5.18</strong>. This difference constitutes exactly <strong>0.0142%</strong> (~1.4 parts in 10,000) of Seasonal MPC's total operating cost.
            </p>
            <p>
              <strong>Why did they report identical throughput (31,540.80 kWh)?</strong><br />
              Audit of the 1,313 hourly records revealed that both controllers took identical actions in 1,301 out of 1,313 hours (99.09%). They differed in only <strong>12 hours</strong> during a single 35-hour solar event across November 28–29:
            </p>
            <ul className="list-disc pl-4 space-y-1 text-slate-500">
              <li>On Nov 28, GNN anticipated solar surplus and charged early, absorbing <strong>64.81 kWh</strong> that Seasonal exported at $0.04/kWh.</li>
              <li>On Nov 29, Seasonal (relying on yesterday's lag) grid-charged when solar was absent, purchasing 64.81 kWh at $0.12/kWh.</li>
              <li>Because both controllers fully traversed the battery capacity (0.67 → 0.90 → 0.10 SOC), their cumulative throughput was <strong>mathematically identical to double precision</strong> (621.80 kWh during the event).</li>
              <li>The $5.18 difference is the exact net value of the 64.81 kWh timing delta ($7.78 avoided grid import - $2.59 lost export revenue).</li>
            </ul>
          </div>
        </div>

        {/* Finding 2: Winter Heuristic Inversion */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
              Winter Heuristic Inversion
            </h3>
          </div>

          <div className="space-y-2 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            <p>
              A major empirical finding of Part 8 is that rule-based price-aware heuristics that performed well in autumn validation (saving $220–$233) <strong>inverted into net economic losses</strong> under winter test conditions:
            </p>
            <ul className="list-disc pl-4 space-y-1 text-slate-500">
              <li>Continuous Price-Aware: <strong>-$15.87 net loss</strong> vs zero-action ($36,626.52 adjusted cost).</li>
              <li>Discrete Price-Aware: <strong>-$3.14 net loss</strong> vs zero-action ($36,613.79 adjusted cost).</li>
            </ul>
            <p>
              <strong>The Physical Cause:</strong> In winter, solar generation dropped by 33.6% (76.3 MWh vs 114.9 MWh). The narrow $0.06/kWh price spread ($0.12 off-peak to $0.18 on-peak) was insufficient to overcome:
            </p>
            <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-xl font-mono text-[11px] text-slate-700 dark:text-slate-300">
              Breakeven Price = (0.12 / 0.9025 roundtrip) + $0.02 degradation = $0.153/kWh
            </div>
            <p>
              Without horizon optimization, the heuristics cycled blindly (40–45 FCE), incurring $800–$910 in degradation and conversion losses. Both MPC formulations dynamically throttled throughput to 31.54 FCE, preserving positive economics.
            </p>
          </div>
        </div>

        {/* Finding 3: Reinforcement Learning (PPO) Conservatism */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
              PPO Policy Conservatism & Inventory Accounting
            </h3>
          </div>

          <div className="space-y-2 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            <p>
              The frozen PPO continuous policy (trained for 16,384 steps in Part 6) learned an inactive policy, discharging only <strong>190.0 kWh AC</strong> over the entire 1,313 hours (0.19 FCE).
            </p>
            <p>
              In raw unadjusted cost, PPO recorded $36,591.65 (which superficially appeared lower than zero-action by $19.00). However, depleting the battery from SOC 0.50 to 0.10 left a 200 kWh deficit:
            </p>
            <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-xl font-mono text-[11px] text-slate-700 dark:text-slate-300">
              Terminal Adjustment = -200 kWh × $0.147368/kWh = -$29.47<br />
              Adjusted Cost = $36,591.65 - (-$29.47) = $36,621.13 (+$10.48 loss vs zero)
            </div>
            <p>
              This underscores the necessity of terminal inventory credit accounting to prevent controllers from appearing artificially cost-effective by exhausting storage assets.
            </p>
          </div>
        </div>

        {/* Finding 4: Scientific Scope & Remaining Limitations */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
              Scientific Boundaries & Limitations
            </h3>
          </div>

          <div className="space-y-2 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            <ul className="list-disc pl-4 space-y-2 text-slate-500">
              <li>
                <strong>Single-Episode Test Evaluation:</strong> The held-out test split is a single 55-day realization (winter 2026). No multi-year asymptotic statistical significance can be claimed without multi-season validation.
              </li>
              <li>
                <strong>Oracle Definition:</strong> The rolling 24-hour oracle ($36,314.90, +$295.75) is a <em>noncausal diagnostic reference</em> with 24-hour lookahead, not an unconstrained global mathematical optimum over the full 1,313 hours.
              </li>
              <li>
                <strong>Simulated Latency vs Field SCADA:</strong> GNN-MPC's 14.55 ms mean execution latency confirms computational tractability, but does not prove field SCADA readiness, which entails telemetry loss, meter latency, and communication drops.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
