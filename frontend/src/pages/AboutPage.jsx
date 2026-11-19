import React from 'react';
import {
  BookOpen,
  Zap,
  Cpu,
  GitBranch,
  ShieldCheck,
  Server,
  Layers,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import SectionHeader from '../components/SectionHeader';

export default function AboutPage() {
  return (
    <div className="space-y-8 max-w-5xl">
      <SectionHeader
        title="About the Adaptive Energy Brain Project"
        subtitle="A generalized, AI-driven energy management system engineered for renewable communities, microgrids, smart buildings, and commercial facilities; demonstrated on a hypothetical campus microgrid case study."
      />

      {/* High-Level Overview */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Zap className="w-5 h-5 text-emerald-500" />
          <span>Project Mission & Overview</span>
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          The <strong>Adaptive Energy Brain</strong> is a comprehensive research and engineering platform designed to optimize energy dispatch across renewable communities, microgrids, smart buildings, university campuses, and commercial facilities. Validated on a multi-building campus microgrid demonstration case study, the system coordinates local solar photovoltaics (PV), wind generation, stationary battery storage, and utility grid imports to achieve three foundational goals:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 space-y-2">
            <div className="font-bold text-xs uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
              1. Minimize Operating Cost
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Arbitraging two-tier time-of-use tariffs by scheduling battery charging during off-peak windows ($0.12/kWh) and discharging during on-peak windows ($0.18/kWh).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 space-y-2">
            <div className="font-bold text-xs uppercase tracking-wider text-blue-600 dark:text-blue-400">
              2. Protect Battery Lifespan
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Penalizing physical battery cycling ($0.02/kWh AC throughput) and enforcing strict non-simultaneous charge/discharge flow constraints between 10% and 90% SOC.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 space-y-2">
            <div className="font-bold text-xs uppercase tracking-wider text-purple-600 dark:text-purple-400">
              3. Maximize Clean Energy
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Anticipating solar and wind variability using 24-hour neural lookahead to capture renewable self-consumption and prevent curtailment or low-value feed-in export.
            </p>
          </div>
        </div>
      </div>

      {/* System Architecture Flow Diagram */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Layers className="w-5 h-5 text-blue-500" />
          <span>End-to-End System Architecture</span>
        </h2>

        {/* Visual Workflow Steps */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-1.5">
            <div className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-[10px]">1</span>
              <span>Demonstration Microgrid</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-[11px]">
              8-node topology with 5 buildings, 300 kW PV array, 150 kW wind turbine, and a 500 kWh / 150 kW battery.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-1.5">
            <div className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-950 text-emerald-600 flex items-center justify-center text-[10px]">2</span>
              <span>GNN-Transformer</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-[11px]">
              Spatial message passing across electrical feeders + temporal attention predicting 24 hours of load and solar.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-1.5">
            <div className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-blue-100 dark:bg-blue-950 text-blue-600 flex items-center justify-center text-[10px]">3</span>
              <span>HiGHS LP Solver</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-[11px]">
              Model Predictive Control solves a rolling 24-hour linear program to optimize power dispatch and degradation.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 space-y-1.5">
            <div className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-purple-100 dark:bg-purple-950 text-purple-600 flex items-center justify-center text-[10px]">4</span>
              <span>Physical Simulator</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-[11px]">
              Deterministic physical engine executes dispatch, updates SOC, checks constraints, and records audited costs.
            </p>
          </div>
        </div>
      </div>

      {/* Control Paradigms: MPC vs PPO vs Heuristics */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-purple-500" />
          <span>Understanding Control Approaches: MPC vs PPO vs Rule-Based</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-bold text-emerald-600 dark:text-emerald-400 text-sm">
              Model Predictive Control (MPC)
            </div>
            <div className="text-slate-500 font-medium">Model-Based Optimization (HiGHS LP)</div>
            <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
              Formulates battery dispatch as a mathematical linear program over a rolling 24-hour horizon. Directly incorporates battery efficiencies (95%), degradation costs ($0.02/kWh), and physical limits.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-emerald-700 dark:text-emerald-300">
              Result: Best performer across both autumn ($43,702.87) and winter ($36,532.73).
            </div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-bold text-blue-600 dark:text-blue-400 text-sm">
              Proximal Policy Optimization (PPO)
            </div>
            <div className="text-slate-500 font-medium">Model-Free Reinforcement Learning</div>
            <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
              Learns a neural actor-critic policy via trial-and-error interactions. Without an explicit physical simulator inside the decision loop, PPO struggled with long discount horizons and converged to conservative near-zero dispatch.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-amber-700 dark:text-amber-300">
              Result: Near-zero cycling (0.19 FCE); penalized by terminal inventory adjustment.
            </div>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2">
            <div className="font-bold text-purple-600 dark:text-purple-400 text-sm">
              Price-Aware Rule Heuristics
            </div>
            <div className="text-slate-500 font-medium">Fixed Time-of-Use Arbitrage Logic</div>
            <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
              Commands battery charging during off-peak hours ($0.12/kWh) and discharging during on-peak hours ($0.18/kWh). Does not evaluate weather forecasts or physical conversion losses.
            </p>
            <div className="pt-2 text-[11px] font-semibold text-rose-700 dark:text-rose-300">
              Result: Inverted in winter; degradation and round-trip losses exceeded price spreads.
            </div>
          </div>
        </div>
      </div>

      {/* Technology Stack & Reproducibility */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Server className="w-5 h-5 text-emerald-500" />
          <span>Technology Stack & Reproducibility</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs text-slate-600 dark:text-slate-400">
          <div className="space-y-2">
            <h4 className="font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider text-[11px]">
              Backend & AI Environment
            </h4>
            <ul className="list-disc pl-4 space-y-1">
              <li>Python 3.12 with virtual environment (<code className="font-mono text-emerald-600">.venv</code>)</li>
              <li>PyTorch 2.7.0 (CPU inference) for GNN-Transformer models</li>
              <li>SciPy 1.18.1 (HiGHS LP solver) for Model Predictive Control</li>
              <li>Stable-Baselines3 2.7.1 & Gymnasium for DRL policies</li>
              <li>Standard library <code className="font-mono text-emerald-600">http.server</code> for lightweight, zero-dependency REST API</li>
            </ul>
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider text-[11px]">
              Frontend & UX Architecture
            </h4>
            <ul className="list-disc pl-4 space-y-1">
              <li>React 18 with modern functional components & hooks</li>
              <li>Vite 6 for lightning-fast HMR and optimized production bundling</li>
              <li>Tailwind CSS 3 for responsive, dark-mode native SaaS interface</li>
              <li>Recharts 2.13 for responsive microgrid time-series and comparative bar charts</li>
              <li>Lucide React for consistent, accessible iconography</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
