import React, { useState, useEffect } from 'react';
import {
  Battery,
  Zap,
  TrendingDown,
  Repeat,
  ShieldAlert,
  Percent,
  CheckCircle2,
  DollarSign,
  ShieldCheck,
  Info,
  Sliders,
} from 'lucide-react';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import MetricCard from '../components/MetricCard';
import BatteryGauge from '../components/BatteryGauge';
import CausalityBadge from '../components/CausalityBadge';
import { formatNumber, formatCurrency, formatPercent } from '../utils/formatters';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

export default function BatteryManagement() {
  const [controllerId, setControllerId] = useState('mpc_gnn_transformer');
  const [batteryData, setBatteryData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadBattery() {
      setLoading(true);
      try {
        const res = await api.getBatteryDispatch({ controllerId, partition: 'test' });
        setBatteryData(res);
      } catch (err) {
        console.error('Failed to load battery data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadBattery();
  }, [controllerId]);

  if (loading || !batteryData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const { specs, metrics, soc_trajectory_sample } = batteryData;

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Battery Storage Management & Health"
        subtitle="Track stored energy, charging patterns, and battery wear."
        action={
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Select Controller:</span>
            <select
              value={controllerId}
              onChange={(e) => setControllerId(e.target.value)}
              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-1.5 text-xs font-semibold text-slate-800 dark:text-slate-200 shadow-sm focus:ring-2 focus:ring-blue-500"
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
        }
      />

      {/* 1. Main Battery Gauge & Physical Specs Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <BatteryGauge
            soc={metrics.terminal_soc}
            powerKw={0}
            capacityKwh={specs.nominal_capacity_kwh}
          />

          {/* Quick Metrics Grid with InfoTooltips */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Throughput"
              technicalTitle="Total Cycled Energy"
              value={`${formatNumber(metrics.throughput_kwh_ac, 0)} kWh`}
              unit=""
              subtext="Total AC Energy Cycled"
              icon={Repeat}
              color="emerald"
              tooltipData={{
                what: 'The cumulative electrical energy discharged and charged through the inverter across all 1,313 hours.',
                why: 'More cycling creates more opportunities for price arbitrage, but also increases cell wear.',
                how: 'Intelligent controllers cycle selectively only when tariff spreads justify cell degradation.',
                technical: 'Throughput = sum |P_{batt,t}| * Delta t',
              }}
            />

            <MetricCard
              title="Full Cycles"
              technicalTitle="Full Cycle Equivalents (FCE)"
              value={`${formatNumber(metrics.full_cycle_equivalents, 1)}`}
              unit="FCE"
              subtext="Over 1,313 hours (54.7 days)"
              icon={Battery}
              color="blue"
              tooltipData={{
                what: 'The number of complete 100% discharge-and-recharge cycles experienced by the battery pack.',
                why: 'Lithium iron phosphate (LFP) cells are rated for ~3,000 to 5,000 full cycles before reaching 80% capacity.',
                how: '31.5 FCE over 54.7 days equates to ~0.58 cycles per day, well within healthy utility lifespan limits.',
                technical: 'FCE = Throughput / (2 * Capacity)',
              }}
            />

            <MetricCard
              title="Degradation Cost"
              technicalTitle="Cell Wear Amortization"
              value={formatCurrency(metrics.degradation_cost_usd)}
              unit=""
              subtext="$0.02 / kWh AC Throughput"
              icon={TrendingDown}
              color="amber"
              tooltipData={{
                what: 'The economic cost allocated to battery cell aging for every kilowatt-hour transferred.',
                why: 'Batteries are not free to cycle. Without this cost term, naive algorithms cycle frantically for pennies of gain.',
                how: 'Fixed conservatively at $0.02/kWh, reflecting modern commercial battery replacement and warranty costs.',
                technical: 'C_deg = 0.02 * Throughput_kWh',
              }}
            />

            <MetricCard
              title="Inventory Value"
              technicalTitle="Terminal SOC Credit"
              value={formatCurrency(metrics.terminal_inventory_credit_usd)}
              unit=""
              subtext={`Terminal SOC: ${formatPercent(metrics.terminal_soc)}`}
              icon={DollarSign}
              color={metrics.terminal_inventory_credit_usd >= 0 ? 'emerald' : 'rose'}
              tooltipData={{
                what: 'Financial value of the energy left inside the battery at the end of the 1,313-hour test.',
                why: 'Prevents cheating where a controller drains the battery to 0% to show artificial grid savings.',
                how: 'Valued at the average electricity tariff ($0.12/kWh). Positive means net energy stored for future use.',
                technical: 'Credit = (SOC_terminal - SOC_initial) * Capacity * Avg_Price',
              }}
            />
          </div>
        </div>

        {/* Physical Specs Sidebar */}
        <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800/80">
            <h3 className="font-bold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Storage Physical Limits
            </h3>
            <CausalityBadge causality={controllerId === 'mpc_oracle_rolling_24h' ? 'noncausal_lookahead' : 'strictly_causal'} />
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Nominal Energy Capacity</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">{specs.nominal_capacity_kwh} kWh</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Symmetric Inverter Limit</span>
              <span className="font-bold text-violet-600 dark:text-violet-400">±{specs.max_power_kw} kW</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Charging Efficiency (η_c)</span>
              <span className="font-bold text-blue-600 dark:text-blue-400">{(specs.charging_efficiency * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Discharging Efficiency (η_d)</span>
              <span className="font-bold text-blue-600 dark:text-blue-400">{(specs.discharging_efficiency * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Round-Trip Efficiency (η_c × η_d)</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">{(specs.round_trip_efficiency * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Operating SOC Range</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">10% – 90% (400 kWh usable)</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Degradation Fee</span>
              <span className="font-bold text-rose-600 dark:text-rose-400">${specs.degradation_rate_usd_per_kwh} / kWh</span>
            </div>
          </div>

          <div className="p-3 bg-violet-50/70 dark:bg-violet-950/20 border border-violet-200/80 dark:border-violet-800/40 rounded-2xl text-[11px] text-violet-900 dark:text-violet-300 space-y-1">
            <span className="font-bold block">Arbitrage Threshold Principle:</span>
            <p className="leading-relaxed">
              The battery only dispatches when the electricity price spread exceeds the round-trip efficiency penalty plus the $0.02/kWh cell wear fee.
            </p>
          </div>
        </div>
      </div>

      {/* 2. SOC Trajectory Chart */}
      <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100 dark:border-slate-800/80">
          <div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
              State of Charge (SOC) Trajectory under Selected Controller
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Notice how MPC maintains battery SOC strictly inside the 10%–90% safe buffer, timing charge during solar hours and discharge during tariff peaks.
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
            10% Min Buffer • 90% Max Buffer
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={soc_trajectory_sample} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.15} />
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} tickFormatter={(s) => `Step ${s}`} tickLine={false} />
              <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={11} tickLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip formatter={(val) => [`${(val * 100).toFixed(1)}%`, 'Battery SOC']} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
              <Line type="monotone" dataKey="soc" name="Battery SOC" stroke="#8b5cf6" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
