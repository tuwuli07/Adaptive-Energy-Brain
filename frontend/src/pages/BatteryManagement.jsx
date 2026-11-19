import React, { useState, useEffect } from 'react';
import {
  Battery,
  Zap,
  TrendingDown,
  Repeat,
  ShieldAlert,
  Percent,
  CheckCircle2,
  DollarSign
} from 'lucide-react';
import { api } from '../api/client';
import SectionHeader from '../components/SectionHeader';
import StatCard from '../components/StatCard';
import BatteryGauge from '../components/BatteryGauge';
import CausalityBadge from '../components/CausalityBadge';
import { formatNumber, formatCurrency } from '../utils/formatters';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
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
        <div className="w-8 h-8 border-3 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const { specs, metrics, soc_trajectory_sample } = batteryData;

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Battery Storage Management & Degradation Analytics"
        subtitle="Detailed physical telemetry, cycle aging, non-simultaneous power limits, and state-of-charge regulation."
        action={
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Select Controller:</span>
            <select
              value={controllerId}
              onChange={(e) => setControllerId(e.target.value)}
              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-1.5 text-xs font-semibold text-slate-800 dark:text-slate-200 shadow-sm focus:ring-2 focus:ring-emerald-500"
            >
              <option value="mpc_gnn_transformer">GNN-Transformer MPC</option>
              <option value="mpc_seasonal_persistence">Seasonal Persistence MPC</option>
              <option value="price_aware_continuous">Continuous Price-Aware Baseline</option>
              <option value="price_aware_discrete">Discrete Price-Aware Baseline</option>
              <option value="ppo_continuous_control">PPO Continuous Control (16k)</option>
              <option value="zero_action">Zero-Action Baseline</option>
              <option value="mpc_oracle_rolling_24h">Oracle MPC (24h Diagnostic)</option>
            </select>
          </div>
        }
      />

      {/* Main Battery Gauge & Physical Specs Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <BatteryGauge
            soc={metrics.terminal_soc}
            powerKw={0}
            capacityKwh={specs.nominal_capacity_kwh}
          />

          {/* Quick Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard
              title="Throughput"
              value={`${formatNumber(metrics.throughput_kwh_ac, 0)} kWh`}
              subtext="Total AC Energy Cycled"
              icon={Repeat}
              color="brand"
            />
            <StatCard
              title="Full Cycles (FCE)"
              value={`${formatNumber(metrics.full_cycle_equivalents, 2)}`}
              subtext="Throughput / (2 × Cap)"
              icon={Battery}
              color="blue"
            />
            <StatCard
              title="Degradation Cost"
              value={formatCurrency(metrics.degradation_cost_usd)}
              subtext="$0.02 / kWh AC Throughput"
              icon={TrendingDown}
              color="amber"
            />
            <StatCard
              title="Inventory Credit"
              value={formatCurrency(metrics.terminal_inventory_credit_usd)}
              subtext={`Terminal SOC: ${(metrics.terminal_soc * 100).toFixed(0)}%`}
              icon={DollarSign}
              color={metrics.terminal_inventory_credit_usd >= 0 ? 'brand' : 'red'}
            />
          </div>
        </div>

        {/* Physical Specs Sidebar */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Storage Physics Specifications
            </h3>
            <CausalityBadge causality={controllerId === 'mpc_oracle_rolling_24h' ? 'noncausal_lookahead' : 'strictly_causal'} />
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Nominal Energy Capacity</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">{specs.nominal_capacity_kwh} kWh</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Maximum Inverter Power</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">±{specs.max_power_kw} kW</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Charging Efficiency (η_c)</span>
              <span className="font-bold text-emerald-600 dark:text-emerald-400">{(specs.charging_efficiency * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Discharging Efficiency (η_d)</span>
              <span className="font-bold text-emerald-600 dark:text-emerald-400">{(specs.discharging_efficiency * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Round-Trip Efficiency (η_c × η_d)</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">{(specs.round_trip_efficiency * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100 dark:border-slate-800">
              <span className="text-slate-500">Operational SOC Limits</span>
              <span className="font-bold text-slate-800 dark:text-slate-200">10% – 90% (400 kWh usable)</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Degradation Rate</span>
              <span className="font-bold text-rose-600 dark:text-rose-400">${specs.degradation_rate_usd_per_kwh}/kWh</span>
            </div>
          </div>
        </div>
      </div>

      {/* SOC Trajectory Chart */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm space-y-3">
        <h3 className="font-semibold text-sm text-slate-800 dark:text-slate-200">
          Sample State of Charge (SOC) Trajectory under Selected Controller
        </h3>
        <p className="text-xs text-slate-400">
          Showing battery SOC evolution. Notice how MPC keeps SOC bounded between 10% and 90%, cycling strictly when arbitrage exceeds $0.02/kWh + roundtrip loss.
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={soc_trajectory_sample} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.5} />
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} tickFormatter={(s) => `Step ${s}`} tickLine={false} />
              <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={11} tickLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip formatter={(val) => [`${(val * 100).toFixed(1)}%`, 'Battery SOC']} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
              <Line type="monotone" dataKey="soc" name="Battery SOC" stroke="#10b981" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
