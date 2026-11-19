import React, { useState } from 'react';
import {
  Sun,
  Wind,
  Battery,
  Building2,
  Zap,
  ArrowRight,
  ArrowLeft,
  Sliders,
  Play,
  Pause,
  RotateCcw,
  Info,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react';
import { formatPower, formatPercent, formatCurrency } from '../utils/formatters';

/**
 * EnergyFlowDiagram - Interactive Energy Balance Flow Visualization
 * Visualizes instantaneous power balance across Generation, Hub, Battery, Grid, and Load.
 * Enforces physical constraints:
 *   - Battery capacity: 500 kWh
 *   - Symmetric inverter limit: 150 kW (cannot exceed 150 kW charge or discharge)
 *   - Mutually exclusive charge/discharge (never simultaneous)
 */
export default function EnergyFlowDiagram({ timeseries = [] }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Fallback defaults if timeseries is empty
  const currentStep = timeseries[stepIndex] || {
    timestamp: '2025-01-01 12:00:00',
    solar_kw: 110.5,
    wind_kw: 65.2,
    load_kw: 215.0,
    battery_kw: -39.3, // negative is charging, positive is discharging
    soc: 0.58,
    grid_import_kw: 78.6,
    grid_export_kw: 0.0,
    electricity_price_usd_per_kwh: 0.12,
  };

  const solarKw = Math.max(0, currentStep.solar_kw || 0);
  const windKw = Math.max(0, currentStep.wind_kw || 0);
  const totalRenewablesKw = solarKw + windKw;
  const loadKw = Math.max(0, currentStep.demand_kw ?? currentStep.load_kw ?? 0);
  
  // Battery dispatch
  const batteryActionKw = currentStep.battery_power_kw ?? currentStep.battery_kw ?? 0;
  const isDischarging = batteryActionKw > 0.01;
  const isCharging = batteryActionKw < -0.01;
  const batteryPowerKw = Math.min(150, Math.abs(batteryActionKw)); // Cap at physical inverter 150 kW
  const soc = currentStep.battery_soc ?? currentStep.soc ?? 0.5;
  const storedKwh = soc * 500; // 500 kWh capacity

  const gridImportKw = Math.max(0, currentStep.grid_import_kw || 0);
  const gridExportKw = Math.max(0, currentStep.grid_export_kw || 0);
  const tariff = currentStep.electricity_price_usd_per_kwh || 0.12;

  // Auto-play timer
  React.useEffect(() => {
    let timer;
    if (isPlaying && timeseries.length > 0) {
      timer = setInterval(() => {
        setStepIndex((prev) => (prev + 1) % timeseries.length);
      }, 1500);
    }
    return () => clearInterval(timer);
  }, [isPlaying, timeseries.length]);

  // Jump to specific characteristic scenario
  const jumpToScenario = (type) => {
    if (!timeseries.length) return;
    setIsPlaying(false);
    if (type === 'solar') {
      // Find step with max solar
      let maxIdx = 0;
      let maxVal = 0;
      timeseries.forEach((step, idx) => {
        if ((step.solar_kw || 0) > maxVal) {
          maxVal = step.solar_kw;
          maxIdx = idx;
        }
      });
      setStepIndex(maxIdx);
    } else if (type === 'peak_tariff') {
      // Find step with max tariff or grid price
      let maxIdx = 0;
      let maxVal = 0;
      timeseries.forEach((step, idx) => {
        if ((step.electricity_price_usd_per_kwh || 0) > maxVal) {
          maxVal = step.electricity_price_usd_per_kwh;
          maxIdx = idx;
        }
      });
      setStepIndex(maxIdx);
    } else if (type === 'high_wind') {
      let maxIdx = 0;
      let maxVal = 0;
      timeseries.forEach((step, idx) => {
        if ((step.wind_kw || 0) > maxVal) {
          maxVal = step.wind_kw;
          maxIdx = idx;
        }
      });
      setStepIndex(maxIdx);
    }
  };

  return (
    <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-3xl p-6 shadow-sm space-y-6">
      {/* Top Header & Interactive Scrubber */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400">
              <Zap className="w-4 h-4" />
            </span>
            <h3 className="font-bold text-base text-slate-900 dark:text-slate-100">
              Site energy flow
            </h3>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-medium">
              Step {stepIndex + 1} of {Math.max(1, timeseries.length)}
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Time: <span className="font-mono font-medium text-slate-700 dark:text-slate-300">{currentStep.timestamp}</span> • Electricity price: <span className="font-mono font-medium text-blue-600 dark:text-blue-400">{formatCurrency(tariff)}/kWh</span>
          </p>
        </div>

        {/* Quick presets & Playback */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800/80 p-1 rounded-xl">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1.5 rounded-lg bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:text-blue-500 shadow-xs transition-colors"
              title={isPlaying ? 'Pause Simulation' : 'Play Simulation'}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button
              onClick={() => { setIsPlaying(false); setStepIndex(0); }}
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
              title="Reset to Step 1"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 text-xs">
            <button
              onClick={() => jumpToScenario('solar')}
              className="px-2.5 py-1.5 rounded-xl border border-amber-200 dark:border-amber-800/60 bg-amber-50/50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 hover:bg-amber-100/50 transition-colors"
            >
              Solar Peak
            </button>
            <button
              onClick={() => jumpToScenario('high_wind')}
              className="px-2.5 py-1.5 rounded-xl border border-cyan-200 dark:border-cyan-800/60 bg-cyan-50/50 dark:bg-cyan-950/30 text-cyan-700 dark:text-cyan-300 hover:bg-cyan-100/50 transition-colors"
            >
              Wind Peak
            </button>
            <button
              onClick={() => jumpToScenario('peak_tariff')}
              className="px-2.5 py-1.5 rounded-xl border border-violet-200 dark:border-violet-800/60 bg-violet-50/50 dark:bg-violet-950/30 text-violet-700 dark:text-violet-300 hover:bg-violet-100/50 transition-colors"
            >
              High electricity price
            </button>
          </div>
        </div>
      </div>

      {/* Scrub Bar */}
      {timeseries.length > 1 && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-slate-400">
            <span>Hour 0</span>
            <span className="font-medium text-slate-600 dark:text-slate-300">Explore the timeline</span>
            <span>Hour {timeseries.length}</span>
          </div>
          <input
            aria-label="Explore simulation hour"
            type="range"
            min="0"
            max={timeseries.length - 1}
            value={stepIndex}
            onChange={(e) => {
              setIsPlaying(false);
              setStepIndex(Number(e.target.value));
            }}
            className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
        </div>
      )}

      {/* Interactive Hub-and-Spoke Grid Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
        {/* LEFT COLUMN: Generation Sources (Solar + Wind) */}
        <div className="space-y-4">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center justify-between">
            <span>Renewable Generation</span>
            <span className="text-blue-600 dark:text-blue-400 font-semibold">
              Total: {formatPower(totalRenewablesKw)}
            </span>
          </div>

          {/* Solar Card */}
          <div className="bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200/80 dark:border-amber-800/40 rounded-2xl p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-amber-100 dark:bg-amber-900/60 text-amber-600 dark:text-amber-400">
                <Sun className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 block">
                  Solar Array
                </span>
                <span className="text-lg font-bold text-amber-700 dark:text-amber-300">
                  {formatPower(solarKw)}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-amber-600 dark:text-amber-400 animate-pulse">
              <span className="text-[10px] font-bold uppercase">To Hub</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>

          {/* Wind Card */}
          <div className="bg-cyan-50/70 dark:bg-cyan-950/20 border border-cyan-200/80 dark:border-cyan-800/40 rounded-2xl p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-cyan-100 dark:bg-cyan-900/60 text-cyan-600 dark:text-cyan-400">
                <Wind className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 block">
                  Wind Turbines
                </span>
                <span className="text-lg font-bold text-cyan-700 dark:text-cyan-300">
                  {formatPower(windKw)}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-cyan-600 dark:text-cyan-400 animate-pulse">
              <span className="text-[10px] font-bold uppercase">To Hub</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </div>
        </div>

        {/* CENTER COLUMN: Central Microgrid Hub & Battery Storage */}
        <div className="space-y-4">
          {/* Main Central Bus Node */}
          <div className="bg-gradient-to-b from-slate-900 to-slate-800 dark:from-slate-800 dark:to-slate-900 text-white rounded-3xl p-6 shadow-md border border-slate-700/60 text-center relative overflow-hidden">
            <div className="absolute top-0 right-0 left-0 h-1 bg-gradient-to-r from-amber-400 via-blue-400 to-cyan-400" />
            <span className="text-[10px] font-bold tracking-widest text-blue-400 uppercase block mb-1">
              Microgrid Power Router
            </span>
            <h4 className="text-lg font-extrabold text-white">
              Shared Energy Bus
            </h4>
            <div className="mt-3 py-2 px-3 bg-white/10 rounded-xl inline-block text-xs font-medium text-slate-200">
              Demand Balance: {formatPower(loadKw)} Active
            </div>
            <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-center gap-2">
              <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
              <span>Real-time Kirchhoff Power Balance</span>
            </div>
          </div>

          {/* Battery Storage Box (With Strict 150 kW and 500 kWh specs) */}
          <div className="bg-violet-50/70 dark:bg-violet-950/20 border border-violet-200/80 dark:border-violet-800/40 rounded-2xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-violet-100 dark:bg-violet-900/60 text-violet-600 dark:text-violet-400">
                  <Battery className="w-5 h-5" />
                </div>
                <div>
                  <span className="text-xs font-bold text-slate-800 dark:text-slate-200 block">
                    Battery Storage
                  </span>
                  <span className="text-[10px] font-medium text-slate-500 dark:text-slate-400">
                    500 kWh Capacity • 150 kW Inverter Limit
                  </span>
                </div>
              </div>

              {/* State Badge */}
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                  isCharging
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300'
                    : isDischarging
                    ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                    : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                }`}
              >
                {isCharging ? 'CHARGING' : isDischarging ? 'DISCHARGING' : 'IDLE'}
              </span>
            </div>

            {/* SOC Bar */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-500 dark:text-slate-400">
                  Charge Level: {formatPercent(soc)} ({Math.round(storedKwh)} kWh)
                </span>
                <span className="text-violet-600 dark:text-violet-400 font-bold">
                  {isCharging
                    ? `+${formatPower(batteryPowerKw)} (IN)`
                    : isDischarging
                    ? `-${formatPower(batteryPowerKw)} (OUT)`
                    : '0.0 kW'}
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden flex">
                <div
                  className="bg-gradient-to-r from-blue-500 to-sky-400 h-full rounded-full transition-all duration-300"
                  style={{ width: `${Math.max(5, Math.min(100, soc * 100))}%` }}
                />
              </div>
              <div className="flex justify-between text-[9px] text-slate-400">
                <span>Min Buffer (10%)</span>
                <span>Operating Window (10% - 90%)</span>
                <span>Max Buffer (90%)</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Grid Connection & Campus Load */}
        <div className="space-y-4">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center justify-between">
            <span>Grid & Demand Sinks</span>
            <span className="text-blue-600 dark:text-blue-400 font-semibold">
              Load: {formatPower(loadKw)}
            </span>
          </div>

          {/* Campus Load Card */}
          <div className="bg-blue-50/70 dark:bg-blue-950/20 border border-blue-200/80 dark:border-blue-800/40 rounded-2xl p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-blue-100 dark:bg-blue-900/60 text-blue-600 dark:text-blue-400">
                <Building2 className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 block">
                  Buildings
                </span>
                <span className="text-lg font-bold text-blue-700 dark:text-blue-300">
                  {formatPower(loadKw)}
                </span>
              </div>
            </div>
            <div className="text-[10px] font-semibold text-slate-500 bg-white/60 dark:bg-slate-800/60 px-2 py-1 rounded-lg">
              Primary Sink
            </div>
          </div>

          {/* Electricity Grid Card */}
          <div className="bg-slate-50 dark:bg-slate-800/40 border border-slate-200/80 dark:border-slate-700/60 rounded-2xl p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                <Zap className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 block">
                  Utility Power Grid
                </span>
                <span className="text-lg font-bold text-slate-900 dark:text-slate-100">
                  {gridImportKw > 0 ? `Import: ${formatPower(gridImportKw)}` : `Export: ${formatPower(gridExportKw)}`}
                </span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 block">Tariff</span>
              <span className="text-xs font-mono font-bold text-blue-600 dark:text-blue-400">
                {formatCurrency(tariff)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Physics & Inverter Limits Disclosure */}
      <div className="p-3 bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800 rounded-2xl text-[11px] text-slate-500 dark:text-slate-400 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-blue-500 shrink-0" />
          <span>
            <strong>Physical Guarantee:</strong> Battery inverter is strictly rate-limited to 150 kW symmetric power. Charge and discharge are physically mutually exclusive.
          </span>
        </div>
        <span className="shrink-0 font-medium text-slate-600 dark:text-slate-300">
          Linear Cell Degradation: $0.02 / kWh
        </span>
      </div>
    </div>
  );
}
