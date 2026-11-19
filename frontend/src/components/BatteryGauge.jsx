import React from 'react';
import { Battery, Zap } from 'lucide-react';

export default function BatteryGauge({ soc = 0.5, powerKw = 0, capacityKwh = 500, size = 'md' }) {
  const percentage = Math.min(100, Math.max(0, Math.round(soc * 100)));
  const storedKwh = (soc * capacityKwh).toFixed(1);

  // Dynamic status colors based on SOC
  let barColor = 'bg-emerald-500';
  let badgeColor = 'text-emerald-600 bg-emerald-50 dark:bg-emerald-950/50';
  if (percentage < 20) {
    barColor = 'bg-rose-500';
    badgeColor = 'text-rose-600 bg-rose-50 dark:bg-rose-950/50';
  } else if (percentage < 40) {
    barColor = 'bg-amber-500';
    badgeColor = 'text-amber-600 bg-amber-50 dark:bg-amber-950/50';
  }

  const isCharging = powerKw < -0.1;
  const isDischarging = powerKw > 0.1;

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400">
            <Battery className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Battery Storage State
            </h4>
            <div className="text-xl font-bold text-slate-900 dark:text-slate-50">
              {percentage}% <span className="text-xs font-normal text-slate-500">({storedKwh} / {capacityKwh} kWh)</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full ${badgeColor}`}>
            <Zap className={`w-3.5 h-3.5 ${isCharging || isDischarging ? 'animate-pulse' : ''}`} />
            {isCharging && `Charging (${Math.abs(powerKw).toFixed(1)} kW)`}
            {isDischarging && `Discharging (${powerKw.toFixed(1)} kW)`}
            {!isCharging && !isDischarging && 'Idle (0 kW)'}
          </span>
        </div>
      </div>

      {/* Visual Battery Bar */}
      <div className="relative w-full h-5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-200/60 dark:border-slate-700/60">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${percentage}%` }}
        />
        {/* SOC Guidelines (10% and 90% operating limits) */}
        <div className="absolute top-0 bottom-0 left-[10%] w-0.5 bg-rose-400/60 border-dashed" title="Min SOC Limit (10%)" />
        <div className="absolute top-0 bottom-0 left-[90%] w-0.5 bg-rose-400/60 border-dashed" title="Max SOC Limit (90%)" />
      </div>

      <div className="flex justify-between items-center mt-2 text-[11px] text-slate-400">
        <span>0% (Empty)</span>
        <span className="text-slate-500 font-medium">Safe Operational Range (10% - 90%)</span>
        <span>100% (500 kWh)</span>
      </div>
    </div>
  );
}
