import React from 'react';
import { Info } from 'lucide-react';

export default function StatCard({
  title,
  value,
  subtext,
  icon: Icon,
  trend,
  trendPositive = true,
  tooltip,
  color = 'brand',
}) {
  const colorMap = {
    brand: 'bg-emerald-50 text-emerald-600 dark:bg-emerald-950/50 dark:text-emerald-400',
    amber: 'bg-amber-50 text-amber-600 dark:bg-amber-950/50 dark:text-amber-400',
    blue: 'bg-blue-50 text-blue-600 dark:bg-blue-950/50 dark:text-blue-400',
    purple: 'bg-purple-50 text-purple-600 dark:bg-purple-950/50 dark:text-purple-400',
    red: 'bg-rose-50 text-rose-600 dark:bg-rose-950/50 dark:text-rose-400',
    slate: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
  };

  const iconClasses = colorMap[color] || colorMap.brand;

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all duration-200 flex flex-col justify-between">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              {title}
            </span>
            {tooltip && (
              <span className="cursor-help text-slate-400 hover:text-slate-600 dark:hover:text-slate-200" title={tooltip}>
                <Info className="w-3.5 h-3.5" />
              </span>
            )}
          </div>
          <div className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-50">
            {value}
          </div>
        </div>
        {Icon && (
          <div className={`p-2.5 rounded-xl ${iconClasses}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      {(subtext || trend) && (
        <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800/60 flex items-center justify-between text-xs">
          {subtext && <span className="text-slate-500 dark:text-slate-400">{subtext}</span>}
          {trend && (
            <span className={`font-semibold px-2 py-0.5 rounded-full ${
              trendPositive
                ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                : 'bg-rose-50 text-rose-700 dark:bg-rose-950 dark:text-rose-300'
            }`}>
              {trend}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
