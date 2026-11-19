import React from 'react';
import InfoTooltip from './InfoTooltip';

/**
 * MetricCard - Modern Two-Layer Energy Metric Card
 * Displays friendly high-level information with an integrated Layer-2 InfoTooltip
 * for technical accuracy and depth.
 */
export default function MetricCard({
  title,
  technicalTitle,
  value,
  unit,
  subtext,
  icon: Icon,
  color = 'emerald', // 'emerald' | 'amber' | 'cyan' | 'blue' | 'violet' | 'rose' | 'slate'
  tooltipData, // { what, why, how, technical, link }
  trend,
  trendPositive = true,
  badgeText,
  badgeColor = 'slate',
}) {
  const colorMap = {
    emerald: {
      bg: 'bg-blue-50 dark:bg-blue-950/40',
      text: 'text-blue-600 dark:text-blue-400',
      border: 'border-blue-200 dark:border-blue-800/40',
      glow: 'shadow-blue-500/5',
    },
    amber: {
      bg: 'bg-amber-50 dark:bg-amber-950/40',
      text: 'text-amber-600 dark:text-amber-400',
      border: 'border-amber-200 dark:border-amber-800/40',
      glow: 'shadow-amber-500/5',
    },
    cyan: {
      bg: 'bg-cyan-50 dark:bg-cyan-950/40',
      text: 'text-cyan-600 dark:text-cyan-400',
      border: 'border-cyan-200 dark:border-cyan-800/40',
      glow: 'shadow-cyan-500/5',
    },
    blue: {
      bg: 'bg-blue-50 dark:bg-blue-950/40',
      text: 'text-blue-600 dark:text-blue-400',
      border: 'border-blue-200 dark:border-blue-800/40',
      glow: 'shadow-blue-500/5',
    },
    violet: {
      bg: 'bg-violet-50 dark:bg-violet-950/40',
      text: 'text-violet-600 dark:text-violet-400',
      border: 'border-violet-200 dark:border-violet-800/40',
      glow: 'shadow-violet-500/5',
    },
    rose: {
      bg: 'bg-rose-50 dark:bg-rose-950/40',
      text: 'text-rose-600 dark:text-rose-400',
      border: 'border-rose-200 dark:border-rose-800/40',
      glow: 'shadow-rose-500/5',
    },
    slate: {
      bg: 'bg-slate-100 dark:bg-slate-800',
      text: 'text-slate-600 dark:text-slate-300',
      border: 'border-slate-200 dark:border-slate-700',
      glow: 'shadow-slate-500/5',
    },
  };

  const scheme = colorMap[color] || colorMap.emerald;

  return (
    <div
      className={`metric-card relative bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all duration-200 flex flex-col justify-between ${scheme.glow}`}
    >
      <div>
        {/* Top Header: Title, Technical Subtitle, InfoTooltip, and Icon */}
        <div className="flex items-start justify-between gap-2">
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                {title}
              </span>
              {tooltipData && (
                <InfoTooltip
                  title={title}
                  what={tooltipData.what}
                  why={tooltipData.why}
                  how={tooltipData.how}
                  technical={tooltipData.technical}
                  link={tooltipData.link}
                  linkText={tooltipData.linkText}
                />
              )}
            </div>

          </div>

          {Icon && (
            <div className={`p-2.5 rounded-xl shrink-0 ${scheme.bg} ${scheme.text} border ${scheme.border}`}>
              <Icon className="w-5 h-5" />
            </div>
          )}
        </div>

        {/* Primary Value Display */}
        <div className="mt-3 flex items-baseline gap-1.5">
          <span className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-slate-50">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-semibold text-slate-400 dark:text-slate-500">
              {unit}
            </span>
          )}
        </div>
      </div>

      {/* Footer / Subtext / Trend / Badges */}
      {(subtext || trend || badgeText) && (
        <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-xs gap-2">
          {subtext && (
            <span className="text-slate-500 dark:text-slate-400 break-words">
              {subtext}
            </span>
          )}

          <div className="flex items-center gap-1.5 ml-auto shrink-0">
            {badgeText && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                {badgeText}
              </span>
            )}

            {trend && (
              <span
                className={`font-semibold px-2 py-0.5 rounded-full text-[11px] ${
                  trendPositive
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/80 dark:text-blue-300 border border-blue-200 dark:border-blue-800/50'
                    : 'bg-rose-50 text-rose-700 dark:bg-rose-950/80 dark:text-rose-300 border border-rose-200 dark:border-rose-800/50'
                }`}
              >
                {trend}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
