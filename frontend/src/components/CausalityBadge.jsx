import React from 'react';
import { ShieldCheck, Eye } from 'lucide-react';

export default function CausalityBadge({ causality, size = 'sm' }) {
  const isCausal = causality === 'strictly_causal';
  const sizeClasses = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1';

  if (isCausal) {
    return (
      <span className={`inline-flex items-center gap-1 font-medium rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 ${sizeClasses}`}>
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
        <span>Causal / Deployable</span>
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center gap-1 font-medium rounded-full bg-purple-50 text-purple-700 dark:bg-purple-950/60 dark:text-purple-300 border border-purple-200 dark:border-purple-800 ${sizeClasses}`}>
      <Eye className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
      <span>Noncausal Diagnostic</span>
    </span>
  );
}
