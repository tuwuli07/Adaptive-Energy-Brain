import React from 'react';
import { ShieldCheck, Eye } from 'lucide-react';

export default function CausalityBadge({ causality, size = 'sm' }) {
  const isCausal = causality === 'strictly_causal';
  const sizeClasses = size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-3 py-1';

  if (isCausal) {
    return (
      <span className={`inline-flex items-center gap-1 font-medium rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800 ${sizeClasses}`}>
        <ShieldCheck className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
        <span>Uses available information</span>
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center gap-1 font-medium rounded-full bg-purple-50 text-purple-700 dark:bg-purple-950/60 dark:text-purple-300 border border-purple-200 dark:border-purple-800 ${sizeClasses}`}>
      <Eye className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
      <span>Perfect-knowledge reference</span>
    </span>
  );
}
