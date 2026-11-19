/**
 * Standard utility formatters for energy and economic metrics.
 */

export function formatCurrency(val) {
  if (val === null || val === undefined || isNaN(val)) return '$0.00';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(val);
}

export function formatNumber(val, decimals = 1) {
  if (val === null || val === undefined || isNaN(val)) return '0';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(val);
}

export function formatEnergy(valKwh) {
  if (valKwh === null || valKwh === undefined || isNaN(valKwh)) return '0 kWh';
  if (Math.abs(valKwh) >= 1000) {
    return `${(valKwh / 1000).toFixed(1)} MWh`;
  }
  return `${valKwh.toFixed(1)} kWh`;
}

export function formatPercent(val) {
  if (val === null || val === undefined || isNaN(val)) return '0.00%';
  const prefix = val > 0 ? '+' : '';
  return `${prefix}${val.toFixed(2)}%`;
}

export function formatDate(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'UTC',
  }) + ' UTC';
}
