import React, { useState } from 'react';
import { Sun, Wind, Building2, Battery, Zap } from 'lucide-react';

export const amount = (value, unit = '', digits = 1) => Number.isFinite(value) ? `${value.toLocaleString('en-US', { maximumFractionDigits: digits })}${unit ? ` ${unit}` : ''}` : 'Not available';
export function RenewableMix({ solar, wind, ratio, period = 'Selected period' }) {
  const available = Number.isFinite(solar) && Number.isFinite(wind) && solar >= 0 && wind >= 0;
  const total = available ? solar + wind : 0;
  const fraction = total ? solar / total : 0;
  return <section className="surface p-6 renewable-panel"><span className="eyebrow">RENEWABLES</span><h2>Clean energy mix</h2><p className="caption">{period}</p><div className="flex flex-wrap items-center gap-6 mt-5"><div className="donut-wrap"><svg viewBox="0 0 120 120" role="img" aria-label={total ? `Solar ${Math.round(fraction * 100)}%, wind ${Math.round((1 - fraction) * 100)}% of renewable generation` : 'No complete renewable mix available'}><circle cx="60" cy="60" r="48" fill="none" stroke="currentColor" className="text-slate-200 dark:text-slate-700" strokeWidth="12" />{total > 0 && <><circle cx="60" cy="60" r="48" fill="none" stroke="#3774d7" strokeWidth="12" /><circle cx="60" cy="60" r="48" fill="none" stroke="#F4B84A" strokeWidth="12" strokeDasharray={`${fraction * 301.59} 301.59`} transform="rotate(-90 60 60)" /></>}</svg><div><strong>{total ? amount(total, '', 0) : '--'}</strong><span>kWh generated</span></div></div><dl className="space-y-4 text-sm"><div><dt className="flex items-center gap-2"><i className="legend-dot bg-[#F4B84A]" />Solar</dt><dd className="font-semibold mt-1">{amount(solar, 'kWh')}</dd></div><div><dt className="flex items-center gap-2"><i className="legend-dot bg-[#3774d7]" />Wind</dt><dd className="font-semibold mt-1">{amount(wind, 'kWh')}</dd></div></dl></div><p className="caption mt-5">The ring divides renewable generation only. It is not a share of electricity consumed.</p>{Number.isFinite(ratio) && <p className="mt-3 text-sm">Renewable output equals <strong>{amount(ratio, '%')}</strong> of demand over the same interval. Exports and storage mean this is not self-consumption.</p>}</section>;
}

export function EnergySnapshot({ record = {}, capacity = null, safeRange = null }) {
  const battery = record.battery_power_kw;
  const soc = record.battery_soc;
  const nodes = [
    { name: 'Solar', field: 'solar_kw', icon: Sun, position: 'solar', direction: 'in' },
    { name: 'Wind', field: 'wind_kw', icon: Wind, position: 'wind', direction: 'in' },
    { name: 'Buildings', field: 'demand_kw', icon: Building2, position: 'buildings', direction: 'out' },
    { name: 'Grid import', field: 'grid_import_kw', icon: Zap, position: 'grid', direction: 'in' },
    { name: 'Grid export', field: 'grid_export_kw', icon: Zap, position: 'export', direction: 'out' },
    { name: 'Battery', field: 'battery_power_kw', icon: Battery, position: 'battery', direction: battery > 0 ? 'in' : 'out' },
  ];
  return <section className="surface flow-panel p-5 sm:p-7"><div className="flex flex-wrap justify-between gap-3"><div><span className="eyebrow">ENERGY NETWORK</span><h2>Where power moves</h2></div><span className="caption">{record.timestamp ? new Date(record.timestamp).toLocaleString('en-GB', { timeZone: 'UTC' }) + ' UTC' : 'No timestamp available'}</span></div><div className="energy-network">{nodes.map(({ name, field, icon: Icon, position, direction }) => { const value = record[field]; return <div key={position} className={`flow-node flow-${position}`}><Icon size={20} aria-hidden="true" /><div><span>{name}</span><strong>{amount(Number.isFinite(value) ? Math.abs(value) : null, 'kW')}</strong>{position === 'battery' && <small>{Number.isFinite(battery) ? battery < -.01 ? 'Charging' : battery > .01 ? 'Discharging' : 'Idle' : Number.isFinite(soc) ? `${amount(soc * 100, '%')} charge; power unavailable` : 'Not measured'}</small>}</div><span className="flow-arrow" aria-label={value ? direction === 'in' ? 'To shared network' : 'From shared network' : 'No known flow'}>{Number.isFinite(value) && Math.abs(value) > .01 ? direction === 'in' ? '\u2192' : '\u2190' : '\u2014'}</span></div>; })}<div className="network-hub"><Zap size={25} /><span>Shared<br />energy network</span></div></div><p className="caption">Arrows show measured flow at the shared connection, not how individual sources supply each building. Missing measurements remain unavailable.</p></section>;
}

export function BatteryPortrait({ soc, power, capacity, safeRange }) {
  const known = Number.isFinite(soc);
  const percent = known ? Math.round(soc * 100) : 0;
  return <section className="surface battery-portrait p-6"><span className="eyebrow">STORAGE</span><h2>Battery charge</h2><div className="battery-illustration" role="img" aria-label={known ? `${percent}% charged` : 'Battery charge not measured'}><div style={{ height: `${percent}%` }} /><strong>{known ? `${percent}%` : '--'}</strong></div><p className="text-center font-medium mt-5">{Number.isFinite(power) ? power < -.01 ? 'Charging' : power > .01 ? 'Supplying energy' : 'Idle' : 'Battery power not measured'}</p><dl className="mt-5 space-y-3 text-sm"><div className="flex justify-between gap-2"><dt>Stored energy</dt><dd>{known && capacity ? amount(soc * capacity, 'kWh') : 'Capacity not supplied'}</dd></div><div className="flex justify-between"><dt>Battery power</dt><dd>{amount(Number.isFinite(power) ? Math.abs(power) : null, 'kW')}</dd></div><div className="flex justify-between gap-2"><dt>Safe range</dt><dd>{safeRange || 'Not supplied'}</dd></div></dl></section>;
}

export function ActionTimeline({ records = [], onSelect, selected = 0, field = 'battery_power_kw', label = 'Battery decisions' }) {
  const visible = records.length > 168 ? records.filter((_, i) => i % Math.ceil(records.length / 168) === 0) : records;
  return <section className="surface p-6"><h2>{label}</h2><p className="caption mb-4">{field === 'battery_power_kw' ? 'Charge / discharge / idle. Gray means not measured.' : 'Each segment shows the selected strategy at that hour.'}{records.length > 168 && ' Overview sampled to 168 segments.'}</p><div className="action-timeline">{visible.map((r, i) => { const v = r[field]; const title = `${r.timestamp || r.timestamp_utc || `Hour ${i}`}: ${typeof v === 'string' ? v : Number.isFinite(v) ? `${v.toFixed(1)} kW` : 'Not measured'}`; return <button key={i} title={title} aria-label={title} onClick={() => onSelect?.(records.indexOf(r))} className={typeof v === 'string' ? `strategy-${v.includes('dro') ? 'dro' : v.includes('cvar') ? 'cvar' : 'forecast'}` : !Number.isFinite(v) ? 'unknown' : v < -.01 ? 'charge' : v > .01 ? 'discharge' : 'idle'} />; })}</div><div className="flex flex-wrap gap-4 caption mt-4"><span>Teal: charge / forecast-led</span><span>Amber: discharge / high-cost protection</span><span>Cyan: uncertainty protection</span></div></section>;
}

export function DemandHeatmap({ records = [] }) {
  const days = new Map();
  records.forEach(r => { if (!Number.isFinite(r.demand_kw) || !r.timestamp) return; const d = new Date(r.timestamp); if (Number.isNaN(+d)) return; const day = d.toISOString().slice(0, 10); if (!days.has(day)) days.set(day, Array.from({ length: 24 }, () => [])); days.get(day)[d.getUTCHours()].push(r.demand_kw); });
  const rows = [...days].slice(0, 14);
  const max = Math.max(1, ...records.map(r => r.demand_kw || 0));
  return <section className="surface p-6"><h2>When demand is highest</h2><p className="caption mb-4">Average recorded power by hour (UTC), first 14 days. Darker cells mean higher demand. Blank cells have no measurements.</p>{!rows.length ? <p>No demand measurement is mapped.</p> : <div className="overflow-x-auto"><div className="heatmap-grid"><div /><span style={{ gridColumn: 'span 6' }}>00:00</span><span style={{ gridColumn: 'span 6' }}>06:00</span><span style={{ gridColumn: 'span 6' }}>12:00</span><span style={{ gridColumn: 'span 6' }}>18:00</span>{rows.map(([day, hours]) => <React.Fragment key={day}><span>{day.slice(5)}</span>{hours.map((values, hour) => { const avg = values.length ? values.reduce((a,b) => a+b,0) / values.length : null; return <div key={hour} role="img" aria-label={`${day} ${hour}:00 UTC: ${amount(avg, 'kW')}`} title={`${day} ${hour}:00 UTC: ${amount(avg, 'kW')}`} style={{ background: avg === null ? 'transparent' : `rgba(59, 130, 246, ${.15 + .85 * avg / max})` }} />; })}</React.Fragment>)}</div></div>}</section>;
}
