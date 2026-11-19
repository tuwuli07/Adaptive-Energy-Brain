import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { RenewableMix, EnergySnapshot, DemandHeatmap, amount } from '../components/EnergyVisuals';
import { Skeleton } from '../components/SourceBoundary';
import { integratePower } from '../utils/energyAnalysis';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, Legend } from 'recharts';

export default function EnergyMonitoring() {
  const [controller, setController] = useState('mpc_gnn_transformer');
  const [limit, setLimit] = useState(168);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [hour, setHour] = useState(0);
  useEffect(() => { let cancel = false; setResult(null); setError(''); api.getTimeseries({ controllerId: controller, limit, downsample: 1 }).then(r => { if (!cancel) { setResult(r); setHour(0); } }).catch(() => { if (!cancel) setError('Energy records could not be loaded.'); }); return () => { cancel = true; }; }, [controller, limit]);
  if (error) return <div className="error-panel" role="alert">{error}</div>;
  if (!result) return <Skeleton />;
  const rows = result.data || [];
  const demand = integratePower(rows, 'demand_kw');
  const solar = integratePower(rows, 'solar_kw').energy;
  const wind = integratePower(rows, 'wind_kw').energy;
  return <div className="space-y-6"><section className="overview-heading"><div><span className="eyebrow">ENERGY USE</span><h1>Every source. Every hour.</h1><p>See how site demand, renewable generation, and grid electricity change together.</p></div><div className="flex flex-wrap gap-2"><select aria-label="Energy strategy" className="field" value={controller} onChange={e => setController(e.target.value)}><option value="mpc_gnn_transformer">Forecast-led planning</option><option value="mpc_seasonal_persistence">Seasonal planning</option><option value="zero_action">Idle battery</option><option value="mpc_oracle_rolling_24h">Perfect-knowledge reference</option></select><select className="field" aria-label="Observation window" value={limit} onChange={e => setLimit(+e.target.value)}><option value="24">24 records</option><option value="168">7 days</option><option value="336">14 days</option><option value="1313">Full winter study</option></select></div></section><div className="notice">{result._data_source === 'saved_or_illustrative' ? 'Illustrative sample curves: the study trajectory is unavailable. These charts are not frozen benchmark evidence.' : 'Synthetic research reference. These are not live utility measurements.'} Energy totals integrate from first to last timestamp; the final row has no inferred duration.</div><div className="editorial-grid"><div className="space-y-6"><EnergySnapshot record={rows[hour]} /><div className="surface inspection-control"><label htmlFor="energy-hour">Inspect an hour</label><input id="energy-hour" type="range" min="0" max={Math.max(0, rows.length - 1)} value={hour} onChange={e => setHour(+e.target.value)} /><output>{hour + 1} / {rows.length}</output></div></div><div className="space-y-6"><RenewableMix solar={solar} wind={wind} ratio={demand.energy > 0 ? (solar + wind) / demand.energy * 100 : null} period={`${amount(demand.hours, 'hours')} between displayed timestamps`} /><div className="surface p-6"><span className="eyebrow">CONSUMPTION</span><strong className="block text-3xl mt-3">{amount(demand.energy, 'kWh')}</strong><p className="caption mt-2">Power (kW) measures the rate of use. Energy (kWh) accounts for the time that power is used.</p></div></div></div><DemandHeatmap records={rows} /><section className="surface p-6"><h2>Demand & generation trend</h2><p className="caption">One focused view of power across the selected period (UTC).</p><div className="h-72 mt-5"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rows}><XAxis dataKey="timestamp" tickFormatter={v => new Date(v).toLocaleDateString('en-GB', { timeZone: 'UTC' })} minTickGap={35} fontSize={11} /><YAxis unit=" kW" fontSize={11} /><Tooltip /><Legend />{[['demand_kw','Demand','#1453b8'],['solar_kw','Solar','#F4B84A'],['wind_kw','Wind','#67DCE8']].map(([key,name,color]) => <Area key={key} dataKey={key} name={name} stroke={color} fill={color} fillOpacity={.08} isAnimationActive={false} />)}</AreaChart></ResponsiveContainer></div></section></div>;
}
