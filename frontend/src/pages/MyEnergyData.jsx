import React, { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { UploadCloud, FileText, CheckCircle2 } from 'lucide-react';
import { useDataSource } from '../context/DataSourceContext';
import { datasetRequest, uploadDataset } from '../api/datasets';
import { validateUploadFile } from '../utils/dataPolicy';
import { Compatibility } from '../components/SourceBoundary';
import { amount } from '../components/EnergyVisuals';

const FIELDS = [
  ['timestamp', 'Timestamp', ['ISO8601']], ['demand_kw', 'Electricity demand', ['kW', 'W', 'MW']],
  ['solar_kw', 'Solar generation', ['kW', 'W', 'MW']], ['wind_kw', 'Wind generation', ['kW', 'W', 'MW']],
  ['grid_import_kw', 'Grid import', ['kW', 'W', 'MW']], ['grid_export_kw', 'Grid export', ['kW', 'W', 'MW']],
  ['price', 'Electricity price', ['USD/kWh']], ['battery_soc', 'Battery charge', ['fraction', '%']],
];
export default function MyEnergyData() {
  const { datasets, storage, refresh, source, setSource } = useDataSource();
  const [dataset, setDataset] = useState(null);
  const [mapping, setMapping] = useState({});
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [utc, setUtc] = useState(false);
  const [conversions, setConversions] = useState(false);
  const [origin, setOrigin] = useState('');
  const fileRef = useRef();
  const choose = async file => {
    const problem = validateUploadFile(file);
    if (problem) { setError(new Error(problem)); return; }
    setBusy(true); setError(null); setProgress(0); setOrigin('');
    try {
      const result = await uploadDataset(file, setProgress);
      setDataset(result); setMapping(result.mapping); setUtc(false); setConversions(false);
      await refresh();
    } catch (e) { setError(e); }
    finally { setBusy(false); if (fileRef.current) fileRef.current.value = ''; }
  };
  const load = async id => {
    if (!id) return;
    setBusy(true); setError(null); setOrigin('');
    try { const result = await datasetRequest(`/${id}`); setDataset(result); setMapping(result.mapping); setUtc(result.assume_utc || false); setConversions(false); }
    catch (e) { setError(e); }
    finally { setBusy(false); }
  };
  const apply = async e => {
    e.preventDefault(); setBusy(true); setError(null);
    try {
      const result = await datasetRequest(`/${dataset.id}/mapping`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mapping, assume_utc: utc, confirm_conversions: conversions }) });
      setDataset(result); await refresh(); setOrigin('Mapping validated. Your dataset is ready to activate.');
    } catch (e) { setError(e); }
    finally { setBusy(false); }
  };
  const change = (field, property, value) => {
    setMapping(prev => ({ ...prev, [field]: { ...prev[field], [property]: value } }));
    setDataset(prev => ({ ...prev, status: 'needs_mapping' })); setOrigin('');
  };
  const sample = 'data:text/csv;charset=utf-8,' + encodeURIComponent('timestamp,demand_kw,solar_kw,wind_kw,battery_soc\n2026-01-01T00:00:00Z,10,0,2,0.5\n2026-01-01T00:30:00Z,20,0,3,0.55\n2026-01-01T01:00:00Z,30,0,4,0.6\n');
  return <div className="space-y-7 data-page"><section className="overview-heading"><div><span className="eyebrow">YOUR WORKSPACE</span><h1>Bring your energy into focus.</h1><p>Upload a CSV, check its measurements, and explore what your data can tell you.</p></div><a href={sample} download="energy-example.csv" className="button-secondary">Download example CSV</a></section><div className="notice"><strong>{storage?.mode === 'configured_volume' ? 'Configured storage' : 'Temporary storage'}</strong><p>{storage?.message || 'Uploads are temporary unless a persistent storage volume is configured. Keep your original CSV.'}</p></div><div className="data-intake-grid"><section className={`surface upload-zone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); if (!busy) { if (e.dataTransfer.files.length !== 1) setError(new Error('Choose one CSV file at a time.')); else choose(e.dataTransfer.files[0]); } }}><UploadCloud size={38} strokeWidth={1.4} /><h2>Drop your energy CSV here</h2><p>UTF-8 CSV only · Up to 10 MiB · 100,000 records</p><p className="caption">A timestamp and at least one power measurement are required. Optional measurements can stay unmapped.</p><input ref={fileRef} id="csv-file" type="file" accept=".csv,text/csv" className="sr-only" aria-label="Choose energy CSV" disabled={busy} onChange={e => e.target.files[0] && choose(e.target.files[0])} /><button className="button-primary" disabled={busy} onClick={() => fileRef.current.click()}>Browse files</button>{busy && <div className="w-full max-w-sm" role="status"><progress max="100" value={progress} className="w-full" /><p>{progress < 100 ? `Uploading ${progress}%` : 'Checking your dataset...'}</p></div>}</section><aside className="surface p-6 space-y-5"><span className="eyebrow">YOUR LIBRARY</span><h2>Uploaded datasets</h2><label className="block text-sm" htmlFor="dataset-library">Preview a dataset</label><select id="dataset-library" className="field w-full" disabled={busy} value={dataset?.id || ''} onChange={e => load(e.target.value)}><option value="">Select a dataset</option>{datasets.map(d => <option key={d.id} value={d.id}>{d.name} / {d.status === 'ready' ? 'Ready' : 'Needs mapping'}</option>)}</select>{!datasets.length && <p className="caption">Your uploaded files will appear here. You can return to a dataset while storage is available.</p>}<div className="text-sm space-y-3 border-t border-slate-200 dark:border-slate-700 pt-4"><p><strong>1.</strong> Upload your CSV</p><p><strong>2.</strong> Map columns and confirm units</p><p><strong>3.</strong> Activate and explore</p></div></aside></div>{error && <div className="error-panel" role="alert"><strong>{error.message}</strong>{error.validation?.duplicate_timestamps > 0 && <p>{error.validation.duplicate_timestamps} duplicate timestamps found.</p>}{error.validation?.errors?.map((e,i) => <p key={i}>Row {e.row}, {e.field}: {e.message}</p>)}</div>}{origin && <p role="status" className="notice">{origin}</p>}{dataset && <><section className="surface p-6 space-y-5"><div className="flex flex-wrap justify-between gap-4"><div><span className="eyebrow">DATASET PREVIEW</span><h2 className="break-all">{dataset.name}</h2><p className="caption">{dataset.record_count.toLocaleString()} records · Original CSV preserved</p></div><FileText size={26} className="text-blue-500" /></div><div className="table-scroll"><table className="data-table"><thead><tr>{dataset.columns.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{dataset.preview.map((r,i) => <tr key={i}>{dataset.columns.map(c => <td key={c}>{r[c] || <span className="text-slate-400">Missing</span>}</td>)}</tr>)}</tbody></table></div><p className="caption">First {dataset.preview.length} rows. Displayed as plain text; uploaded content is never executed.</p></section><form onSubmit={apply} className="surface p-6 space-y-5"><div><span className="eyebrow">COLUMN MAPPING</span><h2>Tell us what each measurement means</h2><p className="caption">Map a timestamp and at least one power column. kWh is energy and cannot be selected as kW. Prices currently support USD/kWh only.</p></div><div className="mapping-grid">{FIELDS.map(([key,label,units]) => <div className="mapping-row" key={key}><span>{label}<small>{key === 'timestamp' ? 'Required' : 'Optional'}</small></span><label className="sr-only" htmlFor={`column-${key}`}>{label} column</label><select className="field" id={`column-${key}`} value={mapping[key]?.column || ''} onChange={e => change(key, 'column', e.target.value)} disabled={busy}><option value="">Not provided</option>{dataset.columns.map(c => <option key={c}>{c}</option>)}</select><label className="sr-only" htmlFor={`unit-${key}`}>{label} unit</label><select className="field" id={`unit-${key}`} disabled={!mapping[key]?.column || busy} value={mapping[key]?.unit || ''} onChange={e => change(key, 'unit', e.target.value)}><option value="">Select unit</option>{units.map(u => <option key={u}>{u}</option>)}</select></div>)}</div><label className="check-label"><input type="checkbox" checked={utc} onChange={e => { setUtc(e.target.checked); setDataset(d => ({ ...d, status: 'needs_mapping' })); }} />Timestamps without an offset represent UTC. (Leave unchecked if unsure.)</label><label className="check-label"><input type="checkbox" checked={conversions} onChange={e => setConversions(e.target.checked)} />I confirm any selected conversions: W ÷ 1,000 → kW; MW × 1,000 → kW; % ÷ 100 → fraction.</label><p className="caption">Power is interpreted as an interval-start average until the next timestamp. The final row has no inferred duration. Missing values are never filled.</p><button className="button-primary" disabled={busy}>Validate & apply mapping</button></form>{dataset.status === 'ready' && dataset.summary && <section className="surface p-6 space-y-5"><div className="flex items-center gap-3"><CheckCircle2 className="text-blue-500" /><div><h2>{dataset.summary.quality}</h2><p className="caption">Ready means valid timestamps, unique times, supported units, and finite mapped values. Warnings flag missing values or irregular sampling.</p></div></div><dl className="quality-grid"><div><dt>Time coverage</dt><dd>{dataset.summary.start}<br />{dataset.summary.end}</dd></div><div><dt>Typical interval</dt><dd>{amount(dataset.summary.sampling_minutes, 'minutes')}</dd></div><div><dt>Missing measurements</dt><dd>{Object.values(dataset.summary.missing_values).reduce((a,b) => a+b,0)}</dd></div><div><dt>Duplicate timestamps</dt><dd>0</dd></div></dl><p className="caption">Available measurements: {dataset.summary.measurements.map(k => FIELDS.find(f => f[0] === k)?.[1]).join(', ')}.</p>{dataset.summary.warnings.map(w => <p key={w} className="caption">{w}</p>)}<div className="flex flex-wrap gap-3"><button className="button-primary" onClick={() => { setSource(dataset.id); setOrigin('Uploaded data is active. Overview and Energy use now use this dataset.'); }}>{source === dataset.id ? 'Uploaded data is active' : 'Use this dataset'}</button><Link to="/energy" className="button-secondary" onClick={() => setSource(dataset.id)}>Open energy analysis</Link></div><details className="advanced"><summary>What can I do with this dataset?</summary><p className="py-4">{dataset.compatibility.explanation}</p><Compatibility checks={dataset.compatibility.checks} /></details></section>}</>}</div>;
}
