import React, { createContext, useContext, useEffect, useState } from 'react';
import { datasetRequest } from '../api/datasets';
const Context = createContext(null);
export const useDataSource = () => useContext(Context);

export function DataSourceProvider({ children }) {
  const [revision, setRevision] = useState(0);
  const [source, setSource] = useState(() => (localStorage.getItem('energy-source') && localStorage.getItem('energy-source') !== 'campus' ? localStorage.getItem('energy-source') : 'none'));
  const [datasets, setDatasets] = useState([]);
  const [storage, setStorage] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [activeMeta, setActiveMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  async function refresh() {
    const result = await datasetRequest();
    setDatasets(result.datasets);
    setStorage(result.storage);
    setRevision(r => r + 1);
    return result;
  }
  useEffect(() => { refresh().catch(() => {}); }, []);
  useEffect(() => {
    localStorage.setItem('energy-source', source);
    setAnalysis(null); setActiveMeta(null); setError('');
    if (source === 'none') { setLoading(false); return; }
    let cancelled = false;
    setLoading(true);
    Promise.all([datasetRequest(`/${source}/analysis`), datasetRequest(`/${source}`)])
      .then(([result, meta]) => { if (!cancelled) { setAnalysis(result); setActiveMeta(meta); } })
      .catch(e => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [source, revision]);
  return <Context.Provider value={{ source, setSource, datasets, storage, analysis, activeMeta, loading, error, refresh }}>{children}</Context.Provider>;
}
