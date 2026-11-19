import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Bot, X, Minus, Send, ChevronDown, MessageCircle, Loader2 } from 'lucide-react';
import { useDataSource } from '../context/DataSourceContext';
import { api } from '../api/client';

export const AssistantContext = createContext(() => {});
export const useAssistant = () => useContext(AssistantContext);
export default function FloatingAssistant({ open, setOpen }) {
  const { source, activeMeta } = useDataSource();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [provider, setProvider] = useState(() => localStorage.getItem('energy-assistant-provider') || 'ollama');
  const [model, setModel] = useState('');
  const [research, setResearch] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);
  const streamRef = useRef(null);
  const triggerRef = useRef(null);
  const previousFocus = useRef(null);
  const close = () => { setOpen(false); (previousFocus.current?.isConnected ? previousFocus.current : triggerRef.current)?.focus(); };
  useEffect(() => {
    if (!open) return;
    previousFocus.current = document.activeElement;
    inputRef.current?.focus({ preventScroll: true });
    const escape = e => { if (e.key === 'Escape') { e.preventDefault(); close(); } };
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [open]);
  const refreshStatus = () => api.getAssistantStatus().then(result => {
    setStatus(result);
    if (!localStorage.getItem('energy-assistant-provider')) setProvider(result.default_provider || 'ollama');
    setModel(current => result.ollama?.models?.includes(current) ? current : result.ollama?.models?.includes(result.ollama?.model) ? result.ollama.model : result.ollama?.models?.[0] || '');
  }).catch(() => setStatus({ unavailable: true }));
  useEffect(() => { if (open && !status) refreshStatus(); }, [open, status]);

  useEffect(() => { if (open && streamRef.current) streamRef.current.scrollTop = streamRef.current.scrollHeight; }, [messages, busy, open]);
  const send = async (question = input) => {
    const text = question.trim();
    if (!text || busy) return;
    setError(''); setBusy(true); setInput('');
    const history = messages.slice(-6).map(m => ({ role: m.role === 'user' ? 'user' : 'model', content: m.text }));
    setMessages(prev => [...prev, { role: 'user', text }]);
    try {
      const response = await api.sendAssistantMessage(text, history, { provider, model: provider === 'ollama' ? model : undefined, dataset_id: source !== 'none' ? source : undefined, research });
      if (!response.answer) throw new Error('The assistant did not return an answer. Please try again.');
      setMessages(prev => [...prev, { role: 'assistant', text: response.answer, sources: response.sources || [], provider: response.provider, model: response.model }]);
    } catch (e) { setError(e.message || 'The assistant is unavailable. Please try again.'); }
    finally { setBusy(false); inputRef.current?.focus({ preventScroll: true }); }
  };
  return <>
    <section id="energy-assistant-window" hidden={!open} className="floating-assistant" role="dialog" aria-modal="false" aria-labelledby="floating-assistant-title">
      <header className="assistant-titlebar"><div className="assistant-avatar"><Bot size={21} /></div><div><h2 id="floating-assistant-title">Energy assistant</h2><p>{!status ? 'Checking availability...' : status.unavailable ? 'Service unavailable' : provider === 'ollama' ? status.ollama?.available ? 'Local Ollama' : 'Ollama not connected' : status.gemini_configured ? 'Gemini available' : 'Gemini not configured'}</p></div><button aria-label="Minimize assistant" onClick={close}><Minus size={18} /></button><button aria-label="Close assistant" onClick={close}><X size={18} /></button></header>
      <div className="assistant-provider-controls">
        <label>Assistant<select aria-label="Assistant provider" disabled={busy} value={provider} onChange={e => { setProvider(e.target.value); localStorage.setItem('energy-assistant-provider', e.target.value); }}><option value="ollama">Ollama / local</option><option value="gemini">Gemini / cloud</option></select></label>
        {provider === 'ollama' && <label>Local model<select aria-label="Local Ollama model" disabled={busy} value={model} onChange={e => setModel(e.target.value)}>{!model && <option value="">No model installed</option>}{(status?.ollama?.models || []).map(name => <option key={name}>{name}</option>)}</select></label>}
        <button type="button" aria-label="Refresh assistant connection" onClick={refreshStatus} disabled={busy}>Refresh</button>
      </div>
      <div className="assistant-scope">{provider === 'ollama' ? 'Local processing. No automatic cloud fallback.' : 'Cloud processing: questions and any active dataset summary are sent to Gemini.'} {activeMeta ? `Active data: ${activeMeta.name}` : 'No site data selected; general guidance only.'}</div>
      <label className="assistant-research-toggle"><input type="checkbox" checked={research} disabled={busy} onChange={e => setResearch(e.target.checked)} />Include published research references</label>
      {provider === 'ollama' && status && !status.ollama?.available && <div className="assistant-upload-notice">Start Ollama on this computer, then refresh. Install a model with Ollama if none is listed.</div>}
      <div ref={streamRef} className="assistant-stream" role="log" aria-label="Conversation" aria-live="polite" aria-relevant="additions text">
        {!messages.length && <div className="assistant-welcome"><MessageCircle size={28} /><h3>Make sense of your energy.</h3><p>Ask about energy flows, battery decisions, or the results behind this project.</p><div className="assistant-prompts">{['How does the energy system work?', 'Why does battery charge matter?', 'How can I reduce peak energy use?'].map(q => <button disabled={busy} key={q} onClick={() => send(q)}>{q}</button>)}</div></div>}
        {messages.map((m, i) => <article className={`assistant-message ${m.role}`} key={i}><span className="sr-only">{m.role === 'user' ? 'You' : 'Energy assistant'}:</span><div className="whitespace-pre-wrap break-words">{m.text}</div>{m.provider && <small>{m.provider === 'calculated' ? 'Verified calculation' : m.provider === 'ollama' ? 'Ollama / local' : 'Gemini / cloud'}{m.model ? ` / ${m.model}` : ''}</small>}{m.sources?.length > 0 && <details><summary>{m.sources.length} sources <ChevronDown size={12} /></summary><ul>{m.sources.map((s,j) => <li key={j}>{s.title || 'Project reference'}</li>)}</ul></details>}</article>)}
        {busy && <div className="assistant-thinking" role="status"><Loader2 size={15} className="animate-spin" />Finding relevant information...</div>}
      </div>
      {error && <div className="assistant-error" role="alert">{error}</div>}
      <form className="assistant-composer" onSubmit={e => { e.preventDefault(); send(); }}><label className="sr-only" htmlFor="floating-question">Your energy question</label><input ref={inputRef} id="floating-question" value={input} onChange={e => setInput(e.target.value)} disabled={busy} placeholder="Ask an energy question..." maxLength={4000} /><button aria-label="Send message" disabled={busy || !input.trim()}><Send size={18} /></button></form>
    </section>
    <button ref={triggerRef} className="assistant-launcher" onClick={() => open ? close() : setOpen(true)} aria-label={open ? 'Minimize energy assistant' : 'Open energy assistant'} aria-expanded={open} aria-controls="energy-assistant-window"><Bot size={23} /><span>Ask energy assistant</span>{open && <ChevronDown size={15} />}</button>
  </>;
}
