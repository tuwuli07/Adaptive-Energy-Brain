import React, { useState, useEffect, useRef } from 'react';
import {
  Bot,
  Send,
  Sparkles,
  Database,
  Layers,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Trash2,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  BookOpen,
  Cpu,
  User,
} from 'lucide-react';
import SectionHeader from '../components/SectionHeader';
import { api } from '../api/client';

const SUGGESTED_PROMPTS = [
  { label: 'How it works', text: 'How do solar, wind, batteries, and the grid work together?' },
  { label: 'Compare Strategies', text: 'What are the differences between the energy strategies?' },
  { label: 'Battery health', text: 'Why does battery state of charge matter and how is degradation calculated?' },
  { label: 'Challenging conditions', text: 'What happened during the compound adverse stress scenario?' },
  { label: 'Smart control', text: 'How does the system choose an energy strategy?' },
  { label: 'Research Limits', text: 'What are the scientific limitations of these simulation results?' },
];

export default function AIAssistant() {
  const [messages, setMessages] = useState([
    {
      id: 'msg-init',
      sender: 'assistant',
      text: "Welcome! Ask me about campus energy, battery storage, forecasts, or study results. I can help explain the numbers and compare energy strategies.",
      sources: [
        { title: "README.md (Architecture)", type: "rag_document", source_file: "README.md" },
        { title: "MCP Tool: get_system_overview", type: "mcp_tool", source_file: "src/assistant/mcp_server.py" },
      ],
      data_provenance: "Verified Research Artifacts & Simulation Benchmark v1",
      model: "system-grounded",
      status: "initialized",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});
  const messagesEndRef = useRef(null);

  useEffect(() => {
    async function loadStatus() {
      try {
        const res = await api.getAssistantStatus();
        setStatus(res);
      } catch (err) {
        console.warn('Could not load assistant status', err);
      }
    }
    loadStatus();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const toggleSource = (msgId) => {
    setExpandedSources((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  const handleSend = async (textToSend) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userMessage = {
      id: `usr-${Date.now()}`,
      sender: 'user',
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const history = messages.slice(-4).map((m) => ({
        role: m.sender === 'user' ? 'user' : 'model',
        content: m.text,
      }));

      const res = await api.sendAssistantMessage(text, history);

      const botMessage = {
        id: `bot-${Date.now()}`,
        sender: 'assistant',
        text: res.answer || 'No response received from the assistant.',
        sources: res.sources || [],
        data_provenance: res.data_provenance || 'Verified Research Artifacts & Simulation Benchmark v1',
        model: res.model || 'gemini',
        status: res.status || 'success',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMessage = {
        id: `err-${Date.now()}`,
        sender: 'assistant',
        text: 'The assistant is unavailable right now. Please try again shortly.',
        sources: [],
        data_provenance: 'Error',
        model: 'error',
        status: 'error',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([messages[0]]);
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-8">
      {/* 1. Header & Live Grounding Indicators */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
            Ask Your Energy AI
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
            Clear answers to your questions about energy use, storage, and study results.
          </p>
        </div>

        {/* Live Grounding Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-blue-500/20 bg-blue-500/10 text-xs font-semibold text-blue-700 dark:text-blue-300">
            <Sparkles className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
            <span>{status?.gemini_configured ? 'Assistant available' : 'Saved answers'}</span>
            <span className={`w-2 h-2 rounded-full ${status?.gemini_configured ? 'bg-blue-500 animate-pulse' : 'bg-amber-400'}`} />
          </div>

          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs font-medium text-slate-600 dark:text-slate-300">
            <Database className="w-3.5 h-3.5 text-cyan-500" />
            <span>Project knowledge</span>
          </div>

          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs font-medium text-slate-600 dark:text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-violet-500" />
            <span>Energy insights</span>
          </div>
        </div>
      </div>

      {/* 2. Suggested Quick Prompts Grid */}
      <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md rounded-3xl border border-slate-200/80 dark:border-slate-800/80 p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-blue-500" />
            Recommended Questions
          </span>
          <button
            onClick={clearChat}
            className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 flex items-center gap-1 transition-colors"
            title="Reset conversation"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Chat
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
          {SUGGESTED_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => handleSend(prompt.text)}
              disabled={isLoading}
              className="text-left px-3.5 py-2.5 rounded-2xl bg-slate-50 dark:bg-slate-800/60 hover:bg-blue-50/60 dark:hover:bg-blue-950/30 border border-slate-200/60 dark:border-slate-700/60 hover:border-blue-500/40 text-xs text-slate-700 dark:text-slate-300 hover:text-blue-700 dark:hover:text-blue-300 transition-all flex items-center justify-between group"
            >
              <div className="truncate pr-2">
                <span className="font-bold block truncate">{prompt.label}</span>
                <span className="text-[11px] text-slate-400 truncate block">{prompt.text}</span>
              </div>
              <Sparkles className="w-3.5 h-3.5 text-slate-400 group-hover:text-blue-500 shrink-0 transition-colors" />
            </button>
          ))}
        </div>
      </div>

      {/* 3. Main Conversation Stream */}
      <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md rounded-3xl border border-slate-200/80 dark:border-slate-800/80 shadow-md overflow-hidden flex flex-col h-[600px]">
        {/* Messages Scroll Area */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-5">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.sender === 'assistant' && (
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-sky-400 flex items-center justify-center flex-shrink-0 mt-0.5 shadow-md shadow-blue-500/20 text-white">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div className={`max-w-[88%] sm:max-w-[78%] space-y-2`}>
                {/* Bubble Content */}
                <div
                  className={`p-4 rounded-2xl text-xs sm:text-sm leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-blue-600 text-white rounded-br-none shadow-md'
                      : 'bg-slate-50 dark:bg-slate-800/90 text-slate-800 dark:text-slate-200 border border-slate-200/80 dark:border-slate-700/80 rounded-bl-none shadow-xs'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{msg.text}</div>
                </div>

                {/* Sources & Grounding Metadata (Assistant Only) */}
                {msg.sender === 'assistant' && msg.sources && msg.sources.length > 0 && (
                  <div className="space-y-1.5 pl-1">
                    <button
                      onClick={() => toggleSource(msg.id)}
                      className="text-xs font-semibold text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 flex items-center gap-1 transition-colors"
                    >
                      {expandedSources[msg.id] ? (
                        <ChevronDown className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5" />
                      )}
                      <span>View {msg.sources.length} sources</span>
                    </button>

                    {expandedSources[msg.id] && (
                      <div className="bg-slate-100 dark:bg-slate-950/80 rounded-xl p-3 border border-slate-200 dark:border-slate-800 space-y-1.5 text-xs text-slate-600 dark:text-slate-300">
                        {msg.sources.map((src, sIdx) => (
                          <div
                            key={sIdx}
                            className="flex items-center justify-between gap-2 py-1 border-b border-slate-200/60 dark:border-slate-800 last:border-none"
                          >
                            <div className="flex items-center gap-1.5 truncate">
                              <span
                                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                                  src.type === 'mcp_tool'
                                    ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20'
                                    : 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20'
                                }`}
                              >
                                {src.type === 'mcp_tool' ? 'Energy data' : 'Project guide'}
                              </span>
                              <span className="truncate">{src.title}</span>
                            </div>
                            {src.relevance_score && (
                              <span className="text-[10px] text-slate-400 whitespace-nowrap">
                                Score: {(src.relevance_score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Timestamp & Status */}
                <div
                  className={`text-[10px] text-slate-400 px-1 ${
                    msg.sender === 'user' ? 'text-right' : 'text-left'
                  }`}
                >
                  {msg.timestamp}
                  {msg.status === 'offline_fallback' && (
                    <span className="ml-2 text-amber-500">· Saved answer</span>
                  )}
                </div>
              </div>

              {msg.sender === 'user' && (
                <div className="w-8 h-8 rounded-xl bg-slate-200 dark:bg-slate-700 flex items-center justify-center flex-shrink-0 mt-0.5 text-slate-700 dark:text-slate-200">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3 justify-start items-center">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-sky-400 flex items-center justify-center flex-shrink-0 animate-pulse text-white">
                <Bot className="w-4 h-4" />
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200/80 dark:border-slate-700/60 rounded-bl-none text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-500" />
                <span>Finding relevant energy insights...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 bg-slate-50/80 dark:bg-slate-950/70 border-t border-slate-200/80 dark:border-slate-800/80">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              aria-label="Your energy question"
              placeholder="Ask about energy use, batteries, forecasts, or savings..."
              disabled={isLoading}
              className="flex-1 bg-white dark:bg-slate-900 text-xs sm:text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 px-4 py-3 rounded-2xl border border-slate-200 dark:border-slate-700 focus:outline-hidden focus:ring-2 focus:ring-blue-500 transition-colors"
            />
            <button
              aria-label="Send message"
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="bg-blue-500 hover:bg-blue-400 disabled:bg-slate-200 dark:disabled:bg-slate-800 disabled:text-slate-400 text-slate-950 px-5 py-3 rounded-2xl font-bold text-xs sm:text-sm flex items-center gap-2 transition-all shadow-md shadow-blue-500/20"
            >
              <Send className="w-4 h-4" />
              <span className="hidden sm:inline">Send</span>
            </button>
          </form>
        </div>
      </div>

      {/* 4. Scientific Grounding Disclaimer */}
      <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/60 flex items-start gap-3 text-xs text-slate-500 dark:text-slate-400">
        <ShieldCheck className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <p className="font-bold text-slate-800 dark:text-slate-200">About these answers</p>
          <p className="leading-relaxed">
            Answers draw on project documents and simulation results. These describe a hypothetical campus and do not control real equipment.
          </p>
        </div>
      </div>
    </div>
  );
}
