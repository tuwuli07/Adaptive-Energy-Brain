import React, { useState, useRef, useEffect } from 'react';
import { HelpCircle, X, ExternalLink } from 'lucide-react';

/**
 * InfoTooltip - Layer 2 Explanation Component
 * Provides clean, interactive "What this means", "Why it matters", and "How to interpret"
 * popover for complex energy and optimization metrics without cluttering the primary view.
 */
export default function InfoTooltip({
  title,
  what,
  why,
  how,
  technical,
  link,
  linkText = 'Learn more in Research',
  align = 'left',
}) {
  const [isOpen, setIsOpen] = useState(false);
  const popoverRef = useRef(null);

  // Close on outside click or Escape key
  useEffect(() => {
    function handleClickOutside(event) {
      if (popoverRef.current && !popoverRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    function handleKeyDown(event) {
      if (event.key === 'Escape') setIsOpen(false);
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  return (
    <div className="relative inline-flex items-center" ref={popoverRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label={`Learn more about ${title || 'this metric'}`}
        className="p-1 text-slate-400 hover:text-blue-500 dark:hover:text-blue-400 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>

      {isOpen && (
        <div
          className={`absolute top-full mt-2 z-50 w-72 sm:w-80 p-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-xl shadow-slate-900/10 dark:shadow-black/40 text-xs backdrop-blur-md animate-in fade-in zoom-in-95 duration-150 ${
            align === 'right' ? 'right-0' : 'left-0 sm:left-auto'
          }`}
        >
          {/* Header */}
          <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-slate-100 dark:border-slate-800">
            <span className="font-bold text-slate-900 dark:text-slate-100 text-xs tracking-tight">
              {title || 'Metric Explanation'}
            </span>
            <button
              aria-label="Close explanation"
              onClick={() => setIsOpen(false)}
              className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Body Sections */}
          <div className="space-y-2.5 text-slate-600 dark:text-slate-300">
            {what && (
              <div>
                <span className="font-semibold text-blue-600 dark:text-blue-400 block mb-0.5">
                  What this means
                </span>
                <p className="leading-relaxed">{what}</p>
              </div>
            )}

            {why && (
              <div>
                <span className="font-semibold text-cyan-600 dark:text-cyan-400 block mb-0.5">
                  Why it matters
                </span>
                <p className="leading-relaxed">{why}</p>
              </div>
            )}

            {how && (
              <div>
                <span className="font-semibold text-amber-600 dark:text-amber-400 block mb-0.5">
                  How to interpret
                </span>
                <p className="leading-relaxed">{how}</p>
              </div>
            )}

            {technical && (
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400">
                <span className="font-semibold text-slate-700 dark:text-slate-300 block mb-0.5">
                  Technical Definition:
                </span>
                <code className="bg-slate-100 dark:bg-slate-800/80 px-1 py-0.5 rounded text-[10px] font-mono block overflow-x-auto text-slate-800 dark:text-slate-200">
                  {technical}
                </code>
              </div>
            )}

            {link && (
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex justify-end">
                <a
                  href={link}
                  className="inline-flex items-center gap-1 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline"
                >
                  <span>{linkText}</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
