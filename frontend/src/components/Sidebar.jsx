import React from "react";
import { motion, AnimatePresence } from "framer-motion";

function timeAgo(iso) {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "now";
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

export function Sidebar({ conversations, activeId, onSelect, onNew, onDelete, onDeleteAll, uploading }) {
  return (
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0, 1] }}
      className="w-[280px] flex flex-col bg-surface-1/80 backdrop-blur-xl border-r border-white/[0.04] relative z-10"
    >
      {/* header */}
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-purple-500 flex items-center justify-center shadow-lg shadow-accent/20">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          </div>
          <div>
            <h1 className="text-[15px] font-semibold tracking-tight">ChatWithPDF</h1>
            <p className="text-[11px] text-white/25 font-medium">Agentic RAG</p>
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.97 }}
          onClick={onNew}
          disabled={uploading}
          className="w-full flex items-center justify-center gap-2.5 bg-accent/90 hover:bg-accent disabled:opacity-40 transition-colors px-4 py-2.5 rounded-xl text-[13px] font-medium shadow-lg shadow-accent/10"
        >
          {uploading ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.2"/><path d="M12 2a10 10 0 019.8 7.8" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/></svg>
              Indexing...
            </>
          ) : (
            <>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 5v14M5 12h14"/></svg>
              New Chat
            </>
          )}
        </motion.button>
      </div>

      <div className="px-4 mb-2">
        <p className="text-[10px] font-semibold text-white/20 uppercase tracking-widest px-1">Conversations</p>
      </div>

      {/* list */}
      <div className="flex-1 overflow-y-auto px-3 pb-4">
        <AnimatePresence initial={false}>
          {conversations.map((c, i) => {
            const active = activeId === c.conversation_id;
            return (
              <motion.div
                key={c.conversation_id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -40, transition: { duration: 0.2 } }}
                transition={{ delay: i * 0.03, duration: 0.3, ease: [0.25, 0.1, 0, 1] }}
                onClick={() => onSelect(c.conversation_id)}
                className={`group relative flex items-center gap-3 px-3 py-3 rounded-xl cursor-pointer transition-all duration-200 mb-0.5 ${
                  active
                    ? "bg-white/[0.07]"
                    : "hover:bg-white/[0.03]"
                }`}
              >
                {/* active indicator */}
                {active && (
                  <motion.div
                    layoutId="active-indicator"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-accent"
                    transition={{ type: "spring", damping: 25, stiffness: 300 }}
                  />
                )}

                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors ${
                  active ? "bg-accent/15 text-accent" : "bg-white/[0.04] text-white/30"
                }`}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                </div>

                <div className="flex-1 min-w-0">
                  <p className={`text-[13px] truncate leading-tight transition-colors ${active ? "text-white font-medium" : "text-white/60"}`}>
                    {c.title || "Untitled"}
                  </p>
                  <p className="text-[11px] text-white/20 mt-0.5 truncate">{c.pdf_filename}</p>
                </div>

                <span className="text-[10px] text-white/15 flex-shrink-0">{timeAgo(c.updated_at)}</span>

                <motion.button
                  whileHover={{ scale: 1.15 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={(e) => { e.stopPropagation(); onDelete(c.conversation_id); }}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-white/20 hover:text-red-400 p-1 rounded-md hover:bg-red-500/10"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </motion.button>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {conversations.length === 0 && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-[12px] text-white/15 text-center mt-12 px-4 leading-relaxed"
          >
            No conversations yet.
            <br />
            Upload a PDF to start.
          </motion.p>
        )}
      </div>

      {/* footer */}
      <div className="px-5 py-4 border-t border-white/[0.03] space-y-3">
        {conversations.length > 0 && (
          <button
            onClick={onDeleteAll}
            className="w-full flex items-center justify-center gap-2 text-[11px] text-red-400/60 hover:text-red-400 hover:bg-red-500/[0.06] transition-colors px-3 py-2 rounded-lg border border-red-500/[0.08] hover:border-red-500/20"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>
            Clear All Data
          </button>
        )}
        <div className="flex items-center gap-2 text-[10px] text-white/15">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-glow-pulse" />
          Ollama Connected
        </div>
      </div>
    </motion.aside>
  );
}
