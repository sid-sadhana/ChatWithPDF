import React, { useState, useRef } from "react";
import { motion } from "framer-motion";

export function EmptyState({ onUpload, uploading }) {
  const [dragOver, setDragOver] = useState(false);
  const dropRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type === "application/pdf") {
      const input = document.createElement("input");
      input.type = "file";
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
      onUpload(file);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="flex-1 flex items-center justify-center relative"
    >
      {/* background grid */}
      <div className="absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.3) 1px, transparent 0)',
        backgroundSize: '40px 40px',
      }} />

      <div className="text-center max-w-lg px-8 relative z-10">
        {/* animated icon */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.6, ease: [0.25, 0.1, 0, 1] }}
          className="mx-auto mb-8"
        >
          <div className="relative w-24 h-24 mx-auto">
            <motion.div
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="w-24 h-24 rounded-2xl bg-gradient-to-br from-accent/20 to-purple-500/20 border border-white/[0.06] flex items-center justify-center backdrop-blur-sm"
            >
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                <defs>
                  <linearGradient id="icon-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#818cf8" />
                    <stop offset="100%" stopColor="#a78bfa" />
                  </linearGradient>
                </defs>
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke="url(#icon-grad)" />
                <polyline points="14 2 14 8 20 8" stroke="url(#icon-grad)" />
                <line x1="16" y1="13" x2="8" y2="13" stroke="url(#icon-grad)" />
                <line x1="16" y1="17" x2="8" y2="17" stroke="url(#icon-grad)" />
                <line x1="10" y1="9" x2="8" y2="9" stroke="url(#icon-grad)" />
              </svg>
            </motion.div>
            {/* glow */}
            <div className="absolute inset-0 rounded-2xl bg-accent/10 blur-2xl" />
          </div>
        </motion.div>

        <motion.h2
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="text-2xl font-semibold mb-3 tracking-tight"
        >
          Chat with any PDF
        </motion.h2>

        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-white/30 text-[14px] leading-relaxed mb-10 max-w-sm mx-auto"
        >
          Upload a document and ask questions. The AI agent searches, retrieves context, and generates accurate answers.
        </motion.p>

        {/* drop zone + button */}
        <motion.div
          ref={dropRef}
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`drop-zone border-2 border-dashed rounded-2xl p-8 mb-10 transition-all ${
            dragOver
              ? "active border-accent/50 bg-accent/[0.04]"
              : "border-white/[0.06] hover:border-white/[0.1]"
          }`}
        >
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={onUpload}
            disabled={uploading}
            className="inline-flex items-center gap-2.5 bg-gradient-to-r from-accent to-purple-500 hover:from-accent-hover hover:to-purple-400 disabled:opacity-40 text-white text-sm font-medium px-7 py-3 rounded-xl shadow-xl shadow-accent/20 transition-all"
          >
            {uploading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.2"/><path d="M12 2a10 10 0 019.8 7.8" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/></svg>
                Processing...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Upload PDF
              </>
            )}
          </motion.button>
          <p className="text-[12px] text-white/15 mt-4">or drag and drop a file here</p>
        </motion.div>

        {/* feature pills */}
        <motion.div
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="flex items-center justify-center gap-3 flex-wrap"
        >
          {[
            { label: "RAG", color: "emerald" },
            { label: "Pinecone", color: "blue" },
            { label: "Ollama", color: "violet" },
          ].map((f) => (
            <span
              key={f.label}
              className="flex items-center gap-1.5 text-[11px] text-white/20 bg-white/[0.03] border border-white/[0.04] px-3 py-1.5 rounded-full"
            >
              <span className={`w-1.5 h-1.5 rounded-full bg-${f.color}-500/50`} />
              {f.label}
            </span>
          ))}
        </motion.div>
      </div>
    </motion.div>
  );
}
