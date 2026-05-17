import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function MessageBubble({ message, index }) {
  const isUser = message.sender === "user";
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const sources = message.sources || [];

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.4,
        ease: [0.25, 0.1, 0, 1],
        delay: Math.min(index * 0.03, 0.15),
      }}
      className={`group flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}
    >
      {/* avatar */}
      <motion.div
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.1, type: "spring", damping: 15, stiffness: 300 }}
        className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 text-[11px] font-bold shadow-lg ${
          isUser
            ? "bg-accent text-white shadow-accent/20"
            : "bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-purple-500/20"
        }`}
      >
        {isUser ? "U" : "AI"}
      </motion.div>

      {/* bubble */}
      <div className="relative max-w-[75%]">
        <div
          className={`msg-content whitespace-pre-wrap break-words text-[14px] leading-[1.7] px-4 py-3 ${
            isUser
              ? "bg-accent/90 rounded-2xl rounded-tr-lg text-white"
              : "glass rounded-2xl rounded-tl-lg text-white/85"
          }`}
        >
          {message.text}
        </div>

        {/* actions for bot messages */}
        {!isUser && (
          <div className="flex items-center gap-1 mt-1.5 ml-1">
            {sources.length > 0 && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center gap-1 text-[11px] text-white/25 hover:text-white/50 transition-colors px-2 py-1 rounded-md hover:bg-white/[0.04]"
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                {showSources ? "Hide" : "Show"} {sources.length} sources
              </button>
            )}

            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleCopy}
              className="text-white/20 hover:text-white/50 transition-colors p-1 rounded-md hover:bg-white/[0.04]"
              title="Copy"
            >
              {copied ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round"><path d="M20 6L9 17l-5-5"/></svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
              )}
            </motion.button>
          </div>
        )}

        {/* sources panel */}
        <AnimatePresence>
          {showSources && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden mt-2"
            >
              <div className="space-y-2">
                {sources.map((s, i) => (
                  <div
                    key={i}
                    className="glass rounded-xl px-3 py-2.5 border border-white/[0.04]"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-medium text-accent/70 bg-accent/10 px-1.5 py-0.5 rounded">
                        #{i + 1}
                      </span>
                      <span className="text-[10px] text-white/20">
                        score: {(s.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-[12px] text-white/40 leading-relaxed line-clamp-4">
                      {s.text}
                    </p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
