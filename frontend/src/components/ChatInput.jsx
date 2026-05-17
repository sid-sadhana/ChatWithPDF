import React, { useRef, useEffect } from "react";
import { motion } from "framer-motion";

export function ChatInput({ input, setInput, onSend, disabled }) {
  const canSend = input.trim() && !disabled;
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "24px";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  useEffect(() => {
    if (!disabled && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [disabled]);

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ delay: 0.2, duration: 0.4 }}
      className="border-t border-white/[0.03] bg-gradient-to-t from-surface via-surface to-transparent pt-2 pb-0"
    >
      <div className="max-w-3xl mx-auto px-6 pb-5">
        <div className="flex items-end gap-3 glass rounded-2xl px-4 py-3 focus-ring transition-all">
          <textarea
            ref={textareaRef}
            rows={1}
            className="flex-1 bg-transparent text-[14px] text-white placeholder-white/20 focus:outline-none resize-none leading-6"
            placeholder={disabled ? "Agent is thinking..." : "Ask anything about the PDF..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && canSend) {
                e.preventDefault();
                onSend();
              }
            }}
            disabled={disabled}
            style={{ minHeight: "24px", maxHeight: "120px" }}
          />

          <motion.button
            whileHover={canSend ? { scale: 1.08 } : {}}
            whileTap={canSend ? { scale: 0.92 } : {}}
            onClick={onSend}
            disabled={!canSend}
            className={`p-2.5 rounded-xl transition-all flex-shrink-0 mb-[1px] ${
              canSend
                ? "bg-accent hover:bg-accent-hover text-white shadow-lg shadow-accent/25"
                : "text-white/10 cursor-default"
            }`}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 2L11 13" /><path d="M22 2l-7 20-4-9-9-4 20-7z" />
            </svg>
          </motion.button>
        </div>

        <div className="flex items-center justify-center gap-4 mt-3">
          <p className="text-[11px] text-white/10">
            Shift + Enter for new line
          </p>
        </div>
      </div>
    </motion.div>
  );
}
