import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageBubble } from "./MessageBubble";
import { useAutoScroll } from "../hooks/useAutoScroll";

export function MessageList({ messages, loading }) {
  const endRef = useAutoScroll([messages, loading]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
        {messages.length === 0 && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-center mt-20"
          >
            <p className="text-white/15 text-sm">Ask a question about the uploaded PDF.</p>
          </motion.div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <MessageBubble key={m.id} message={m} index={i} />
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0 text-[11px] font-bold text-white mt-0.5 shadow-lg shadow-purple-500/20">
              AI
            </div>
            <div className="glass rounded-2xl rounded-tl-lg px-5 py-4">
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  <span className="dot-1 w-2 h-2 rounded-full bg-accent/60 inline-block" />
                  <span className="dot-2 w-2 h-2 rounded-full bg-accent/60 inline-block" />
                  <span className="dot-3 w-2 h-2 rounded-full bg-accent/60 inline-block" />
                </div>
                <span className="text-[12px] text-white/25 ml-1">Agent is thinking</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={endRef} />
      </div>
    </div>
  );
}
