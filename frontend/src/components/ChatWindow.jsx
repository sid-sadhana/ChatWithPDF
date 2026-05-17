import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";

import { ChatInput } from "./ChatInput";
import { MessageList } from "./MessageList";
import { EmptyState } from "./EmptyState";
import { EvalPanel } from "./EvalPanel";
import { sendMessage, getConversation } from "../services/api";

export function ChatWindow({ conversationId, onUpload, uploading }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [convTitle, setConvTitle] = useState("");
  const [showEval, setShowEval] = useState(false);

  const loadConversation = useCallback(async () => {
    if (!conversationId) {
      setMessages([]);
      setConvTitle("");
      return;
    }
    try {
      const conv = await getConversation(conversationId);
      setConvTitle(conv.pdf_filename || conv.title || "");
      setMessages(
        (conv.messages || []).map((m) => ({
          id: m.id,
          text: m.content,
          sender: m.role === "user" ? "user" : "bot",
        }))
      );
    } catch {
      setMessages([]);
    }
  }, [conversationId]);

  useEffect(() => {
    loadConversation();
  }, [loadConversation]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || !conversationId || loading) return;

    const userMsg = { id: Date.now().toString(), text, sender: "user" };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const result = await sendMessage({ conversationId, query: text });
      const botMsg = {
        id: result.message?.id || Date.now().toString(),
        text: result.message?.content || "No response",
        sender: "bot",
        sources: result.sources || {},
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const detail =
        err?.response?.data?.error ||
        err?.message ||
        "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), text: detail, sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  if (!conversationId) {
    return <EmptyState onUpload={onUpload} uploading={uploading} />;
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="flex-1 flex overflow-hidden"
    >
      {/* chat area */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* top bar */}
        <motion.div
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="px-6 py-3 border-b border-white/[0.03] flex items-center gap-3 bg-surface/50 backdrop-blur-sm"
        >
          <div className="w-7 h-7 rounded-lg bg-red-500/10 text-red-400 flex items-center justify-center flex-shrink-0">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          </div>
          <span className="text-[13px] text-white/50 truncate flex-1">{convTitle}</span>

          {/* eval toggle */}
          <button
            onClick={() => setShowEval(!showEval)}
            className={`flex items-center gap-1.5 text-[10px] px-2.5 py-1.5 rounded-lg transition-all ${
              showEval
                ? "bg-amber-500/15 text-amber-400 border border-amber-500/20"
                : "text-white/25 hover:text-white/40 hover:bg-white/[0.04] border border-transparent"
            }`}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            RAGAS
          </button>
        </motion.div>

        <MessageList messages={messages} loading={loading} />
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={handleSend}
          disabled={loading}
        />
      </div>

      {/* eval panel on right */}
      {showEval && <EvalPanel conversationId={conversationId} />}
    </motion.div>
  );
}
