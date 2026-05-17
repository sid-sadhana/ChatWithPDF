import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { evaluateConversation } from "../services/api";

const METRIC_INFO = {
  faithfulness: {
    label: "Faithfulness",
    desc: "Is the answer grounded in the retrieved context?",
    color: "emerald",
  },
  answer_relevancy: {
    label: "Answer Relevancy",
    desc: "Is the answer relevant to the question?",
    color: "blue",
  },
  context_precision: {
    label: "Context Precision",
    desc: "Are the retrieved chunks relevant to the query?",
    color: "purple",
  },
};

function ScoreBar({ label, desc, value, color }) {
  const pct = value != null ? Math.round(value * 100) : null;
  const colorMap = {
    emerald: { bg: "bg-emerald-500/20", fill: "bg-emerald-500", text: "text-emerald-400" },
    blue: { bg: "bg-blue-500/20", fill: "bg-blue-500", text: "text-blue-400" },
    purple: { bg: "bg-purple-500/20", fill: "bg-purple-500", text: "text-purple-400" },
  };
  const c = colorMap[color] || colorMap.blue;

  return (
    <div className="mb-3">
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-[11px] font-medium text-white/60">{label}</span>
        <span className={`text-[12px] font-bold ${c.text}`}>
          {pct != null ? `${pct}%` : "N/A"}
        </span>
      </div>
      <div className={`h-2 rounded-full ${c.bg} overflow-hidden`}>
        {pct != null && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className={`h-full rounded-full ${c.fill}`}
          />
        )}
      </div>
      <p className="text-[9px] text-white/20 mt-0.5">{desc}</p>
    </div>
  );
}

export function EvalPanel({ conversationId }) {
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const runEval = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await evaluateConversation(conversationId);
      setEvalData(result);
      setExpanded(true);
    } catch (err) {
      setError(
        err?.response?.data?.error || err?.message || "Evaluation failed"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-l border-white/[0.04] flex flex-col w-[320px] min-w-[280px] bg-surface/50 backdrop-blur-sm">
      {/* header */}
      <div className="px-4 py-3 border-b border-white/[0.04] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          </div>
          <span className="text-[12px] font-medium text-white/50">RAGAS Eval</span>
        </div>
        <button
          onClick={runEval}
          disabled={loading}
          className={`text-[10px] px-3 py-1.5 rounded-lg font-medium transition-all ${
            loading
              ? "bg-white/5 text-white/20 cursor-wait"
              : "bg-accent/20 text-accent hover:bg-accent/30 border border-accent/20"
          }`}
        >
          {loading ? "Evaluating..." : evalData ? "Re-run" : "Run"}
        </button>
      </div>

      {/* content */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading && (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <div className="w-6 h-6 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
            <p className="text-[11px] text-white/30">Running RAGAS metrics...</p>
            <p className="text-[9px] text-white/15">This may take a minute</p>
          </div>
        )}

        {error && !loading && (
          <div className="glass rounded-xl px-3 py-3 border border-red-500/20">
            <p className="text-[11px] text-red-400">{error}</p>
          </div>
        )}

        {!loading && !error && !evalData && (
          <div className="flex flex-col items-center justify-center py-12 gap-2 text-center">
            <div className="w-10 h-10 rounded-xl bg-white/[0.03] flex items-center justify-center mb-1">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-white/15"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <p className="text-[11px] text-white/25">
              Evaluate your RAG pipeline with RAGAS
            </p>
            <p className="text-[9px] text-white/15 max-w-[200px]">
              Measures faithfulness, answer relevancy, and context precision across all Q&A pairs
            </p>
          </div>
        )}

        {!loading && evalData && (
          <div>
            {/* aggregate scores */}
            {evalData.aggregate && (
              <div className="mb-5">
                <p className="text-[10px] uppercase tracking-wider text-white/20 mb-3">
                  Aggregate Scores
                </p>
                {Object.entries(METRIC_INFO).map(([key, info]) => (
                  <ScoreBar
                    key={key}
                    label={info.label}
                    desc={info.desc}
                    value={evalData.aggregate[key]}
                    color={info.color}
                  />
                ))}
              </div>
            )}

            {/* per-question breakdown */}
            {evalData.scores && evalData.scores.length > 0 && (
              <div>
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="flex items-center gap-1 text-[10px] text-white/25 hover:text-white/40 transition-colors mb-2"
                >
                  <svg
                    width="8"
                    height="8"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    className={`transition-transform ${expanded ? "rotate-90" : ""}`}
                  >
                    <path d="M8 5l10 7-10 7z" />
                  </svg>
                  Per-question breakdown ({evalData.scores.length})
                </button>

                <AnimatePresence>
                  {expanded && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden space-y-2"
                    >
                      {evalData.scores.map((row, i) => (
                        <div
                          key={i}
                          className="glass rounded-xl px-3 py-2.5 border border-white/[0.04]"
                        >
                          <p className="text-[10px] text-white/40 mb-2 line-clamp-2">
                            Q: {row.question}
                          </p>
                          <div className="grid grid-cols-3 gap-2">
                            {Object.entries(METRIC_INFO).map(([key, info]) => {
                              const val = row[key];
                              const pct = val != null ? Math.round(val * 100) : null;
                              const colorMap = {
                                emerald: "text-emerald-400",
                                blue: "text-blue-400",
                                purple: "text-purple-400",
                              };
                              return (
                                <div key={key} className="text-center">
                                  <p className={`text-[12px] font-bold ${colorMap[info.color]}`}>
                                    {pct != null ? `${pct}%` : "--"}
                                  </p>
                                  <p className="text-[8px] text-white/20">{info.label}</p>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
