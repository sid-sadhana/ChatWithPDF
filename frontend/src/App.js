import React, { useState, useEffect, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { Toast } from "./components/Toast";
import {
  listConversations,
  uploadPDF,
  deleteConversation,
  deleteAllConversations,
} from "./services/api";

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = "info") => {
    setToast({ message, type, id: Date.now() });
  };

  const refresh = useCallback(async () => {
    try {
      const list = await listConversations();
      setConversations(list);
    } catch {
      /* empty sidebar */
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const triggerUpload = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/pdf";
    input.onchange = async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setUploading(true);
      showToast(`Indexing "${file.name}"...`, "loading");
      try {
        const conv = await uploadPDF(file);
        await refresh();
        setActiveId(conv.conversation_id);
        showToast("PDF indexed successfully!", "success");
      } catch (err) {
        showToast(
          err?.response?.data?.error || err?.message || "Upload failed",
          "error"
        );
      } finally {
        setUploading(false);
      }
    };
    input.click();
  };

  const handleDeleteAll = async () => {
    try {
      await deleteAllConversations();
      setActiveId(null);
      setConversations([]);
      showToast("All data cleared", "info");
    } catch {
      showToast("Failed to clear data", "error");
    }
  };

  const handleDelete = async (id) => {
    try {
      await deleteConversation(id);
      if (activeId === id) setActiveId(null);
      await refresh();
      showToast("Conversation deleted", "info");
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="h-screen bg-surface text-white flex overflow-hidden relative">
      {/* ambient glow */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-[40%] -left-[20%] w-[60%] h-[60%] rounded-full bg-accent/[0.03] blur-[120px]" />
        <div className="absolute -bottom-[30%] -right-[10%] w-[50%] h-[50%] rounded-full bg-purple-600/[0.03] blur-[120px]" />
      </div>

      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={triggerUpload}
        onDelete={handleDelete}
        onDeleteAll={handleDeleteAll}
        uploading={uploading}
      />

      <main className="flex-1 flex flex-col min-w-0 relative">
        <AnimatePresence mode="wait">
          <ChatWindow
            key={activeId || "empty"}
            conversationId={activeId}
            onUpload={triggerUpload}
            uploading={uploading}
          />
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {toast && (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onDone={() => setToast(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
