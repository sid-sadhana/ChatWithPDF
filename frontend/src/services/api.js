import axios from "axios";

const baseURL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL,
  timeout: 300_000,
});

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file, file.name);
  const res = await apiClient.post("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function sendMessage({ conversationId, query }) {
  const res = await apiClient.post("/api/chat", {
    conversation_id: conversationId,
    query,
  });
  return res.data;
}

export async function listConversations() {
  const res = await apiClient.get("/api/conversations");
  return res.data;
}

export async function getConversation(conversationId) {
  const res = await apiClient.get(`/api/conversations/${conversationId}`);
  return res.data;
}

export async function deleteConversation(conversationId) {
  await apiClient.delete(`/api/conversations/${conversationId}`);
}

export async function deleteAllConversations() {
  await apiClient.delete("/api/conversations");
}

export async function evaluateConversation(conversationId) {
  const res = await apiClient.post(
    `/api/conversations/${conversationId}/evaluate`
  );
  return res.data;
}
