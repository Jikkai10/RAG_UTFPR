const BASE_URL = "http://localhost:8080";


async function getDocuments() {
  const res = await apiFetch('/all_docs', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function getChats() {
  const res = await apiFetch('/chat', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function getChatHistory(chatId) {
  const res = await apiFetch(`/chat/${chatId}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function deleteChat(chatId) {
  const res = await apiFetch(`/chat/${chatId}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function updateChatTitle(chatId, new_title) {
  const res = await apiFetch(`/chat/${chatId}/${new_title}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function createChat() {
  const res = await apiFetch('/create_chat', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function sendChatMessage(chatId, message) {
  const res = await apiFetch(`/rag/stream/${chatId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message })
  });
  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }
  return res.body;
}

// etapa 1 da avaliação: responde cada pergunta com o RAG e devolve o dataset
async function makeEvalData(questions) {
  const res = await apiFetch('/evaluate/data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(questions)
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

// etapa 2 da avaliação: pontua o dataset com as métricas do ragas
async function scoreEvalData(evalData, contextSource) {
  const res = await apiFetch('/evaluate/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: evalData, context_source: contextSource })
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

async function postUrlDoc(message) {
  
  const res = await apiFetch(`/docs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      name : message["name"] ,
      url: message["url"],
      doc_type: message["doc_type"],
      pai_id: message["pai_id"]
    })
  });
  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.json() });
  }
  return res.body;
}

export async function uploadPDF(message) {
  const formData = new FormData();

  formData.append("name", message["name"]);
  formData.append("doc_type", message["doc_type"]);
  formData.append("pai_id", message["pai_id"]);
  formData.append("file", message["file"]);

  const response = await apiFetch("/upload-pdf/", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("Erro ao enviar PDF");
  }

  return await response.json();
}

async function deleteDocument(id) {
  
  const res = await apiFetch(`/docs/${id}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' }
  });
  const data = await res.json();
  if (!res.ok) {
    return Promise.reject({ status: res.status, data });
  }
  return data;
}

function parseFilename(disposition) {
  if (!disposition) return "download";

  // starlette usa filename*=utf-8'' quando o nome tem acentos
  const utf8 = disposition.match(/filename\*=\s*utf-8''([^;]+)/i);
  if (utf8) return decodeURIComponent(utf8[1]);

  const ascii = disposition.match(/filename="?([^";]+)"?/i);
  return ascii ? ascii[1] : "download";
}

async function downloadFile(doc_id) {
  const res = await apiFetch(`/download/${doc_id}`, { method: 'GET' });

  if (!res.ok) {
    return Promise.reject({ status: res.status });
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = parseFilename(res.headers.get("Content-Disposition"));
  document.body.appendChild(a);
  a.click();

  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function apiFetch(path, options = {}) {

  const token = localStorage.getItem("token")

  // FormData precisa definir o próprio Content-Type com o boundary do multipart
  const isFormData = options.body instanceof FormData

  const res = await fetch(BASE_URL + path, {
    ...options,
    headers: {
      ...(!isFormData && { "Content-Type": "application/json" }),
      ...(options.headers || {}),
      ...(token && { Authorization: `Bearer ${token}` })
    }
  })

  if (res.status === 401) {
    localStorage.removeItem("token")
    window.location.href = "/login"
  }

  return res
}



export default {
  createChat, sendChatMessage, getDocuments, postUrlDoc, deleteDocument, 
  uploadPDF, downloadFile, getChatHistory, getChats, apiFetch,
  updateChatTitle, deleteChat, makeEvalData, scoreEvalData
};