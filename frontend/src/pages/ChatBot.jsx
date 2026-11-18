import { useState, useEffect } from "react";
import { useImmer } from "use-immer";
import Sidebar from "../components/SideBar";
import ChatWindow from "../components/ChatWindow";
import SourcesModal from "../components/SourcesModal";
import api from "../Api";
import parseSSEStream from "../utils";

function ChatBot() {

  const [messages, setMessages] = useImmer([]);

  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);

  const [loading, setLoading] = useState(false);
  const [sourcesView, setSourcesView] = useState(null);

  useEffect(() => {
    getChats();
  }, []);

  async function getChats() {
    const data = await api.getChats();

    setChats(data);
  }

  async function initChat(id) {
    setActiveChat(id);
    if (!id) {
      setMessages([]);
      return;
    }

    const data = await api.getChatHistory(id);

    setMessages(data);
  }

  async function createNewChat() {
    const data = await api.createChat();

    const chat = {
      "id": data["id"],
      "title": data["title"]
    };

    setChats(prev => [chat, ...prev]);
    setActiveChat(chat["id"]);

    return chat["id"];
  }

  async function updateChat(chatId, title) {
    await api.updateChatTitle(chatId, title);
    getChats();
  }

  async function deleteChat(chatId) {
    await api.deleteChat(chatId);
    getChats();
  }

  function openSources(message, tab = "cited") {
    setSourcesView({
      cited: message.sources || [],
      all: message.contexts || [],
      tab,
      focus: null
    });
  }

  // o marcador [n] é o índice do documento na lista completa de contextos
  function openCitation(message, number) {
    const all = message.contexts || [];
    const source = all[number - 1];

    if (!source) return;

    setSourcesView({
      cited: message.sources || [],
      all,
      tab: "all",
      focus: { source, number }
    });
  }

  async function handleSend(message) {

    setMessages(draft => {
      draft.push({ role: "user", content: message });
    });

    setLoading(true);

    let chatId = activeChat;

    if (!activeChat) {
      chatId = await createNewChat();
      
    }

    setMessages(draft => {
      draft.push({
        role: "assistant",
        content: "",
        loading: true,
        searching: null,
        sources: [],
        contexts: []
      });
    });

    const stream = await api.sendChatMessage(chatId, message);

    for await (const { event, data } of parseSSEStream(stream)) {

      if (event === "token") {
        const token = JSON.parse(data);

        setMessages(draft => {
          const last = draft[draft.length - 1];
          last.content += token;
          last.searching = null;
        });
      }

      if (event === "searching") {
        const searching = JSON.parse(data);

        setMessages(draft => {
          draft[draft.length - 1].searching = searching;
        });
      }

      if (event === "contexts") {
        const contexts = JSON.parse(data);

        setMessages(draft => {
          draft[draft.length - 1].contexts = contexts;
        });
      }

      if (event === "sources") {
        const sources = JSON.parse(data);

        setMessages(draft => {
          draft[draft.length - 1].sources = sources;
        });
      }

      if (event === "title") {
        const title = JSON.parse(data)

        setChats(prev =>
          prev.map(draft =>
            draft.id === chatId
              ? { ...draft, title }
              : draft
          )
        )
      }

      if (event === "end") {
        setMessages(draft => {
          const last = draft[draft.length - 1];
          last.loading = false;
          last.searching = null;
        });

        setLoading(false);
      }
    }
  }

  return (
    <div className="flex h-full overflow-hidden bg-gray-900 text-white">

      <Sidebar
        chats={chats}
        activeChat={activeChat}
        setActiveChat={(id) => { initChat(id) }}
        deleteChat={deleteChat}
        updateChat={updateChat}
      />

      <ChatWindow
        messages={messages}
        onSend={handleSend}
        loading={loading}
        onOpenSources={openSources}
        onOpenCitation={openCitation}
      />

      {sourcesView && (
        <SourcesModal
          cited={sourcesView.cited}
          all={sourcesView.all}
          tab={sourcesView.tab}
          focus={sourcesView.focus}
          onClose={() => setSourcesView(null)}
        />
      )}
    </div>
  );
}

export default ChatBot;