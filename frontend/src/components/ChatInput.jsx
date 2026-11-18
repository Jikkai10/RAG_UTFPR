import { useState, useRef } from "react";
import { Loader2, Send } from "lucide-react"; 

function ChatInput({ onSend, loading }) {
  const [input, setInput] = useState("");
  const textareaRef = useRef(null);

  const MAX_HEIGHT = 200;

  function handleSubmit() {
    if (!input.trim() || loading) return;

    onSend(input);
    setInput("");

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.overflowY = "hidden";
    }
  }

  function handleChange(e) {
    setInput(e.target.value);

    const el = e.target;
    el.style.height = "auto";

    if (el.scrollHeight > MAX_HEIGHT) {
      el.style.height = MAX_HEIGHT + "px";
      el.style.overflowY = "auto";
    } else {
      el.style.height = el.scrollHeight + "px";
      el.style.overflowY = "hidden";
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !loading) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="relative bg-gray-800/80 backdrop-blur-md 
                    border border-gray-700 
                    rounded-2xl shadow-xl p-1
                    transition-all duration-200">

      
      <div className="pointer-events-none absolute top-0 left-0 right-0 h-6 
                      bg-linear-to-b from-gray-800 to-transparent 
                      rounded-t-2xl opacity-0 transition-opacity"
           id="topFade" />

      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => {
            handleChange(e);

            // 🔥 Mostrar fade se houver scroll
            const el = e.target;
            const fade = document.getElementById("topFade");

            if (el.scrollHeight > MAX_HEIGHT && el.scrollTop > 0) {
              fade.style.opacity = "1";
            } else {
              fade.style.opacity = "0";
            }
          }}
          onScroll={(e) => {
            const el = e.target;
            const fade = document.getElementById("topFade");

            if (el.scrollTop > 5) {
              fade.style.opacity = "1";
            } else {
              fade.style.opacity = "0";
            }
          }}
          onKeyDown={handleKeyDown}
          placeholder="Digite sua pergunta..."
          rows={1}
          
          className={`flex-1 resize-none bg-transparent 
                      p-3 focus:outline-none 
                      overflow-hidden 
                      hide-scrollbar
                      transition-all duration-200
                      min-h-12
                      `}
        />

        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`px-5 py-3 rounded-xl transition justify-center
          ${
            loading
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ?
          <Loader2 className="animate-spin" />
          : <Send />}
        </button>
      </div>
    </div>
  );
}

export default ChatInput;
