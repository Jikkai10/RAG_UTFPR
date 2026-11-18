import { useRef, useEffect, useState } from "react";
import Message from "./Message";
import ChatInput from "./ChatInput";
import { motion, AnimatePresence } from "framer-motion";

function ChatWindow({ messages, onSend, loading, onOpenSources, onOpenCitation }) {
  const scrollRef = useRef(null);
  const bottomRef = useRef(null);

  const [isAtBottom, setIsAtBottom] = useState(true);

  const isEmpty = messages.length === 0;

  // Detecta se o usuário está no final
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsAtBottom(entry.isIntersecting);
      },
      {
        root: scrollRef.current,
        threshold: 1.0
      }
    );

    if (bottomRef.current) {
      observer.observe(bottomRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Scroll automático
  useEffect(() => {
    if (!isAtBottom) return;

    const el = scrollRef.current;
    if (!el) return;

    el.scrollTo({
      top: el.scrollHeight,
      behavior: "smooth"
    });

  }, [messages, isAtBottom]);

  return (
    <div className="relative flex flex-col flex-1 bg-gray-900 overflow-hidden">

      <div
        ref={scrollRef}
        className="overflow-y-auto chat-scroll flex-1 px-6 pt-8 pb-24 space-y-6"
      >
        {messages.map((msg, index) => (
          <Message
            key={index}
            message={msg}
            onOpenSources={onOpenSources}
            onOpenCitation={onOpenCitation}
          />
        ))}

        {/* sentinel */}
        <div ref={bottomRef} />
      </div>

     
      <motion.div
      layout
        animate={{
          top: isEmpty ? "40%" : "auto",
          bottom: isEmpty ? "auto" : "0%",
          
        }}
        transition={{ duration: 0.4, ease: "easeInOut" }}
        className="absolute left-0 w-full px-6 pb-6"
      >
        <div className="max-w-3xl mx-auto text-center">

          <AnimatePresence>
            {isEmpty && (
              <motion.h1
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="text-3xl font-semibold mb-6"
              >
                Como posso ajudar você hoje?
              </motion.h1>
            )}
          </AnimatePresence>

          <ChatInput
          
          onSend={onSend} loading={loading} />

        </div>
      </motion.div>
    </div>
  );
}

export default ChatWindow;
