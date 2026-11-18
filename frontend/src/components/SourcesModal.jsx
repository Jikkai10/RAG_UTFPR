import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import SourcesCard from "./SourceCard";

// o marcador [n] da resposta é o índice do documento na lista completa
function citationNumber(src, all) {
  const index = all.findIndex(
    (item) => item.metadata?.id && item.metadata.id === src.metadata?.id
  );

  return index >= 0 ? index + 1 : null;
}

function SourcesModal({ cited = [], all = [], tab = "cited", focus = null, onClose }) {
  const [hasShadow, setHasShadow] = useState(false);
  const [activeTab, setActiveTab] = useState(tab);
  const [focused, setFocused] = useState(focus);

  const handleScroll = (e) => {
    setHasShadow(e.target.scrollTop > 0);
  };

  useEffect(() => {
    function handleEsc(e) {
      if (e.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const tabs = [
    { key: "cited", label: "Fontes citadas", list: cited },
    { key: "all", label: "Todas as fontes", list: all }
  ];

  const list = activeTab === "cited" ? cited : all;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
        onClick={onClose}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          onClick={(e) => e.stopPropagation()}
          className="bg-gray-800 p-6 rounded-2xl w-125 max-h-[80vh] shadow-xl flex flex-col"
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
        >
          <div className={`
            flex justify-between items-center pb-4
            transition-all duration-200
            ${hasShadow ? "shadow-lg shadow-black/40" : ""}
          `}>
            <h2 className="text-lg font-bold">
              {focused ? `Fonte [${focused.number}]` : "Fontes"}
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>

          {focused ? (
            <>
              <button
                onClick={() => setFocused(null)}
                className="flex items-center gap-2 text-sm text-gray-300 hover:text-white pb-3"
              >
                <ArrowLeft size={16} />
                Ver todas as fontes
              </button>

              <div className="flex-1 overflow-y-auto hide-scrollbar">
                <SourcesCard
                  src={focused.source}
                  id={focused.number}
                  number={focused.number}
                  defaultExpanded
                />
              </div>
            </>
          ) : (
            <>
              <div className="flex gap-2 pb-4">
                {tabs.map(({ key, label, list: tabList }) => (
                  <button
                    key={key}
                    onClick={() => setActiveTab(key)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition
                      ${activeTab === key
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
                  >
                    {label} ({tabList.length})
                  </button>
                ))}
              </div>

              <div
                onScroll={handleScroll}
                className="flex-1 overflow-y-auto space-y-3 hide-scrollbar"
              >
                {list.length === 0 ? (
                  <p className="text-sm text-gray-400">
                    {activeTab === "cited"
                      ? "A resposta não citou nenhuma fonte."
                      : "Nenhuma fonte foi recuperada."}
                  </p>
                ) : (
                  list.map((src, i) => (
                    <SourcesCard
                      key={src.metadata?.id || i}
                      src={src}
                      id={i}
                      number={
                        activeTab === "all" ? i + 1 : citationNumber(src, all)
                      }
                    />
                  ))
                )}
              </div>
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

export default SourcesModal;
