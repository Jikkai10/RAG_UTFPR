import { useState, useRef, useEffect } from "react";
import { Download, Trash2, Plus } from "lucide-react";
import { useAuth } from "../context/AuthContext"

function DocumentCard({
  document,
  onDelete,
  onDownload,
  onAddNormative,
}) {
  const [expanded, setExpanded] = useState(false);
  const contentRef = useRef(null);
  const [height, setHeight] = useState(0);
  const { user } = useAuth()
  const hasNormatives = document.norms?.length > 0;
  // calendários (tipo 3) não recebem resoluções
  const isCalendar = document.tipo === 3;

  useEffect(() => {
    if (expanded) {
      setHeight(contentRef.current.scrollHeight);
    } else {
      setHeight(0);
    }
  }, [expanded, document.norms]);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-2xl transition-all duration-300">

      {/* HEADER */}
      <div
        onClick={() => setExpanded(!expanded)}
        className="p-6 cursor-pointer flex justify-between items-center hover:bg-gray-700/40 transition"
      >
        <div>
          <h2 className="text-xl font-semibold">{document.titulo}</h2>
          <p className="text-gray-400 text-sm">
            {isCalendar
              ? "Calendário acadêmico"
              : `${document.norms?.length || 0} resoluções`}
          </p>
        </div>

        <div className="flex items-center gap-3">

          {user?.role === "admin" && !isCalendar && <button
            onClick={(e) => {
              e.stopPropagation();
              onAddNormative(document.id);
            }}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm"
          >
            <Plus className="cursor-pointer hover:text-blue-500" size={18} />
          </button>}

          {user?.role === "admin" && <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(document.id);
            }}
            className="px-3 py-2 bg-red-600 hover:bg-red-700 rounded-xl text-sm"
          >
            <Trash2 className="cursor-pointer hover:text-red-500" size={18} />
          </button>}

          <button
            onClick={(e) => {
              e.stopPropagation();
              onDownload(document.id);
            }}
            className="px-3 py-2  bg-gray-500 hover:bg-gray-600 rounded-xl text-sm"
          >
            <Download className="cursor-pointer hover:text-blue-500" size={18} />
          </button>

          <div
            className={`transition-transform duration-300 ${
              expanded ? "rotate-180" : ""
            }`}
          >
            ▼
          </div>
        </div>
      </div>

      
      <div
        style={{
          height: height,
          transition: "height 300ms ease",
          overflow: "hidden",
        }}
      >
        <div ref={contentRef} className="px-6 pb-3 pt-3 space-y-3">
          {hasNormatives &&
            document.norms.map((norm) => (
              <div
                key={norm.id}
                className="flex justify-between items-center bg-gray-700/50 p-3 rounded-xl"
              >
                <span>{norm.titulo}</span>
                <div className="flex items-center gap-3">
                  {user?.role === "admin" && <button
                    onClick={() =>
                      onDelete(norm.id)
                    }
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded-lg"
                  >
                    <Trash2 className="cursor-pointer hover:text-red-500" size={16} />
                  </button>}

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDownload(norm.id);
                    }}
                    className="px-3 py-1 bg-gray-500 hover:bg-gray-600 rounded-lg"
                  >
                    <Download className="cursor-pointer hover:text-blue-500" size={16} />
                  </button>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

export default DocumentCard;
