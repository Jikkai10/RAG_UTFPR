import { useState, useEffect, useRef } from "react";
import { UploadCloud, Link, FileText, X, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

function AddDocumentModal({ isOpen, onClose, onSave, title }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("url");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [doc_type, setType] = useState(0);
  const fileInput = useRef();

  // calendário só pode ser enviado como PDF
  const isCalendar = doc_type === 3;

  function handleTypeChange(e) {
    const value = parseInt(e.target.value, 10);

    setType(value);

    if (value === 3) {
      setMode("file");
      setUrl("");
    }
  }

  function close(){
    setMessage("");
    setError("");
    setName("");
    setUrl("");
    setFile(null);
    setType(0);
    setMode("url");
    onClose();
  }
  
  useEffect(() => {
    function handleEsc(e) {
      if (e.key === "Escape" && !loading) close();
    }
    
    if (isOpen) {
      document.addEventListener("keydown", handleEsc);
    }

    return () => {
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, onClose]);

  function handleDrop(e) {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];

    if (!dropped) return;

    if (dropped.type !== "application/pdf") {
      setError("Apenas arquivos PDF são permitidos.");
      return;
    }
    setError(null);
    setFile(dropped);
  }

  function handleFileSelect(e) {
    const f = e.target.files[0];

    if (!f) return;

    if (f.type !== "application/pdf") {
      setError("Apenas arquivos PDF são permitidos.");
      return;
    }
    setError(null);
    setFile(f);
  }

  async function handleSubmit() {
    if (!name.trim()) return;

    if ((mode === "file" && file) || (mode === "url" && url)){
      if (mode === "url") setFile(null);
      setLoading(true)
      await onSave({
        id: Date.now(),
        name,
        url,
        file,
        doc_type
      });
      setMessage("Enviado com sucesso!");
      setLoading(false);
      setName("");
      setUrl("");
      setFile(null);
      setType(0);
    }
    
  }

  

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50"
    >
      
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fadeIn"
        onClick={close}
      />

      
      <div
        className="relative bg-gray-800 text-white w-full max-w-lg p-6 rounded-2xl shadow-2xl border border-gray-700
                   animate-scaleIn transition-all"
      >
        <h2 className="text-xl font-semibold mb-4">
          {title}
        </h2>

        <div className="space-y-4">
          <input
            type="text"
            placeholder="Nome do documento"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full p-3 rounded-xl bg-gray-700 focus:outline-none"
          />

          <select
            value={doc_type}
            onChange={handleTypeChange}
            disabled={loading}
            className="w-full p-3 rounded-xl bg-gray-700 focus:outline-none"
          >
            <option value='0'>Documento</option>
            <option value='3'>Calendario</option>
          </select>

          <div className="flex gap-6">

            <label className={`flex items-center gap-2 ${isCalendar ? "opacity-50" : ""}`}>
              <input
                type="radio"
                checked={mode === "url"}
                onChange={() => setMode("url")}
                disabled={loading || isCalendar}
              />
              URL
            </label>

            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={mode === "file"}
                onChange={() => setMode("file")}
                disabled={loading}
              />
              Arquivo
            </label>


        </div>

        {isCalendar && (
          <p className="text-xs text-gray-400">
            Calendários são processados apenas a partir de PDF.
          </p>
        )}

        
        <AnimatePresence mode="wait">

          {/* upload arquivo */}
          {mode === "file" && (
            <motion.div
              key="file"
              initial={{height: 0, opacity: 0 }}
              animate={{height: "auto", opacity: 1 }}
              exit={{height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
            >

              {!file && (
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => fileInput.current.click()}
                  className="border-2 border-dashed rounded-xl p-10 text-center cursor-pointer hover:bg-gray-50 transition"
                >
                  <UploadCloud
                    size={40}
                    className="mx-auto text-gray-400 mb-3"
                  />

                  <p className="text-sm text-gray-600">
                    Arraste um arquivo aqui
                  </p>

                  <p className="text-xs text-gray-400">
                    ou clique para selecionar
                  </p>

                  <input
                    ref={fileInput}
                    type="file"
                    disabled={loading}
                    accept="application/pdf"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                </div>
              )}

              {file && (
                <div className="flex items-center justify-between border rounded-lg p-3">

                  <div className="flex items-center gap-3">

                    <FileText className="text-gray-500"/>

                    <div>
                      <p className="text-sm font-medium truncate max-w-55">
                        {file.name}
                      </p>

                      <p className="text-xs text-gray-400">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>

                  </div>

                  <button
                    onClick={() => setFile(null)}
                    disabled={loading}
                    className="text-gray-400 hover:text-red-500"
                  >
                    <X size={16}/>
                  </button>

                </div>
              )}
              {error && (
                <p className="text-red-500 text-xs mt-2">
                  {error}
                </p>
              )}
            </motion.div>
          )}

          {/* campo url */}
          {mode === "url" && (
            <motion.div
              key="url"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >

              <div className="flex items-center border rounded-lg px-3 py-2">

                <Link
                  size={18}
                  className="text-gray-400 mr-2"
                />

                <input
                  type="text"
                  placeholder="https://site.com/documento"
                  value={url}
                  disabled={loading}
                  onChange={(e) => setUrl(e.target.value)}
                  className="flex-1 outline-none text-sm"
                />

              </div>

            </motion.div>
          )}

        </AnimatePresence>
        </div>
          {message && (
                <div className="flex justify-end mt-1">
                <p className="text-green-500 text-xs mt-2">
                  {message}
                </p>
                </div>
              )}
        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={close}
            disabled={loading}
            className="px-4 py-2 rounded-xl bg-gray-600 hover:bg-gray-500"
          >
            Cancelar
          </button>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700"
          >
            {
              loading ?
              <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
              :
              <p>Salvar</p>
            }
            
          </button>
        </div>
      </div>
    </div>
  );
}

export default AddDocumentModal;
