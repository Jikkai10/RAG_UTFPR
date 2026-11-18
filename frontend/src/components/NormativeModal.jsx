import { useState, useEffect } from "react";

function AddDocumentModal({ isOpen, onClose, onSave }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);

  useEffect(() => {
    function handleEsc(e) {
      if (e.key === "Escape") onClose();
    }

    if (isOpen) {
      document.addEventListener("keydown", handleEsc);
    }

    return () => {
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, onClose]);

  function handleSubmit() {
    if (!name.trim()) return;

    onSave({
      id: Date.now(),
      name,
      url,
      file,
    });

    setName("");
    setUrl("");
    setFile(null);
    onClose();
  }

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50"
    >
      {/* Overlay */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fadeIn"
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className="relative bg-gray-800 text-white w-full max-w-md p-6 rounded-2xl shadow-2xl border border-gray-700
                   animate-scaleIn transition-all"
      >
        <h2 className="text-xl font-semibold mb-4">
          Novo Documento
        </h2>

        <div className="space-y-4">
          <input
            type="text"
            placeholder="Nome do documento"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full p-3 rounded-xl bg-gray-700 focus:outline-none"
          />

          <input
            type="text"
            placeholder="URL do documento (opcional)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="w-full p-3 rounded-xl bg-gray-700 focus:outline-none"
          />

          <div>
            <label className="cursor-pointer bg-gray-700 px-4 py-2 rounded-xl block text-center hover:bg-gray-600 transition">
              Selecionar arquivo
              <input
                type="file"
                className="hidden"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </label>

            {file && (
              <p className="text-sm text-gray-400 mt-2">
                {file.name}
              </p>
            )}
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-gray-600 hover:bg-gray-500"
          >
            Cancelar
          </button>

          <button
            onClick={handleSubmit}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700"
          >
            Salvar
          </button>
        </div>
      </div>
    </div>
  );
}

export default AddDocumentModal;
