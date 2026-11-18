import { useEffect, useState } from "react";
import DocumentCard from "../components/DocumentCard";
import AddDocumentModal from "../components/AddDocumentModal";
import DocumentCardSkeleton from "../components/DocumentCardSkeleton";
import SearchBar from "../components/Searchbar";
import api from "../Api";
import { useAuth } from "../context/AuthContext"
import { Plus } from "lucide-react";



function Documents() {
  const [documents, setDocuments] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [loading, setLoading] = useState(false);
  const [titleModal, setTtileModal] = useState("");
  const [valueSearch, setValueSearch] = useState("");
  const { user } = useAuth();

  useEffect(() => {
    fetchDocuments();
  }, []);

  async function fetchDocuments() {
    setLoading(true);
    const data = await api.getDocuments();

    setDocuments(data);
    setLoading(false);
  }

  async function handleAddDocument(newDoc) {
    if (selectedDocumentId) {
      newDoc["doc_type"] = 1;
      newDoc["pai_id"] = selectedDocumentId.trim();
      setSelectedDocumentId("");
    } else {
      newDoc["pai_id"] = " ";
    }
    
    if (newDoc["file"])
      await api.uploadPDF(newDoc);
    else
      await api.postUrlDoc(newDoc);

    fetchDocuments();
  }

  function popUpNormative(documentId) {
    setSelectedDocumentId(documentId);
    setTtileModal("Nova Resolução");
    setIsModalOpen(true);
  }

  async function deleteDocument(id) {
    setLoading(true);
    await api.deleteDocument(id)
    fetchDocuments();
  }

  function downloadDocument(documentId) {
    api.downloadFile(documentId);
  }


  return (
    <div className="h-full overflow-y-auto bg-gray-900 text-white p-10">

      <div className="max-w-5xl mx-auto">

        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Documentos</h1>

          {user?.role === "admin" && <button
            onClick={() => {
              setIsModalOpen(true);
              setTtileModal("Novo Documento");
            }}
            className="bg-blue-600 hover:bg-blue-700 justify-between px-3 py-2 rounded-xl flex">
              <Plus className="mr-2" />
              <span >Novo Documento</span>
          </button>}
        </div>

        <div className="pb-6">
          <SearchBar
            setValue={setValueSearch}
          />
        </div>

        <div className="grid gap-6">
          {
            loading ?
              Array.from({ length: 5 }).map((_, i) => (

                <DocumentCardSkeleton key={i} />

              ))
              :
              documents
                .filter((doc) =>
                  doc.titulo.toLowerCase().includes(valueSearch.toLowerCase())
                )
                .map((doc) => (
                  <DocumentCard
                    key={doc.id}
                    document={doc}
                    onDelete={deleteDocument}
                    onAddNormative={popUpNormative}
                    onDownload={downloadDocument}
                  />
                ))
          }
        </div>

      </div>


      <AddDocumentModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleAddDocument}
        title={titleModal}

      />
    </div>
  );
}

export default Documents;
