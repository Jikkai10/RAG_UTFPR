import { useRef, useState } from "react";
import {
  Upload,
  Play,
  Download,
  Loader2,
  ClipboardList,
  BarChart3,
  AlertTriangle,
  Trash2,
} from "lucide-react";
import api from "../Api";
import { useAuth } from "../context/AuthContext";

const METRIC_LABELS = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
};

function metricLabel(name) {
  return METRIC_LABELS[name] || name.replaceAll("_", " ");
}

function downloadJson(content, filename) {
  const blob = new Blob([JSON.stringify(content, null, 4)], {
    type: "application/json",
  });
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  a.remove();
  window.URL.revokeObjectURL(url);
}

function errorMessage(error) {
  if (!error) return null;

  const detail = error?.data?.detail;

  if (typeof detail === "string") return detail;
  if (detail) return JSON.stringify(detail);
  if (error.status) return `Erro ${error.status}`;

  return error.message || "Erro inesperado";
}

// mesmo formato do questions.json do backend; campos extras são ignorados
function parseQuestions(text) {
  const parsed = JSON.parse(text);

  if (!Array.isArray(parsed)) {
    throw new Error("O JSON precisa ser uma lista de perguntas.");
  }

  return parsed.map((item, index) => {
    if (!item?.question || !item?.ground_truth) {
      throw new Error(
        `Item ${index + 1} precisa ter os campos "question" e "ground_truth".`
      );
    }

    return { question: item.question, ground_truth: item.ground_truth };
  });
}

function parseEvalData(text) {
  const parsed = JSON.parse(text);

  if (!Array.isArray(parsed)) {
    throw new Error("O JSON precisa ser uma lista de exemplos.");
  }

  return parsed.map((item, index) => {
    if (!item?.question || !item?.answer || !item?.ground_truth) {
      throw new Error(
        `Item ${index + 1} precisa ter "question", "answer" e "ground_truth".`
      );
    }

    return {
      question: item.question,
      answer: item.answer,
      contexts_all: item.contexts_all || [],
      contexts_cited: item.contexts_cited || [],
      ground_truth: item.ground_truth,
    };
  });
}

function Evaluation() {
  const [questions, setQuestions] = useState([]);
  const [evalData, setEvalData] = useState([]);
  const [results, setResults] = useState(null);
  const [contextSource, setContextSource] = useState("all");
  const [loadingData, setLoadingData] = useState(false);
  const [loadingScore, setLoadingScore] = useState(false);
  const [error, setError] = useState(null);
  const [pasted, setPasted] = useState("");

  const questionsInput = useRef();
  const dataInput = useRef();

  const { user } = useAuth();

  if (user?.role !== "admin") {
    return (
      <div className="h-full flex items-center justify-center bg-gray-900 text-white p-10">
        <div className="flex items-center gap-3 text-gray-300">
          <AlertTriangle className="text-yellow-500" />
          <span>Apenas administradores podem avaliar o modelo.</span>
        </div>
      </div>
    );
  }

  function readFile(file, parse, apply) {
    if (!file) return;

    const reader = new FileReader();

    reader.onload = () => {
      try {
        apply(parse(reader.result));
        setError(null);
      } catch (e) {
        setError({ message: e.message });
      }
    };

    reader.readAsText(file);
  }

  function handleQuestionsFile(e) {
    readFile(e.target.files[0], parseQuestions, (parsed) => {
      setQuestions(parsed);
      setEvalData([]);
      setResults(null);
    });

    e.target.value = "";
  }

  function handleDataFile(e) {
    readFile(e.target.files[0], parseEvalData, (parsed) => {
      setEvalData(parsed);
      setResults(null);
    });

    e.target.value = "";
  }

  function handlePaste() {
    try {
      setQuestions(parseQuestions(pasted));
      setEvalData([]);
      setResults(null);
      setPasted("");
      setError(null);
    } catch (e) {
      setError({ message: e.message });
    }
  }

  async function generateData() {
    if (!questions.length) return;

    setLoadingData(true);
    setError(null);
    setResults(null);

    try {
      setEvalData(await api.makeEvalData(questions));
    } catch (e) {
      setError(e);
    }

    setLoadingData(false);
  }

  async function score() {
    if (!evalData.length) return;

    setLoadingScore(true);
    setError(null);

    try {
      setResults(await api.scoreEvalData(evalData, contextSource));
    } catch (e) {
      setError(e);
    }

    setLoadingScore(false);
  }

  function clearAll() {
    setQuestions([]);
    setEvalData([]);
    setResults(null);
    setError(null);
  }

  const metricNames = results ? Object.keys(results.media) : [];

  return (
    <div className="h-full overflow-y-auto bg-gray-900 text-white p-10">
      <div className="max-w-5xl mx-auto">
        <div className="flex justify-between items-center mb-2">
          <h1 className="text-3xl font-bold">Avaliação</h1>

          <button
            onClick={clearAll}
            disabled={loadingData || loadingScore}
            className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gray-800 hover:bg-gray-700 disabled:opacity-50"
          >
            <Trash2 size={18} />
            <span>Limpar</span>
          </button>
        </div>

        <p className="text-gray-400 mb-8">
          Gera as respostas do RAG para um conjunto de perguntas e pontua com as
          métricas do ragas.
        </p>

        {error && (
          <div className="flex items-start gap-3 mb-6 p-4 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-300">
            <AlertTriangle size={18} className="mt-0.5 shrink-0" />
            <span className="text-sm">{errorMessage(error)}</span>
          </div>
        )}

        {/* etapa 1 */}
        <Section
          icon={<ClipboardList size={20} />}
          step="1"
          title="Perguntas"
          subtitle="JSON no formato [{ question, ground_truth }]"
        >
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => questionsInput.current.click()}
              disabled={loadingData}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-700 hover:bg-gray-600 disabled:opacity-50"
            >
              <Upload size={18} />
              <span>Carregar questions.json</span>
            </button>

            <input
              ref={questionsInput}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={handleQuestionsFile}
            />

            <button
              onClick={() => dataInput.current.click()}
              disabled={loadingData}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-700 hover:bg-gray-600 disabled:opacity-50"
            >
              <Upload size={18} />
              <span>Carregar data.json</span>
            </button>

            <input
              ref={dataInput}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={handleDataFile}
            />
          </div>

          <textarea
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            placeholder='Ou cole aqui: [{"question": "...", "ground_truth": "..."}]'
            rows={4}
            disabled={loadingData}
            className="w-full mt-4 p-3 rounded-xl bg-gray-700 text-sm focus:outline-none"
          />

          <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
            <span className="text-sm text-gray-400">
              {questions.length
                ? `${questions.length} pergunta(s) carregada(s)`
                : "Nenhuma pergunta carregada"}
            </span>

            <button
              onClick={handlePaste}
              disabled={!pasted.trim() || loadingData}
              className="px-4 py-2 rounded-xl bg-gray-700 hover:bg-gray-600 disabled:opacity-50"
            >
              Usar JSON colado
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-3 mt-6">
            <button
              onClick={generateData}
              disabled={!questions.length || loadingData || loadingScore}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
            >
              {loadingData ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Play size={18} />
              )}
              <span>Gerar respostas</span>
            </button>

            {evalData.length > 0 && (
              <button
                onClick={() => downloadJson(evalData, "data.json")}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-700 hover:bg-gray-600"
              >
                <Download size={18} />
                <span>Baixar data.json</span>
              </button>
            )}

            <span className="text-sm text-gray-400">
              {loadingData
                ? "Respondendo as perguntas, isso pode demorar alguns minutos..."
                : `${evalData.length} resposta(s) prontas para pontuar`}
            </span>
          </div>
        </Section>

        {/* etapa 2 */}
        <Section
          icon={<BarChart3 size={20} />}
          step="2"
          title="Pontuação"
          subtitle="Métricas do ragas sobre as respostas geradas"
        >
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={contextSource}
              onChange={(e) => setContextSource(e.target.value)}
              disabled={loadingScore}
              className="p-3 rounded-xl bg-gray-700 focus:outline-none"
            >
              <option value="all">Todos os contextos recuperados</option>
              <option value="cited">Apenas os contextos citados</option>
            </select>

            <button
              onClick={score}
              disabled={!evalData.length || loadingScore || loadingData}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
            >
              {loadingScore ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Play size={18} />
              )}
              <span>Avaliar</span>
            </button>

            {results && (
              <button
                onClick={() => downloadJson(results, "results.json")}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-700 hover:bg-gray-600"
              >
                <Download size={18} />
                <span>Baixar results.json</span>
              </button>
            )}

            {loadingScore && (
              <span className="text-sm text-gray-400">
                Pontuando com o modelo juiz...
              </span>
            )}
          </div>

          {results && (
            <div className="mt-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {metricNames.map((name) => (
                  <MetricCard
                    key={name}
                    label={metricLabel(name)}
                    value={results.media[name]}
                  />
                ))}
              </div>

              <p className="text-sm text-gray-400 mt-4">
                {results.n_examples} exemplo(s) · juiz: {results.judge_model} ·
                contextos: {results.context_source}
              </p>

              <div className="overflow-x-auto mt-6 rounded-2xl border border-gray-700">
                <table className="w-full text-sm">
                  <thead className="bg-gray-800 text-gray-300">
                    <tr>
                      <th className="text-left p-3">Pergunta</th>
                      {metricNames.map((name) => (
                        <th key={name} className="text-right p-3 whitespace-nowrap">
                          {metricLabel(name)}
                        </th>
                      ))}
                    </tr>
                  </thead>

                  <tbody>
                    {results.por_exemplo.map((row, index) => (
                      <tr
                        key={index}
                        className="border-t border-gray-700 hover:bg-gray-800/50"
                      >
                        <td className="p-3 max-w-md">{row.question}</td>

                        {metricNames.map((name) => (
                          <td
                            key={name}
                            className={`p-3 text-right tabular-nums ${scoreColor(
                              row[name]
                            )}`}
                          >
                            {formatScore(row[name])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}

function Section({ icon, step, title, subtitle, children }) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-2xl p-6 mb-6">
      <div className="flex items-center gap-3 mb-1">
        <div className="p-2 rounded-xl bg-gray-700 text-blue-400">{icon}</div>

        <div>
          <h2 className="text-lg font-semibold">
            <span className="text-gray-500 mr-2">{step}.</span>
            {title}
          </h2>
          <p className="text-xs text-gray-400">{subtitle}</p>
        </div>
      </div>

      <div className="mt-5">{children}</div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-4">
      <p className="text-xs text-gray-400">{label}</p>

      <p className={`text-2xl font-bold mt-1 tabular-nums ${scoreColor(value)}`}>
        {formatScore(value)}
      </p>
    </div>
  );
}

// null vem do backend quando a métrica deu NaN (sem contexto ou juiz falhou)
function formatScore(value) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

function scoreColor(value) {
  if (value === null || value === undefined) return "text-gray-500";
  if (value >= 0.8) return "text-green-400";
  if (value >= 0.5) return "text-yellow-400";

  return "text-red-400";
}

export default Evaluation;
