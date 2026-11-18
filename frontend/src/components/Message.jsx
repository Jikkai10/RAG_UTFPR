import { Children } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

const CITATION_RE = /\[(\d+)\]/g;

const SEARCH_LABELS = {
  searchRegulations: "Regulamentos",
  searchCalendar: "Calendário acadêmico"
};

function searchingLabel({ tool, encontrados }) {
  const target = SEARCH_LABELS[tool] || "Documentos";

  if (encontrados === 0) {
    return `${target} — nenhum trecho encontrado`;
  }

  if (encontrados > 0) {
    return `${target} — ${encontrados} ${encontrados === 1 ? "trecho" : "trechos"}`;
  }

  return `Consultando ${target.toLowerCase()}`;
}

// troca os marcadores [n] por botões que abrem a fonte
function withCitations(children, onCite, max) {
  if (!onCite) return children;

  return Children.toArray(children).flatMap((child, childIndex) => {
    if (typeof child !== "string") return child;

    const parts = [];
    let last = 0;

    for (const match of child.matchAll(CITATION_RE)) {
      const number = Number(match[1]);

      if (match.index > last) {
        parts.push(child.slice(last, match.index));
      }

      if (number >= 1 && number <= max) {
        parts.push(
          <button
            key={`cite-${childIndex}-${match.index}`}
            title="Ver esta fonte"
            onClick={(e) => {
              e.stopPropagation();
              onCite(number);
            }}
            className="align-super mx-0.5 px-1.5 rounded bg-blue-600/70 hover:bg-blue-500
                       text-[0.7em] font-mono leading-tight transition"
          >
            {number}
          </button>
        );
      } else {
        parts.push(match[0]);
      }

      last = match.index + match[0].length;
    }

    if (parts.length === 0) return child;

    if (last < child.length) {
      parts.push(child.slice(last));
    }

    return parts;
  });
}

function Message({ message, onOpenSources, onOpenCitation }) {
  const isUser = message.role === "user";
  const isEmpty = message.content === "";

  const shouldShowCursor =
    message.loading && !isUser && !isEmpty;

  const citeCount = message.contexts?.length || 0;

  const onCite =
    !isUser && citeCount > 0
      ? (number) => onOpenCitation(message, number)
      : null;

  const hasSources =
    !isUser &&
    (message.sources?.length > 0 || message.contexts?.length > 0);

  const withCursor = (Tag, { cite = true } = {}) => ({ node, children, ...props }) => {
    const isLast =
      node?.position?.end?.offset === message.content.length;

    return (
      <Tag {...props}>
        {cite ? withCitations(children, onCite, citeCount) : children}
        {shouldShowCursor && isLast && (
          <span className="inline-block ml-1 animate-pulse align-baseline">
            ▍
          </span>
        )}
      </Tag>
    );
  };

  return (
    <motion.div
    initial={{ opacity: 0, y: -20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4, ease: "easeInOut" }}
    className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        onClick={() => hasSources && onOpenSources(message)}
        className={`max-w-2xl p-5 rounded-2xl shadow transition
        ${isUser ? "bg-blue-600" : "bg-gray-700 hover:bg-gray-600"}
        ${hasSources ? "cursor-pointer" : ""}`}
      >
        <div className="prose prose-invert max-w-none">

            <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkBreaks, remarkMath]} rehypePlugins={[rehypeKatex]}
            components={{
              p: withCursor("p"),
              li: withCursor("li"),
              strong: withCursor("strong"),
              em: withCursor("em"),
              code: withCursor("code", { cite: false }),
              td: withCursor("td"),
              th: withCursor("th"),
            }}
            >
              {message.content}
            </ReactMarkdown>

            {
              isEmpty && !message.searching && (
                <div className="flex space-x-1 justify-center items-center">
                  <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></div>
                </div>
              )
            }


        </div>

        {message.searching && (
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Loader2 size={14} className="animate-spin" />
            <span>{searchingLabel(message.searching)}</span>
          </div>
        )}

        {hasSources && !message.loading && (
          <div className="text-xs text-gray-400 mt-3">
            Clique para ver fontes
          </div>
        )}
      </div>
    </motion.div>
  );
}

export default Message;
