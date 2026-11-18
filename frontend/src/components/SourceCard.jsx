import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

function sourceSubtitle(metadata) {
    if (metadata.source_type === "calendario") {
        const term = [
            metadata.Periodo && `${metadata.Periodo} período`,
            metadata.Ano
        ].filter(Boolean).join(" de ");

        return [
            metadata.Data || metadata.Mes,
            metadata.Categoria,
            term,
            metadata.Campus
        ].filter(Boolean).join(" — ");
    }

    // mensagens antigas foram salvas com a chave acentuada
    const secao = metadata.Secao || metadata["Seção"];

    return [
        metadata.Capitulo && `Capítulo ${metadata.Capitulo}`,
        secao && `Seção ${secao}`,
        metadata.Artigo && `Artigo ${metadata.Artigo}`,
        metadata.Pagina && `p. ${metadata.Pagina}`
    ].filter(Boolean).join(" — ");
}

function SourcesCard({ src, id, number, defaultExpanded = false }) {
    const [expanded, setExpanded] = useState(defaultExpanded);
    const contentRef = useRef(null);
    const [height, setHeight] = useState(0);
    const subtitle = sourceSubtitle(src.metadata);

    useEffect(() => {
        if (expanded) {
        setHeight(contentRef.current.scrollHeight);
        } else {
        setHeight(0);
        }
    }, [expanded, src]);

    return (
        <div className="p-3 bg-gray-700 rounded-lg">
            <div
                key={id}
                onClick={() => setExpanded(!expanded)}
                className=" flex justify-between items-center"
            >
                <div className="flex items-start gap-3 min-w-0">
                {number && (
                    <span className="shrink-0 mt-0.5 px-2 py-0.5 rounded-md bg-blue-600/80 text-xs font-mono">
                        {number}
                    </span>
                )}
                <div className="min-w-0">
                <div className="font-semibold">
                    {src.metadata.Documento ||
                        (src.metadata.source_type === "calendario"
                            ? "Calendário acadêmico"
                            : "Documento")}
                </div>
                {subtitle && (
                    <div className="text-sm text-gray-200">
                        {subtitle}
                    </div>
                )}
                </div>
                </div>
                <div
                className={`transition-transform duration-300 ${
                    expanded ? "rotate-180" : ""
                }`}
                >
                ▼
                </div>
                
            </div>

            <div
                style={{
                height: height,
                transition: "height 300ms ease",
                overflow: "hidden",
                }}
            >
                <div ref={contentRef} className="prose prose-invert w-full pt-2 overflow-x-auto">
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks, remarkMath]} rehypePlugins={[rehypeKatex]}>
                    {src.content}
                    </ReactMarkdown>
                </div>

            </div>
        </div>
    );
}

export default SourcesCard;