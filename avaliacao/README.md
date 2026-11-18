# Avaliação

- `perguntas_respostas.json`: as 40 perguntas com a resposta de referência (`ground_truth`).
- Uma pasta por fluxo, cada uma com:
  - `data.json`: respostas e contextos gerados pelo RAG (`evaluate.py make-data`);
  - `results_r1.json` a `results_r5.json`: métricas do ragas em 5 rodadas (`evaluate.py eval`).

Respostas geradas com `gemma4:e2b` e avaliadas com o juiz `gemma4:31b`, usando todos os contextos recuperados.

## Fluxos

Todos com `RAG_ARTICLE_K=10`, `RAG_FINAL_K=10`, `RAG_ARTICLE_TOP_N=2`, `RAG_EVENT_K=10`, `RAG_EVENT_TOP_N=3`, `RAG_NUM_CTX=4096` e `RAG_RERANK_CROSS_WEIGHT=0.75`.

| Pasta | Multi-query | Rerank | Subperguntas |
|---|---|---|---|
| `base` | não | não | não |
| `rerank` | não | sim | não |
| `rerank_mq` | sim | sim | não |
| `subperguntas` | não | sim | sim (até 4) |

## Médias das 5 rodadas

| Fluxo | Faithfulness | Answer relevancy | Context precision | Context recall | Answer correctness |
|---|---|---|---|---|---|
| `base` | 0,970 | 0,935 | 0,657 | 0,875 | 0,784 |
| `rerank` | 0,961 | 0,920 | 0,782 | 0,888 | 0,784 |
| `rerank_mq` | 0,971 | 0,934 | 0,776 | 0,904 | 0,782 |
| `subperguntas` | 0,979 | 0,952 | 0,760 | 0,957 | 0,821 |

## Reproduzir um fluxo

Com a API rodando na configuração do fluxo, dentro de `chatbot/`:

    EVAL_DATA_FILE=../avaliacao/base/data.json EVAL_RESULTS_FILE=../avaliacao/base/results_r1.json python evaluate.py all
