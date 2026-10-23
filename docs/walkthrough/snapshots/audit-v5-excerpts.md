# Excertos da auditoria independente V5

Fonte: GETNET_V4_AUDIT_V5_2026-10-23.md. Recorte sobre d7e2125, 23/10/2026. Não é nova execução nem nota oficial.

- `S02`: proposta pendente/receipts0; confirmação de hash exato sem scope recebe403/receipts0; titular com scope recebe200 e um recibo simulado; replay retorna mesmo action ID e mantém uma linha. Não foi enviado comando a terminal.

- **Suíte independente:**484 passed,1 warning,4,69s, Python3.13.7; sockets bloqueados, dotenv suprimido e estado/índice temporários. [Runner](getnet_v5_evaluation/run_suite_independent.py), [log](getnet_v5_evaluation/audit-v5-suite.stdout). Warning upstream Starlette/httpx. Nenhuma nova mudança justificou repetir a suíte.

- **CI:** [run39980766855](https://github.com/arthurdilorenzo/getnet-multiagent-support/actions/runs/39980766855) success noSHA exato; evidência fornecida pelo integrador, sem consulta remota deste revisor. [Recibo](getnet_v5_evaluation/ci-list.json).

- **Componente nativo18:**SHA limpo,13 candidatos por caso, modelo `gemini-3.1-flash-lite`;358,572s totais, mediana24,881s por caso. Esses tempos são do componente, não p95 de chat/carga. [Recibo integral](getnet_v5_evaluation/native-semantic-results.json). Somente5 casos tiveram autorização publicável. Não foi feito retry substitutivo pelo revisor.

**Gate de promoção 95+ não satisfeito, apesar de CI verde.** O congelamento deve ser preservado até decisão do integrador; nenhuma promoção foi feita por esta auditoria.
