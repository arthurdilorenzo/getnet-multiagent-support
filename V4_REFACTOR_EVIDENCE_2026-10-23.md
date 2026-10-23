# Evidências de refatoração Getnet V4 — 23/10/2026

## Contexto e decisão

Arthur autorizou refatoração dos cinco gaps reportados sobre `b6c8b09`, spec-driven com multiagentes, testes, commit e push em `main` e confirmação de Actions verde. A referência externa continua sendo 39/100 na versão `d40f44c`; 95+ é meta, não nota comprovada. Requisitos e contratos: [SPEC.md](SPEC.md). Resumo operacional local: `../../context/WORKLOG.md` (entrada de 23/10 sobre execução V4).

Fontes primárias: feedback local `feedback_arthur_dilorenzo_v4.md`, auditoria no WORKLOG às 09:20 de 23/10, código/testes do commit baseline e GitHub Actions do próprio repositório. Arquivos privados de carreira e feedback interno não integram a publicação.

## Método e divisão

- Segurança/Auth: configuração, JWT, endpoints, propriedade de sessões, fila HITL e regressões.
- Gemini: function calling nativo no router/support, registry e testes do protocolo SDK.
- CI/Corpus/README: checker Python, higiene de ingestão/snapshot/índice, documentação literal e testes.
- Integração: consistência entre interfaces, regressões de ações, gate de métricas, revisão, validação e publicação.

Os agentes compartilham checkout e possuem arquivos delimitados. Nenhum agente faz commit/push isoladamente; o integrador publica apenas após validação conjunta.

## Baseline reproduzida

- Checkout local `pipeline/getnet_challenge`, branch `main`, HEAD `b6c8b09be0405a2e27e3740695eb7ec698491af5`, sem alterações prévias.
- `source .venv/bin/activate && pytest -q`: **33 passed, 1 warning, 2.12s**. Warning: depreciação do uso de httpx pelo TestClient da dependência Starlette/FastAPI.
- `gh run list --repo arthurdilorenzo/getnet-multiagent-support --limit 3 --json databaseId,headSha,status,conclusion,url`: baseline com `conclusion=failure`, [run 39950167067](https://github.com/arthurdilorenzo/getnet-multiagent-support/actions/runs/39950167067).
- Execução local sem GEMINI_API_KEY; qualquer validação Gemini nesta sessão distingue testes de protocolo simulados de chamadas reais ao provedor.

## Regressões observadas antes dos fixes

- Gate do CLI: 5 falhas por ausência de `main()`/código de saída por thresholds; implementação passou os 5 casos depois. Regressões asseguram falha por routing<1.0, tool recall<1.0, groundedness<0.90 e dataset vazio.
- Integridade operacional: 3 falhas — terminal estrangeiro/inexistente gerava ação no primeiro terminal; falha da tool HITL era descrita como protocolo persistido. Após fix: 3 passaram; terminal inválido não escreve e falha HITL abstém sem protocolo fictício.
- Relatos das frentes terão confirmação pela suíte integrada; não substituem evidência final do integrador.

## Implementação entregue

1. **CI:** o falso positivo da docstring foi removido e a verificação passou de grep textual para AST Python. Comentários/strings/docstrings e arquivos não Python não disparam; chamadas diretas, multilinha e aliases importados de builtins são detectados.
2. **Gemini:** Router usa function calls de delegação validadas; suporte recebe schemas do registry, observa resultados em `FunctionResponse` e pode pedir novas chamadas dependentes. Execução automática do SDK desativada. O conteúdo completo do modelo, IDs e thought signatures é preservado; respostas de função usam role `user`, conforme contrato GenerateContent. Sem chave: regras locais. Com chave/provedor inválido: falha controlada, sem seleção léxica silenciosa. Identidade/sessão não são escolhidas pelo modelo.
3. **Segurança:** HITL list/detail, métricas e evals exigem operador. JWT valida cabeçalho, assinatura e claims; produção exige segredos externos fortes e independentes, autenticação estrita não permite downgrade. Sessões usam proprietário persistido; criação/reply HITL são transacionais. Terminal inexistente/estrangeiro e cliente sem frota não produzem ação. Idempotência é escopada por ferramenta/tenant/sessão.
4. **Corpus:** snapshot passou de 47 para 45 chunks (24 mirrors locais + 21 web), sem alterar threshold. Higiene compartilhada na ingestão e snapshot, URLs/datas preservadas, hashes SHA-256 completos e vetores regenerados. RAG descrito como BM25 com expansão léxica + hashing/âncoras conceituais + score heurístico e RRF como desempate.
5. **Honestidade:** README distingue mocks operacionais, fila local persistida, demo sem identidade verificada, critérios lexicais do harness, limitações de rede, SQLite e timeouts. Removidas garantias de grounding/semântica e preços de referência não medidos; coeficientes de custo são ilustrativos. A seção “Trade-offs e Limitações Declaradas” permanece.

## Revisão independente e correções adicionais

O revisor não participou da implementação e revisou diff, novos arquivos, testes e contratos. Encontrou e reproduziu:

- **P1:** terminal citado em negação ou ordem citada como mensagem no visor autorizava mutation. Gate foi restringida a pedidos explícitos, bloqueia citação/informação e alvos negados. Regressões RED→GREEN.
- **P2:** recusa de segurança era apagada por sucesso posterior da mesma tool em alvo diferente. Recusas passam a ser persistentes; correção de argumentos ordinários ainda permite retry. Regressão RED→GREEN.
- **P2:** sucesso de suporte ocultava a mensagem de falha HITL em pedido composto. Agora a falha parcial aparece e marca abstention; não há confirmação fictícia de encaminhamento. Regressão RED→GREEN.

O revisor reexecutou quatro reproducers e **56 testes focados**, todos passando; liberou o gate pré-push sem achados críticos/altos pendentes. Esse resultado não equivale a certificação de segurança ou nota externa.

## Verificação final do integrador

Ambiente local: Python 3.13.7 na `.venv`; Actions configurado para Python 3.11. Comandos executados no diretório do repositório:

| Comando | Resultado |
|---|---|
| `source .venv/bin/activate && pytest -v` | **165 passed**, 1 warning, 3.34s; exit 0 |
| `python -m app.evals.evaluator` | exit 0; routing **1.0**, tool recall **1.0**, grounding lexical **0.9615**, pass rate **1.0** |
| `python scripts/check_no_print.py app` | exit 0 |
| `git diff --check` | exit 0 |
| `git check-ignore` para `.db`, `.db-wal`, `.db-shm`, `.DS_Store` | todos ignorados; runtime SQLite também excluído do contexto Docker |
| `git ls-files '*.db' '*.db-*' '.DS_Store' '.env'` | nenhum arquivo encontrado |

O warning vem de `fastapi.testclient`/Starlette pelo uso upstream de httpx, não é falha de teste. Os logs integrais desta execução estão em `/tmp/getnet-v4-pytest.log` e `/tmp/getnet-v4-final-eval.log` (artefatos temporários locais, não publicados). O CI reexecuta as verificações no commit publicado.

## Limites e próximo passo

- Não houve chamada Gemini ao vivo: protocolo SDK validado por respostas simuladas e componentes locais reais. Referências primárias: [Google GenAI SDK](https://googleapis.github.io/python-genai/) e [GenerateContent function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling?hl=en).
- O grounding de 0.9615 é score lexical em 13 casos do próprio projeto; não mede qualidade semântica independente nem prova ausência universal de alucinação.
- Consentimento e sanitização são heurísticos, sujeitos a falsos positivos/negativos; ações de terminal permanecem mocks. Produção real exige identidade corporativa, políticas e conectores operacionais reais.
- Bancos SQLite antigos podem conter dados históricos produzidos pela versão vulnerável; não foram migrados ou apagados. Não promover dados antigos sem revisão de propriedade/conteúdo.
- Publicação autorizada em `main` é a etapa posterior a este registro de validação. O SHA final e o resultado do run estão no WORKLOG operacional local e em [GitHub Actions](https://github.com/arthurdilorenzo/getnet-multiagent-support/actions). Reavaliação V4 externa permanece pendente; não afirmar nota 95 obtida.
