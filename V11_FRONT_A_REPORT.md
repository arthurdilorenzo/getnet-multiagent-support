# V11 Frente A — retomada concluída; focos verdes

Data: 2026-10-23. Worktree: `/private/tmp/getnet-v11-worktree`; branch `codex/getnet-v11-conditional-pix`; baseline F519 `f519090b24fd44341e3ba5c0733c5c059255714b`. Sem commit, push, merge, rede/Gemini, credencial privada ou alteração do bundle nativo congelado. Nenhuma nota nova atribuída.

## Estado vigente após retomada

Arthur esclareceu que a comparação de baseline era uma pergunta de viabilidade; o root autorizou retomar exclusivamente as pendências conhecidas. Elas foram corrigidas e os focos estão verdes. Produto estável para a suíte integral centralizada do root. Os registros abaixo preservam a cronologia da interrupção e os REDs; não representam o estado final.

- **8 regressões condicionais corrigidas:** a classificação distingue predicado documental vinculado de leitura privada independente; a autorização continua exclusivamente no gate tipado. Não foi relaxada a exigência de tenant, alvo, status/firmware exatos ou rejeição de predicados extras. Consulta pós-prova preserva texto completo, e follow-up privado continua excluído.
- **2 formatos RED corrigidos:** formato puramente metalinguístico PT/EN vincula-se à necessidade, sem unidade factual extra.
- **P2 revisor corrigido:** diretivas com nova cláusula interrogativa/imperativa deixam de ser qualifiers. A pergunta de tarifa de cancelamento permanece no argumento real de binding; regressões PT/EN adicionadas. Probe do revisor foi executado sem alteração.
- `python /private/tmp/getnet_v11_offline_verify.py tests/test_v11_canonical_e2e.py tests/test_conditional_orchestration_v11.py /private/tmp/test_v11_review_contract_edges.py -q`: **267 passed**, 1 warning upstream; log `/private/tmp/v11-front-a-resume-green.txt`.
- `python /private/tmp/getnet_v11_offline_verify.py tests/test_conditional_dependencies_v11.py tests/test_v11_canonical_e2e.py tests/test_documentary_partial_v10.py tests/test_semantic_integration_v4.py tests/test_state_hitl_supervisor_v3.py -q`: **239 passed**, 1 warning upstream; log `/private/tmp/v11-front-a-resume-other-focused.txt`.
- `git diff --check`: passou. Sem falhas remanescentes nos focos; integral fica com root. Nenhum commit ou acesso de rede/Gemini.

## Registro histórico da interrupção

O integrador informou que Arthur solicitou pausar implementações e reconsiderar o baseline V9 93+. Parei imediatamente; árvore preservada, sem rollback. **A frente não está pronta para promoção nem integralmente verde.** Root integra projeção final no mesmo `orchestrator.py`; o diff desse arquivo contém contribuições de ambos.

## Causas e alterações implementadas

- Unidades: fronteiras antes de qualificadores de fonte eram reparadas só para frases exatas. Instruções de fonte passam a acompanhar a necessidade literal; unidades só de whitespace/pontuação deixam de ser necessidades.
- Documentação: contexto declarativo público e tarefa imediatamente seguinte compartilham escopo; Router/Knowledge reconstruem consulta por escopo canônico. Referência anafórica de aplicabilidade herda identidade pública de produto, sem copiar termos ou país anterior. Fonte explicitamente externa continua externa.
- Replan: teste HTTP PT/EN faz Knowledge local útil Brasil, replan FC sintético Web com query só de qualifier, falha Web injetada. Argumento real conserva Getnet/Crediário/país e os fatos brasileiros/citação/ledger sobrevivem.
- Condição: alvo próprio inequívoco permite preflight read-only mesmo se predicado não reconhecido. Predicado exige observação tipada do alvo e tenant; valores exibidos vêm dessa observação. Consequência deixa de precisar ser literalmente “tell me benefits”, mas extras, ranges, disjunções e qualificadores continuam bloqueando dependente. Leitura independente já agendada não é duplicada.
- Web indisponível: resposta de incidente explicita não confirmação e conserva assunto/local do pedido.
- Consulta derivada de código observado conserva a tarefa “Explique o código 91”.

## Evidência RED/GREEN

Python: `/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python`.

- `python -m pytest tests/test_v11_canonical_e2e.py -q`: RED inicial **11 failed**, após corrigir o spy de teste para `raw_arguments`. Log `/private/tmp/v11-front-a-red.txt`.
- Teste de contexto/tarefa usa FC sintético nativo e decisões semânticas sintéticas em schemas reais, spans locais reais e resposta HTTP serializada. Afirma desligamento, metal para baixo, teste de comunicação, evidence e citação. Não prova qualidade de modelo externo.
- `python /private/tmp/getnet_v11_offline_verify.py tests/test_conditional_dependencies_v11.py tests/test_v11_canonical_e2e.py tests/test_documentary_partial_v10.py tests/test_semantic_integration_v4.py tests/test_state_hitl_supervisor_v3.py -q`: **233 passed**, um warning upstream. Log `/private/tmp/v11-front-a-clean-focus.txt`. Wrapper bloqueia sockets IP antes de importar app.
- `python /private/tmp/getnet_v11_offline_verify.py tests/test_v11_canonical_e2e.py -q`: **19 passed**, um warning upstream, antes dos dois testes de formato novos. Log `/private/tmp/v11-front-a-final-e2e.txt`.
- Último E2E: **2 failed, 19 passed**, `/private/tmp/v11-front-a-format-red.txt`: novos casos “Answer in bullet points” / “Responda em tópicos” demonstram contrato de formato ainda não implementado (unidade extra). Nenhuma alteração de produto foi feita após esse RED.
- `git diff --check`: passou antes da pausa.
- Suíte integral fica a cargo do root. Não há alegação de suíte integral verde.

## Adjudicação autorizada de testes legados

Root autorizou atualizar `tests/test_conditional_orchestration_v11.py` exclusivamente pela spec nova: mocks positivos receberam `user_id='synthetic'`; casos de alvo próprio com condição desconhecida agora exigem exatamente diagnóstico read-only, zero busca dependente/write e nenhuma cobertura fictícia; variantes foram preservadas e nomes atualizados. `tests/test_semantic_integration_v4.py` teve dois asserts autorizados atualizados para exigir contexto canônico de empresa estrangeira/produto/disponibilidade, preservando `request_message` original.

Antes da atualização condicional: **41 failed, 203 passed**. Depois: **8 failed, 236 passed**, `/private/tmp/v11-front-a-legacy-updated.txt`.

As oito falhas remanescentes são todas variantes de:
`tests/test_conditional_orchestration_v11.py::test_native_router_exact_typed_diagnostic_completes_before_any_dependent_retriever_query`.

**Causa real restante, não label obsoleto:** a nova exclusão de estado privado independente em `KnowledgeAgent` é ampla demais: remove também o fragmento firmware/ID que pertence a condição documental já autorizada. No caso Crediário não executa busca; nas variantes postfix pesquisa texto mutilado (ex.: “Tell me the eligible benefits. Only and status is Online.”). O gate confirmou a observação corretamente; a falha está na reconstrução posterior de escopo. Corrigir distinguindo leitura privada independente de condição vinculada já provada, sem transportar unidade privada independente.

## Limitações e próxima retomada

- Formato PT/EN ainda RED; não apresentar como concluído.
- Excluir estado privado sem perder condição já autorizada ainda exige correção e regressões verdes.
- Gramática canônica é finita; herança de assunto usa vocabulário público de produtos e pistas referenciais, não compreensão universal.
- Helpers antigos de reparo SIM ainda constam no arquivo, mas não são mais usados pelo binding; limpeza não realizada pela pausa.
- Obter decisão de Arthur/root sobre baseline antes de retomar. Preservar testes novos e F519 congelado como evidência; não alterar labels nem converter regressão sintética em nota real.
