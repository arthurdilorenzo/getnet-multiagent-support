# AF05 — três pendências reportadas pelo Antigravity

Data: 2026-10-23. Baseline af050745e6fac8613de7de928ff9aa4692449bc2. Spec anterior ao reparo: SPEC_AF05_REMAINDER.md. Escopo: A4/A5/A8, sem rede/Gemini/novo score/push/main, sem novos agentes implementadores. Revisor anterior reutilizado somente para o diff restrito.

## Evidência recebida

Raw do teste em `pipeline/getnet_v11_native_af05074_fresh_20261023/native-chat-results.json` no workspace pai: code_sha af05074, code_dirty false, synthetic_data true, independent_evaluation false, semantic_grade_assigned false. Raw SHA256 393d0fb6548acf4bdeef4833e4251436f8d384f6b080b2a4f2280a1f48fe9f43; oito itens do manifesto conferidos. Bundle consumido/intocado, sem reexecução. Arthur reportou 5 completos/1 parcial/2 falhas e estimativa de cerca92. Não existe tabela V4 ponderada nova neste recibo;92 fica como estimativa reportada, não nota independente confirmada.

A4: uma chamada diagnóstica, sinal correto; segundo alvo não consultado. Também havia qualificadores de formato como pendências e resposta com cabeçalhos apesar do pedido só células.
A5: linguagem nova declarava conta de parceiro e ausência de permissão administrativa; o detector de propriedade não reconheceu. Handler recusou com TERMINAL_NOT_FOUND no titular correto, sem dado estrangeiro. Faltava negar antes da consulta.
A8: proposta explícita referenciava ação/alvo na frase anterior e usava 'Registre', formas não aceitas pelo parser. **Não foi observada tentativa de execução física:** execute_terminal_remote_action é ferramenta que SOMENTE prepara proposta. Houve recusa ACTION_NOT_AUTHORIZED, sem proposta nem execução; resumo externo foi corrigido nesse ponto.

## Reparo

- Modelo continua escolhendo a família diagnóstica por FC. Ao parar depois dessa escolha, o servidor completa somente alvos de leitura explicitamente atribuídos ainda não tentados, dentro do mesmo MAX_TOOL_TURNS e passando por constrain_call/tenant. Não faz compensação de escrita, não repete campo ausente ou chamada com erro e não aumenta budget/modelo/retry. Trace identifica origem bounded_read_contract. Mecanismo não contém IDs nem valores AF05 no produto.
- Pedido de células ID/valor é restrição de apresentação; fonte/formato factual não são confundidos. Normalização de prefixo de apresentação e vocabulário de campos conservam prova typed; resultado estrito é duas linhas ID/tab/valor sem cabeçalho/conselhos. Diretiva com nova ação não é absorvida como formato.
- Ausência explícita de autorização na conta alheia nega terminal único antes de FC Support/registry. Proprietário alheio já reconhecido também recebe recusa antecipada; handler permanece segunda barreira. Recusa é evidence policy_decision, não chamada operacional fictícia.
- Proposta pode herdar somente ação/alvo únicos da cláusula imediatamente anterior com pedido explícito. Veto de proposta, hipótese, citação, alvo divergente e ambiguidade continuam negados. Somente proposta pendente é gravada; confirmação JWT/titular/hash/idempotência posterior permanece intacta.

## Verificação

Regressões novas HTTP usam fixture sintética e transporte FC sintético que retorna somente primeira maquininha e depois para; registry real com spy prova duas consultas exatas, ledger complete e saída minimizada. Estrangeiro exige tools_called vazio e registry não chamado. Proposta exige PENDING_CONFIRMATION e zero terminal_actions. Negativos de formato/veto/hipótese/histórico também exercitados.

RED inicial dos3casos preservado em verification/af05/native_regressions_red.txt; o pytest.fail disparado no spy estrangeiro também provocou erro de teardown do portal durante o RED, não falha de produção adicional. Primeira integral4falhas foi expectativa antiga de FC ainda chamado para estrangeiro; atualizada somente para exigir0, mantendo bloqueio registry/dados/persistência. Recusa pública preservada pela projeção. Outros recibos intermediários preservados, sem apagar falhas.

Revisão restrita AF05_SCOPED_REVIEW.md levantou herança demasiadamente ampla de contexto; novo teste histórico reproduziu e reparo limitou à frase anterior. Asserts positivos foram mantidos. Uma extração de ID após normalização que removia hífens foi corrigida para usar o literal; recibos anteriores com essa falha permanecem.

Recibos finais: verification/af05/focused_verified.txt e full_verified.txt. Runtime Python3.13, wrapper offline bloqueia IP sockets antes imports; AST com gramática3.11/no_print/diff verificados. Evaluator offline sintético em verification/af05/evaluator.txt não é nota V4.

Nova avaliação real e CI exato NÃO executados. As correções locais não autorizam declarar95 ou promoção. Main permanece672c87e, nenhuma chamada de push.

Resultado final confirmado: **43 testes focados passaram**, 1 warning/1,12s; **1556 testes integrais passaram**, 1 warning/12,91s. check_no_print e git diff --check exit0. Revisão restrita sem bloqueador concreto remanescente no diff avaliado; não é avaliação nativa nova.
