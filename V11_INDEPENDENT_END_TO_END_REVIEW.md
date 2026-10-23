# V11 — revisão independente ponta a ponta

Data: 2026-10-23. Revisor separado da implementação. Baseline: `f519090b24fd44341e3ba5c0733c5c059255714b`.

## Estado da revisão

Revisão inicial e rechecks focados concluídos. Os três findings reproduzidos foram corrigidos e fechados no recorte exercitado. Não restam bloqueadores concretos observados nesse recorte. O parecer é de conformidade do reparo local revisado, com as limitações descritas abaixo; não aprova semântica nativa, nota, CI ou promoção. Nenhuma suíte integral, runner nativo, rede, Gemini ou dado privado foi executado pelo revisor.

Autoridade: `SPEC_V11_END_TO_END_REPAIR.md`, handoff e adjudicação F519 no workspace pai. A amostra F519 permanece consumida e imutável. A adjudicação histórica não é reavaliada aqui.

## Leitura inicial do fluxo real

`POST /api/chat` serializa `ChatResponse` devolvido pelo orchestrator. No baseline a resposta do assistente é persistida antes da construção do envelope final; `intent`, `trace`, `sources`, `citations`, `evidence`, `action_proposals` e `request_coverage` possuem caminhos separados. PII masking não implementa minimização por campo/terminal. Portanto, alterar somente renderer ou evidence do especialista não fecha o contrato público A4.

A condição atualmente depende de parser finito que inclui a frase da conclusão; condition=None interrompe até o preflight. A correção deve separar a autorização read-only da autorização do conteúdo dependente, mantendo prova tipada/tenant-scoped antes de liberar o documento.

O contrato documental precisa conservar tarefa e contexto no argumento efetivamente enviado à ferramenta, inclusive após replan. Ledger com evidence_count positivo verifica atribuição, não verdade semântica: não basta marcar unidades como cumpridas ou exempt.

Propostas são persistidas por action_policy, com confirmação JWT externa ao agente, hash e isolamento. A correção de gramática deve autorizar somente propor quando expressamente pedido, sem ampliar execução. O registry diferencia failed de unknown para escrita de resultado indeterminado; preservar essa distinção.

## Riscos e critérios de revisão comunicados antes da implementação

1. **Envelope público integral:** projetar texto/evidence/trace/intent/citações/fontes e demais campos capazes de carregar fatos derivados, antes de persistir conteúdo público. Testar sentinelas distintas em múltiplas superfícies; pedidos independentes legítimos devem sobreviver. Registrar observabilidade interna rica não autoriza retorno público rico.
2. **Estado unknown:** a expressão da spec “Toda recusa/erro é failed” aplica-se a erros/recusas conhecidos. Não converter timeout de escrita ou resultado indeterminado em failed, pois isso destrói reconciliação e pode habilitar repetição insegura.
3. **A5 por alvo:** bloquear declaração explícita de propriedade estrangeira antes de registry/handler, inclusive PT/EN. Palavra “fornecedor” em outra cláusula, citação ou negação não é por si declaração de que todo alvo é estrangeiro. Testar próprio+estrangeiro no mesmo pedido.
4. **Condições desconhecidas:** leitura segura não implica liberação do documento; disjunções, negacões, ranges, atributos extras e ambiguidade continuam fail-closed no dependente. Resultado diagnóstico não pode, por si, cumprir o pedido documental.
5. **Aceitação:** HTTP 200 e testes de helper não demonstram o contrato. Exigir inspeção do JSON HTTP, argumentos reais, persistência e comportamento negativo, por transporte sintético identificado. Sem inferência de generalização Gemini.

## Revisão do candidato — interrompida por mudança de direção

O integrador liberou revisão de produto estável antes do commit/suíte final. A pedido posterior do integrador, a revisão ampla foi pausada para considerar retomar a base 93+. Nenhuma aprovação do diff ou da suíte final é dada. O produto estava em working tree sobre f519090; os achados abaixo se referem ao estado observado, sem SHA de commit final. As correções dos donos posteriores a estes probes ainda não foram revisadas.

### P1 — ausência de diagnóstico desabilita a barreira de texto/evidence

Em `app/security/public_projection.py`, `project_specialist` devolve o output intacto quando não há resultado `diagnose_terminal_issue`; `project_public_response` recompõe o texto somente quando `has_diagnostic` é verdadeiro. Assim, sob pedido que permite só sinal/conexão, um especialista sem diagnóstico pode devolver prosa/evidence não autorizada e ela atravessa HTTP e memória. A resposta também pode manter `answer_status=complete` baseado no relatório do especialista.

Reprodução: `/private/tmp/test_v11_review_no_diagnostic.py` deriva do teste boundary HTTP do integrador e altera somente `tool_results=[]` e `structured_data={}` nos outputs sintéticos. O wrapper offline (sockets IP proibidos) executou 1 teste, que falhou em `assert 'SENTINEL_' not in result.text`: HTTP 200 retornou `SENTINEL_PROSE` e evidence não comprovada. Isso demonstra uma lacuna defensiva com output sintético inválido; não demonstra que Gemini em produção já produziu esse output. Spec 4 requer que ausência de reconhecimento/prova não libere dados.

### P2 — diretiva de fonte pode engolir uma nova pergunta factual

Entrada: `Explain Getnet Pix eligibility. Answer using Getnet material: what is the cancellation fee?`

`KnowledgeAgent._bind_documentary_scopes` com todas as unidades atribuídas e query original emite apenas `Explain Getnet Pix eligibility.`. `_SOURCE_DIRECTIVE` aceita a pergunta adicional como qualifier e `factual_documentary_text` a remove. A pergunta final possui apenas um ponto de interrogação terminal e não é rejeitada pelo guard limitado a `?` seguido de conteúdo ou a certos verbos após and/e. Viola spec 1/2: cláusula substantiva é promovida a qualifier, e o argumento real perde tarefa.

### P2 — declaração explícita plural estrangeira não bloqueia ambos os alvos

Entrada: `Check signal of GET-SMART-QA-44031 and connection of GET-CLASSICA-QA-44032; both terminals belong to my supplier.`

`declared_foreign_terminal` retorna False para ambos, pois a sentença de propriedade não tem ID e o pedido global tem dois IDs. O requisito de rejeição pré-tool explícita não é satisfeito para a referência plural inequívoca. A barreira tenant-scoped no handler continua valendo; isto não é evidência de divulgação ou leitura cross-tenant.

Os dois últimos probes estão em `/private/tmp/test_v11_review_contract_edges.py`: 2 falhas focadas esperadas. Nenhum produto/teste/label do repositório foi modificado pelo revisor. Os probes temporários apenas reproduzem contratos; não são nova amostra nativa nem suíte integral.

## Qualidade e compliance parcial

A integração posiciona o objeto ChatResponse depois de agregação e antes de `append_message`, elimina o minimizador anterior dependente de vocabulário permissivo e conserva observações internas ricas para o gate tipado. O validador de ToolExecutionResult corrige falha conhecida com default completed preservando unknown/pending explícitos. A separação de proposta/execução mantém o endpoint JWT/hash existente; os testes novos cobrem persistência e confirmação/replay com transporte sintético.

Isso é progresso arquitetural verificável em leitura, mas os três gaps acima impedem considerar a spec integralmente atendida. A revisão não completou todos os arquivos/recibos nem revisou o estado final da suíte, e não atribui nota, CI verde ou autorização de promoção. Resultados reportados pelos implementadores permanecem evidência deles; não foram rerodados integralmente pelo revisor.


## Comparação estrutural curta da base anterior

O SHA `e6dd7dd2adb093b5a6536bf2bf2ae70f084b0599` existe, com assunto `Select measured Gemini model and stop non-actionable planning work`. Entre ele e f519090 há 33 commits; diff de app: 12 arquivos, +1195/-50 linhas, com maior acréscimo em Router, Knowledge e orchestrator e novos parsers condicionais/documentais. Isso demonstra superfície menor na base anterior e torna defensável avaliar um reparo novo a partir dela, preservando as regressões como contratos. Não demonstra melhor qualidade, segurança, generalização nem nota 93+: a auditoria histórica precisa ser considerada pelo integrador e os defeitos antigos não desaparecem por checkout. Nenhum checkout ou alteração de produto foi feito nesta comparação read-only.

## Retomada e recheck focado

O integrador esclareceu que a pausa foi temporária e retomou o candidato original, sem rollback. Recheck restrito aos três findings e últimas integrações, sem repetir revisão ampla ou suíte integral.

- **P2 diretiva:** probe original passou; a pergunta de tarifa permanece no argumento.
- **P2 plural:** probe original passou; ambos os IDs recebem declaração estrangeira.
- **P1 disclosure:** probe original agora passa o assert de ausência das sentinelas e exibe corretamente indisponibilidade dos dados, em vez de inventar valores. O teste temporário original ainda falha no assert herdado de sucesso que esperava sinal/conexão apesar de zero tool_results; essa falha não representa regressão do produto.
- **P1 resíduo de ledger ainda aberto:** variante `/private/tmp/test_v11_review_no_diagnostic_recheck.py` troca apenas o assert de sucesso impróprio por indisponibilidade/abstention/evidence vazia e exige que as unidades de campo não continuem fulfilled. A última exigência falhou. O texto/evidence estão limpos e o status deixou de ser complete, mas a segunda unidade iniciada com `; informe` continua fulfilled quando o mock alegava covered sem tool_result. `project_public_response` usa canonical_terminal_read_text sem remover o prefixo de pontuação antes de `_property_read` ao construir affected_field_ids. Este é resíduo do mesmo achado de coerência de prova, não nova frente.

Revisão focada das últimas integrações: fixture positiva planning_gain passou a incluir ToolExecutionResult próprio, ID exato e status Offline, sem reduzir assertion; coerente com o novo contrato. Inclusão do footer de unidades unresolved/veto em trusted_segments usa texto literal do usuário e composição do servidor, preservando o limite de prosa arbitrária.

Recibo integral lido neste momento: `verification/v11_e2e/full_final.txt` tinha 1547 passed e 1 failed em planning_gain. Integrador corrigiu a fixture e executa validação posterior; não atribuo passe integral até o recibo novo.

## Parecer final do reparo local revisado

**Os três findings estão fechados no escopo reproduzido.** Recheck independente final de P1/ledger: `/private/tmp/test_v11_review_no_diagnostic_recheck.py`, **1 passed**, um aviso upstream. O caso verifica HTTP serializado sem diagnóstico, prosa/evidence sentinelas ausentes, resposta indisponível, abstention, ausência de estado complete, unidades de campo sem fulfilled fictício e memória assistente igual ao texto HTTP. Os dois P2 passaram no probe original sem alterações.

A última correção substitui o reconhecimento restrito por `_property_read` por mapeamento de alvo/campo normalizado e prova tipada vinculada ao titular; o estado alegado `covered` pelo output não é suficiente. Assim o resíduo `; informe` do segundo par está corrigido. Observações internas continuam disponíveis ao supervisor e não são publicadas por ChatResponse.

### Conformidade com a spec e qualidade

| Contrato | Parecer no recorte revisado | Evidência e limite |
|---|---|---|
| 1. Unidades e qualifiers | Atendido pelos casos novos e correções revisadas | Offsets/literal preservados; whitespace filtrado; fonte/formato vinculados; pergunta substantiva do probe não é descartada. Gramática finita, sem prova universal de segmentação. |
| 2. Escopo documental e replan | Atendido pelos fluxos de regressão revisados | Router e Knowledge reconstruem escopo canônico; testes novos inspecionam argumentos reais e preservação Brasil após Web falhar. Contexto público é limitado à tarefa adjacente e unidade privada independente é filtrada. Identidade referencial ainda depende de vocabulário finito. |
| 3. Condição tipada | Atendido no fluxo e variantes revisados | Leitura própria separada da liberação do dependente; gate exige titular/ID/status/firmware tipados. Testes legados mantêm dependente bloqueado sob condição desconhecida e agora exigem apenas read-only. Fixture positiva passa a trazer prova tenant explícita. Não foi feito novo julgamento nativo. |
| 4. Egress/persistência | Atendido no envelope e probes focados | Objeto completo projetado depois da agregação e antes da memória; texto/evidence/intent/trace/citações/fontes e coerência do ledger exercitados. Sem diagnóstico não libera prosa/observação arbitrária de Support. Pedidos independentes permanecem pelo pipeline de verificação existente. |
| 5. Estrangeiro pré-tool | Atendido em PT/EN singular/plural testados | Spy de registry e probe de ownership; defesa do handler preservada. Não transforma label pré-tool em prova de falha crítica cross-tenant. |
| 6. Proposta/execução | Atendido no diff e testes de contrato revisados | Autorização explícita de proposta separada de execução; negativos informativos/hipotéticos/citados/vetados; persistência pending e confirmação JWT/hash/replay cobertas pelas regressões dos implementadores. Erro default completed corrigido; unknown/pending explícitos preservados. |
| 7. Web indisponível | Atendido nos casos sintéticos previstos | Mensagem explicita incidente não confirmado com assunto/local solicitado; não inventa pane/citação. Indisponibilidade real da Web não foi reavaliada. |

**Qualidade:** a barreira de egress final e a preservação de prova interna tratam os defeitos na fronteira correta. Os testes HTTP e spies verificam mais que helper isolado. As alterações de fixtures/assertions legadas examinadas refletem contratos novos expressos antes do código; não relaxam o gate documental, autorização de execução ou isolamento. O mock planning_gain recebe evidência tipada que faltava, mantendo assertions de comportamento. O footer restaurado usa texto literal do pedido e montagem do servidor, preservando pendências sem aceitar prosa arbitrária do modelo.

**Limites e riscos residuais não bloqueantes:** a arquitetura mantém vários reconhecedores finitos de linguagem e vocabulário de produtos; não se pode extrapolar as regressões para compreensão universal. A projeção reduz detalhes públicos de observabilidade deliberadamente. O revisor não revisou/executou uma nova amostra Gemini, não reexecutou suíte integral, não confirmou CI remoto nem produção. Os recibos de implementadores/integrador são considerados somente pelo que efetivamente registram. Não há nota nova, equivalência com 95+, autorização de promoção ou reclassificação da amostra F519.

## Recibos finais conferidos pelo revisor

- `verification/v11_e2e/full_release.txt`: **1548 passed, 1 warning in 13.83s**. Resultado terminal observado no arquivo; integrador informa exit 0. O revisor leu o recibo e não repetiu a suíte. O aviso é depreciação Starlette/httpx upstream.
- `verification/v11_e2e/evaluator_release.txt`: execução offline sintética, exit 0 informado pelo integrador. O overall 0.9231 reportado pelo evaluator mede checks próprios limitados; não é nota de desafio, utilidade, entailment ou prova Gemini. O próprio recibo declara esses limites.
- `verification/v11_e2e/static.txt`: Python 3.11 AST, **102 arquivos, 0 erros**; check_no_print e git diff --check registram exit 0.

Esses recibos substituem a execução full_final com uma falha de fixture planning_gain, cuja correção foi revisada acima. Aprovação limitada ao reparo local revisado e às evidências offline explicitadas; nenhum bloqueador concreto permanece dos findings examinados. O candidato ainda estava em working tree sobre f519090 quando revisado: o commit local posterior e qualquer CI precisam ser vinculados pelo integrador ao estado efetivamente validado. Não publicar main com base neste parecer.
