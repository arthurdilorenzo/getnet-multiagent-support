# V9 — disponibilidade observável e replanejamento com ganho

2026-10-23. Baseline385c006, auditoriaV8 93,65nominais não oficiais. Main672c87e protegido. Arthur autorizou spec/dois agentes, testes e auditoria honesta; publicação somente95+ sustentado eCIverde.

## Arquitetura e invariantes

Function calling nativo continua decidindo ferramentas/delegações; gates locais conservam tenant, propósito, alvo, data, campo e consentimento. Gemini configurado não vira heurística offline após falha. Duas passagens/spans de RAG preservadas; nenhuma resposta é inventada por disponibilidade. Budgets30s/turno,60s/pedido,32000tokens,2048output e zero retries continuam. Não ativar cobrança/plano nem expor.env/chave/PII. Não alterar labels/gates/corpus/vídeo ou score por testes. Casos V8 são regressões conhecidas.

## Frente A — Gemini (crit.Agentic15/RAG10/Observability6/Production5)

Ownership: app/agents/function_calling.py, app/config.py, testes novos test_gemini_availability_v9.py. Diagnóstico/recibos fora produto em ../getnet_v9_evaluation/. Não editar router/orchestrator/constraints.

Antes de mudar modelo/configurar recuperação, testar hipótese de falha por modelo/formato e distinguir erroexterno/protocolo/limite/breakerlocal. Comparação pré-declarada limitada: modelo atual gemini-3.1-flash-lite vs gemini-2.5-flash-lite, ambos documentados para function calling/structured outputs. Três formas: chamada nativa mínima, schema real de routing, schema compacto semântico com caso sintético conhecido. Duas repetições por forma/modelo=12 gerações máximas; ordemintercaladaAB/BA,20s mínimos entre gerações, zero retry/substituição. Schemas/inputshashes congelados antesexec, budgets equivalentes e thinking apropriado/documentado da família. Nunca dados de carreira/clientes reais. Se candidato indisponível no catálogo/acesso, registrar, não abrir leque de modelos sem novo plano.

Sucesso diagnóstico exige protocolo/argumentos/estado válidos e não sóHTTP200. Escolha só se candidato demonstra maior sucessoprotocolar; em empate, semnovoerrosegurança e menorlatência, mantendo limitaçõesn12. Não mudar modelo para apagar as falhasanteriores nem alegarSLO. Sem vencedor defensável, preservar default e relatar bloqueio. Nova auditoria semântica final decidirá qualidade/95, não benchmarkdeprotocolo.

Fontes primárias lidas antes do plano: https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite?hl=en ; https://ai.google.dev/gemini-api/docs/troubleshooting ; https://ai.google.dev/gemini-api/docs/thought-signatures.md .503/504/429 têm classes documentadas, mas não provam causa interna das ocorrências V8.

AceitesRED/GREEN: timeout/deadline/outputbudget não ampliados; request/provider/protocol failures separados sempayload; família de thinking não aplicada a modelo incompatível; configuração/default condiz com experimento, override explícito preservado. Limite local/backpressure se útil só failclosed, sem fila invisível que estoure deadline. Não confundir isso com provideravailable.

## Frente B — ganho de planejamento (Agentic15/Tools10/Evaluation14/Judgment4)

Ownership: router_agent.py,orchestrator.py e testes novos test_planning_gain_v9.py. Pode extrair helper de política separado, coordenar se alterarfulfillment/request_constraints/request_units. Não editarfunction_calling/config/fixturesavaliador.

Reproduzir P04: leitura positiva+proibição de ação reconhecida foi tratada como consulta documental e replan, custando59s. Remover esse trabalho sem apagar unidade original/consentimento, sem marcar restrição automaticamente como fato respondido. Parser de proibição finito deve consumir cláusula inteira, alvo explícito e ação suportada; condição/desconhecido/negativa de outro significado permanece incerta. Nenhuma nova isenção de conteúdo sem prova. Preservar visibilidade da restrição e resultado real da leitura; partial conservador é preferível a falsa suficiência.

Aplicar plano inicial e replan: unidades já satisfeitas, negativas de ação reconhecidas ou já agendadas não dão próximo trabalho; subquery/rationale não criam ganho/propósito. Após erro de provedor sem observação acionável, não consultar outro especialista só para gastar. Preservar HITLcrítico, falha operacional recuperável, suporte→documentação de código observado e necessidades positivas mistas realmente pendentes. RED/GREEN metamórficos PT/EN, IDs diferentes, ordens/conjunções, mensagensmaliciosas e condições anexas, zero escrita/proposta para veto.

## Integração e auditoria

Root revisa diffs/testes focados, pytest-v completo, CLIharness syntheticHTTP identificado semrede, DockerPython3.11/6smokes eCI SHAexato. Sem novo modelo escolhido apenas por teste local. Freeze limpo antes de novo auditor independente criar amostra/labels/manifests inéditos e selecionar nativos; uma rodada sem substituir falhas e condiçõesdeenvio pré-declaradas. Separar disponibilidade, protocolo, utilidade e fidelidade por necessidade; onze pesos/gates completos, nota nominalnãooficial e confiança explícita. Main só95+ defensável eCIverde; se promover, confirmarCI tambémemmain. Registrar falhas e decisões em relatório detalhado+WORKLOG.

## Refinamento pré-experimento por documentação atual

A abertura integral das fontes, ANTES das gerações diagnósticas, revelou restrição de acesso ao2.5 para usuários prévios e recomendação explícita3.1FlashLite→3.5FlashLite. Trocar o candidato pré-grade para `gemini-3.5-flash-lite` documentado como estável/comfunctioncalling/structuredoutputs; mesma regra de12calls/20s/ABBA/budgets e seleção, não ampliar leque.2.5 não será testado nesta grade. Thinking3.5 defaultminimal e suporta minimal segundo https://ai.google.dev/gemini-api/docs/thinking (tabela) ; modelo https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite ; restrição2.5 https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite?hl=en ; migração https://ai.google.dev/gemini-api/docs/deprecations . Catálogo só confirma exposição/acesso, não sucesso de chamada. Se3.5 indisponível àconta, registrar e nãoinventar default/lancarbuscaindefinida.
