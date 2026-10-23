# V10 — roteamento documental, múltiplos alvos e utilidade parcial

23/10/2026. Baseline e6dd7dd; auditoria V9: 93,79/100 nominais, não oficiais. Main 672c87e preservado. Spec registrada antes de qualquer alteração de implementação desta rodada.

## Problemas e contratos alvo

V9 D04 desviou instrução de SIM documentada para snippets web; O02 manteve duas leituras com `plus` numa unidade ambígua; B01/B02 perderam fatos locais conhecidos quando outra parte era ausente ou não suportada. Casos V9 agora são regressões, nunca holdout novo.

Manter function calling nativo: o modelo escolhe especialistas/tools; invariantes locais vinculam identidade, propósito, alvo e evidência ao texto original. Nenhum fallback heurístico quando chave configurada. Não trocar modelo, aumentar budgets/tentativas, relaxar gates nem transformar abstenção em cumprimento. Nenhum corpus/label alterado para adequar respostas. Unidade canônica conserva substring, offsets e cobertura literal integral; IDs são determinísticos para a mesma mensagem, nunca derivados de rationale/subquery do modelo. Repartição deve permitir atribuição inequívoca sem permitir que alvo/propósito de outra cláusula autorizem acesso.

Cada necessidade suportada recebe somente spans verificados em duas passagens ou observação tipada própria. Necessidade desconhecida, estrangeira, campo ausente ou proibida permanece explicitamente pendente; não apaga fatos conhecidos independentes nem obtém crédito falso. País/condição/negação não podem ser descartados pelo particionamento. Falha estrutural de protocolo continua fail-closed; decisão semântica de uma unidade não deve obrigar abstenção das demais aprovadas.

## Frentes e interfaces

A — agente documental: app/agents/router_agent.py, app/agents/knowledge_agent.py, app/rag/semantic.py e prompts relacionados; testes novos test_documentary_partial_v10.py. Investigar traces V9, reproduzir RED e corrigir roteamento/composição nativa por unidade. Especialista documental tem prioridade instrucional para produtos Getnet e deve manter fronteira entre informação local e consulta privada, sem roteamento keyword antes do modelo. Coordenar alterações de ownership; não editar request_units/operational_intent/constraints.

B — agente múltiplos alvos: app/security/request_units.py, app/security/operational_intent.py, app/security/request_constraints.py e testes novos test_multi_target_v10.py. Reproduzir `plus` e variantes PT/EN, ampliar fronteiras literais com cuidado de nomes/conjunções/condições, leituras distintas e elipse. Não alterar router/knowledge/semantic/orchestrator. Não relaxar alvo/tenant/propósito/TXN/consentimento; ambiguidades reais continuam bloqueadas.

Integrador: revisar contratos cruzados e app/agents/orchestrator.py somente se reprodução mostrar necessidade. Ajustar documentação factual após implementação. Dois agentes implementadores; auditor novo separado somente após freeze.

## Matriz de requisitos e aceite

|Requisito|Critérios V4|Aceite|
|Documento antes de web genérica quando tema é Getnet documentado|Business8, Agentic15, RAG10|Regressões SIM PT/EN e paráfrases: native tool declarations/prompts mantêm decisão do modelo; fatos slot/orientação/teste somente com fonte verificável; web atual genuína continua possível|
|Múltiplas leituras próprias independentes|Tools10, Security13, Agentic15|plus/PT equivalentes/ordens/IDs diferentes: consulta cada alvo e campo corretos, nenhuma troca entre alvos; fora tenant, TXN errado, veto e elipse ambígua negados|
|Resposta parcial por necessidade|RAG10, Business8, Judgment4|Público conhecido + firmware ausente e Brasil + país não suportado: retém fatos provados, mantém ausente/país pendente, status partial e ledger correto; zero garantia indevida|
|Particionamento íntegro|Security13, Evaluation14|Offsets/substrings recombinam mensagem, IDs determinísticos, condições/negações não somem, atribuição sem rationale inventado; regressões V3–V9 passam|
|Verificação honesta|Evaluation14, Observability6, Production5|pytest completo, evaluator declarado sintético, Docker e CI SHA exato; auditor independente congela novos casos e labels após código, amostra nativa pré-selecionada uma execução sem substituições|

## Protocolo de evidência e entrega

Agentes registram RED/GREEN e causas em ../getnet_v10_evaluation/, sem ler/expor .env/chave ou fazer chamadas externas. Integrador pode executar Gemini autorizado apenas após protocolo de amostra final congelado; modelo3.5, mesmos budgets e zero retry,20s após cada caso, no máximo oito casos nativos da nova auditoria. Nunca rerodar falha para substituí-la. APIs externas reais e fixtures sintéticas devem ser distinguidas.

Auditor avalia onze pesos e quatro gates; nota nominal não oficial e limites explícitos, sem score automático por testes. Main só se >=95 sustentado E CI verde; CI candidato pode rodar em branch. Se aprovado e promovido, confirmar também CI main. Se abaixo95, preservar main e registrar pendências concretas, sem alegar objetivo alcançado. Registrar resultado detalhado e resumo no WORKLOG.

## Sequência de execução verificável

- [x] A: criar testes RED em `tests/test_documentary_partial_v10.py` reproduzindo composição parcial e seleção de ferramentas; executar antes de corrigir. Implementar e rodar regressões semânticas/router/knowledge. Fluxo real do handler com transporte SDK simulado verifica renderer/ledger, sem alegar aderência do Gemini.
- [x] B: criar testes RED em `tests/test_multi_target_v10.py` para `plus`/equivalentes, offsets e autoridade por alvo; executar antes de corrigir. Implementar e verificar duas leituras reais do handler, inclusive campo ausente e alvo estrangeiro. Ampliação de ownership para fulfillment documentada antes de sua alteração.
- [x] Integrador: revisar diffs, executar `.venv/bin/python -m pytest -v`, `.venv/bin/python -m app.evals.evaluator`, AST e `git diff --check`; registrar saídas fora do produto. Corrigir falhas reproduzidas sem mudar labels/gates.
- [ ] Congelar commit limpo, construir/verificar Docker e executar CI somente no branch candidato. Nova auditoria independente recebe SHA congelado, escreve amostra/labels antes da única execução nativa autorizada. Conferir soma dos onze pesos e limitações do parecer.
- [ ] Registrar entrega/WORKLOG e verificar main remoto. Publicação em main depende simultaneamente da auditoria >=95 sustentada e CI verde; confirmar novo CI de main se houver promoção.

Foco da revisão cruzada: condições globais não podem desaparecer ao permitir parcial; fragmentos entre alvos não podem doar identidade/propósito; conectores em nomes não devem criar leituras; fonte web genérica não satisfaz instrução Getnet; ausência de campo e país desconhecido não justificam apagar fato independente verificado.

## Addendum V10.1 — correção empírica do desvio documental

Após a implementação V10 congelada em `0bb3b32`, revisão independente reproduziu que RouterAgent sem chave envia o pedido em inglês para instalação do SIM Algar a `CUSTOMER_SUPPORT_AGENT` (e, no trace V9, Gemini escolheu busca web genérica apesar do manual local). O teste anterior só verificava o prompt e permitia escolha WEB_SEARCH; portanto, o aceite D04 não estava demonstrado. A suite local mantém evidência do corpus com instruções Algar/SIM1/SIM2/teste de comunicação.

Contrato corrigido: function calling nativo continua escolhendo a ação inicialmente. Depois da seleção, um validador local de elegibilidade pode redirecionar uma necessidade de procedimento/funcionamento de produto Getnet para KNOWLEDGE_AGENT somente quando o próprio retrieval retorna fonte primária local pertinente acima de limiar já definido pelo sistema. O validador usa resultado/capacidade de retrieval e escopo tipado; não usa lista de palavras-chave nem substitui a chamada Gemini. Sem evidência local suficiente, preservar busca web para informação externa atual quando pertinente ou declarar necessidade sem resolução. Nunca responder instruções SIM/campo técnico com snippet web genérico; não inventar instruções ausentes do span.

RED/GREEN: reproduzir variações PT/EN e paráfrases de instalação Getnet, incluindo request realmente coberto e um request sem conteúdo local; comprovar que a consulta coberta recebe apenas os spans/citações existentes do manual através do caminho de Knowledge/RAG/ledger, enquanto pedido fora do corpus não é falsamente respondido. Guardar a escolha inicial do modelo e o redirecionamento justificado, sem alegar aderência do Gemini real sem execução. O restante das invariantes e critérios da matriz V10 continua vigente.

## Addendum V10.2 — lacuna de integração de resposta parcial

Revisão independente encontrou contratos V10 para parcial documental e leitura operacional, mas nenhum teste de integração que combine no mesmo turno um fato público coberto (prazo Pix) e firmware `None` do terminal. Integrador deve acrescentar teste RED/GREEN pelo handler/orquestrador, garantindo resposta com o fato coberto, firmware explicitamente pendente/indisponível, status `partial`, ledger coberto/unresolved e sem emprestar campo de outro alvo. Não altera expected labels nem critérios; apenas demonstra o contrato já declarado.
