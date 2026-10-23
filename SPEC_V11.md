# SPEC V11 — Conditional Safety, Pix Grounding e Resposta Parcial por País

**Baseline:** candidate `8c283c8455d02327f39ddb15e1d23e9f81c0a1d4`  
**Auditoria de origem:** `pipeline/getnet_v10_native_network_audit/REPORT.md` — 91,53/100 nominal; `main` permanece em `672c87e`.  
**Objetivo:** corrigir M08, P05 e P06 observados na execução Gemini com rede liberada, preservando o fluxo de Function Calling e sem tratar respostas HTTP 200 como sucesso semântico.

## Arquitetura alvo e contratos

1. **Gate server-side para dependências condicionais (M08).** A solicitação original é a autoridade. Quando uma resposta sobre produto/benefício depende de um predicado verificável de leitura (por exemplo, `firmware == qa-12.08` e `status == Online`), o orquestrador deve executar primeiro somente a leitura operacional do terminal identificado. Uma busca dependente só pode ser agendada após observação tipada, bem-sucedida, do terminal exato e confirmação de todos os predicados. Valor divergente, nulo/ausente, erro, alvo divergente ou evidência ambígua bloqueia a consulta dependente e deixa a unidade pendente. O gate vale também para etapas já propostas pelo modelo no plano inicial; não é instrução apenas no prompt.
2. **Roteamento de Pix Getnet (P05).** Uma pergunta sobre elegibilidade de contas para Pix/Getnet é documental e deve consultar `search_getnet_knowledge` no corpus local, com evidência/citação de `data/getnet_pix_services.md`. A checagem pós-Function-Calling só pode reclassificar uma proposta Web quando o candidato local Getnet é HTTPS, verificado e suficientemente relevante. Perguntas que explicitamente pedem regra pública atual ou fonte regulatória continuam elegíveis a Web Search. Em pedido misto, a unidade de Pix permanece separada da leitura privada do firmware.
3. **Preservação parcial independente por geografia (P06).** Para condições do Crediário no Brasil versus Noruega, o atendimento conserva os fatos brasileiros sustentados pelo chunk Getnet — recebimento em até dois dias úteis, taxa única, sem custo de antecipação — mesmo quando a documentação consultada não cobre a Noruega. A resposta declara o limite documental norueguês sem inferir equivalência ou inexistência do produto. A unidade brasileira fica `fulfilled`, a norueguesa fica `unresolved` ou isenta somente se a semântica autorizar; o resultado geral é parcial. Falha local de orçamento em replanejamento não pode apagar evidência já autorizada nem gerar nova chamada sem ganho.
   - Para a frase comparativa congelada em V10N-P06, o particionador server-side cria unidades literais separadas para o escopo brasileiro e norueguês; não credita a comparação inteira como completa se apenas Brasil tem fonte.

### Interfaces

- `RouterAgent.plan` continua selecionando ferramentas por Function Calling nativo quando Gemini está configurado; o validador local de evidência pode corrigir apenas propostas públicas que contradigam documentação Getnet local verificada.
- `KnowledgeAgent` pode validar a escolha de ferramenta pós-Function-Calling para perguntas de elegibilidade de produto Getnet: usa `search_getnet_knowledge` quando houver evidência local verificada pertinente; pedidos regulatórios/externos explícitos continuam elegíveis a `web_search_general`. A autorização de fatos e citações continua no fluxo normal.
- `MultiAgentOrchestrator.process` permanece responsável por ordem de execução, leitura operacional tipada, avaliação de predicado e supressão de etapa dependente não autorizada.
- `KnowledgeAgent` permanece responsável por obter spans locais, autorização semântica de cobertura por unidade, citações e resposta parcial grounded.
- `RequestCoverageLedger` registra fulfillment somente com evidência atribuível à unidade original; texto do modelo sozinho não concede cobertura.
- A cobertura por país usa unidades literais server-side separadas; respostas parciais preservam o país evidenciado sem promover a comparação inteira a completa.
- `diagnose_terminal_issue` permanece read-only nos casos desta spec. Nenhuma ação remota, mudança de tenant, expansão de escopo ou retry de chamada é introduzido.

## Matriz de requisitos × Régua V4

| Requisito | Critérios V4 afetados | Evidência mínima de aceite |
|---|---|---|
| M08: nenhuma busca dependente após predicado falso/desconhecido | 2 Agentic Reasoning, 4 Tool Engineering, 5 Security, 11 Engineering Judgment | Testes de integração com saída de tool ledger: firmware divergente/null/erro bloqueia Knowledge; firmware+status exatos permitem Knowledge somente depois da leitura. |
| P05: elegibilidade Pix Getnet vai ao corpus interno | 1 Business Fit, 2 Agentic Reasoning, 3 Context/RAG, 4 Tool Engineering, 8 Evaluation | Teste com proposta inicial Web remapeia para Knowledge somente com span Getnet verificado; resposta inclui as três categorias documentadas e citação. Caso regulatório explicitamente externo mantém Web. |
| P05 misto: Pix e firmware null com autoridade separada | 1 Business Fit, 3 Context/RAG, 4 Tool Engineering, 5 Security, 8 Evaluation | Handler de ponta a ponta preserva Pix como fato parcial/grounded, firmware ausente como pendente, alvo exato e sem campos extras ou dados de outro terminal. |
| P06: fato Brasil mantido apesar de Noruega desconhecida | 1 Business Fit, 2 Agentic Reasoning, 3 Context/RAG, 6 Resilience, 8 Evaluation, 9 Reliability | Handler/ledger mostra Brasil fulfilled + Noruega unresolved, com citação no chunk Crediário e sem replan após ausência documental independente. |
| Promoção condicionada | 8 Evaluation, 10 Production | Suíte completa, evaluator offline documentado, auditoria nativa congelada independente e CI verde no SHA candidato. Publicar em `main` apenas se nova nota independente sustentar 95+ e Semantic Quality Gate passar. |

## Plano de validação

- RED/GREEN TDD para cada comportamento; os testes devem observar chamadas de ferramenta, ordem, alvo, evidência, ledger e conteúdo entregue, não só o prompt ou o HTTP status.
- Executar pytest completo no worktree usando `GEMINI_API_KEY=''`; registrar eventual warning upstream sem mascará-lo.
- Executar `python -m app.evals.evaluator` offline e reportar os campos literalmente, sem converter harness sintético em nota semântica.
- Manter as rodadas V10F/V10N intactas como histórico; após implementação e CI, obter amostra nativa nova, congelada antes de chamadas e auditada por agente independente. Não substituir casos por erro nem repetir a mesma amostra para selecionar resultado.
- Manter `main` em `672c87e` enquanto nota <95, gate semântico não satisfeito ou CI do SHA final não verde.

## Limites

- O corpus é limitado e usa mirrors Getnet revisados por agente; não equivale a validação humana externa nem cobertura global de termos por país.
- Os predicados condicionais aceitos são uma gramática finita explícita; linguagem ambígua deve permanecer pendente, não ser interpretada permissivamente.
- Três correções não garantem 95+. A nota depende da auditoria V4 independente com os resultados observados, e o conjunto nativo é purposive e pequeno.
