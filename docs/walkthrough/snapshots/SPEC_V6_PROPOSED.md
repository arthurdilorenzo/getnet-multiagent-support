# V6 — protocolo compacto, cobertura observável e entrega demonstrável

2026-10-23. Baseline congelado `d7e2125`, auditoria independente **89,27/100 nominais**, sem95 sustentado. Main preservado em `672c87e`. Esta spec precede código V6. Fontes: [auditoria V5](../GETNET_V4_AUDIT_V5_2026-10-23.md), [opções do protocolo](../GETNET_V6_PROTOCOL_OPTIONS_2026-10-23.md) e recibos em `../getnet_v5_evaluation/`.

## Problemas observados e decisão

-18 testes nativos:4 completos corretos,1 parcial útil,9 falhas503/504,4 rejeições de protocolo após STOP. Não houve alegação proibida nos5 publicáveis;13 falhas não comprovam fidelidade. Não repetir chamadas para substituir falhas dessa amostra.
- A análise rejeitada não fornece código específico; o modelo repete texto/offset/hash já pertencentes ao servidor. Adotar decisões por IDs e reconstrução canônica local. A redundância não foi demonstrada como causa de todos os erros nem de503/504; medir sem prometer cura.
- Metadados de cobertura não têm ledger entre especialistas; cláusula documental pode sumir em execução operacional. Memória persiste entidade mas uma referência sem deíticos reconhecidos não alcança diagnóstico. Corrigir os contratos, sem decorar as24 perguntas.
- Vídeo pedido no enunciado ainda não existe. Produzir walkthrough técnico local com narração sintética explicitada e evidências reais; não simular autoria humana ou telas de execução inexistentes.

## Frente1 — semântica compacta

Ownership: `app/rag/semantic.py`, `knowledge_agent.py` se necessário e novos testes próprios. Sem editar schemas globais, router/orchestrator/store/metrics.

- Análise e verificação continuam duas passagens sobre o ORIGINAL inteiro. O servidor enumera unidades e spans, conserva corpos/hashes/offsets/quotes e resolve as seleções. O modelo não devolve cópias desses campos.
- Saída de análise: `request_hash`, `units` como objeto de chaves de unidade; valores têm tipo/decisão/requisitos/qualificadores e `span_ids`. Construir `properties/required` dinâmicos para todas as unidades, `additionalProperties:false`, enum unitário do hash e enum de spans permitidos. Sem `const`; o SDK transporta JSONschema cru. Mesmo com constrained decoding, validar localmente todos esses invariantes.
- Verificação: mesmo hash, objeto `checks` com exatamente as chaves esperadas, verdict/issues e completude. JSON com chaves duplicadas deve ser rejeitado antes do parser. Ordem das propriedades não altera significado.
- Não fabricar subject/relation para reaproveitar DTO antigo. Construir resultado canônico a partir das decisões efetivas e dados imutáveis do servidor. Renderer continua extrativo e checa identidade do corpo ao renderizar.
- Erros tipados com `stage/code`, contagens e campos conhecidos: separar parse/schema/hash/keyset/state/span/source/budget/deadline/finish/provider. Nunca registrar pergunta, JSON rejeitado, quotes, headers, chave, input/context Pydantic ou corpo da exceção. Taxonomia explícita permite diagnosticar falhas sem preservar PII.
- A análise deve poder receber `delegation_scope` do servidor: unidades atribuídas a este especialista e unidades atribuídas a outros. Todas as unidades e condições ainda são revistas. Unidade só atribuída a outro especialista pode ser marcada `delegated`, sem receber crédito factual; isso não torna a cobertura GLOBAL completa. Um qualificador pertinente não pode desaparecer por estar em outra unidade.

## Frente2 — contrato de unidades, ledger e referência

Ownership: novo `app/security/request_units.py` (enumeração pura sem dependência SDK); `request_constraints.py`, `operational_intent.py`, router, store, suporte, orchestrator e testes próprios. Coordenar imports com frente1. Não alterar schemas globais nem metrics; integrador adiciona os campos abaixo.

- Centralizar a enumeração literal atual e manter alias compatível `app.rag.semantic.request_units`. Unidades são dados do servidor, nunca inventadas pelo modelo.
- `PlanStep.request_unit_ids: list[str]` default[]; router nativo recebe unidades e associa IDs às delegações. IDs fora do pedido são rejeitados. Plano legado com um passo pode ter associação conservadora pelo servidor; múltiplas intenções reconhecidas não podem desaparecer só por falta desse campo.
- Ledger por requisição conserva unidade, atribuições, evidência reportada e estado `fulfilled/unresolved/delegated`. Especialista recebe original e escopo separado; não substituir original pela subquery. Unidades pendentes no fim devem aparecer literalmente como não respondidas. `ChatResponse.request_coverage` expõe esse ledger, sem confundir delegação/HTTP200/resultado encontrado com cumprimento.
- `SpecialistOutput.request_unit_ids`, `covered_unit_ids` e `unresolved_unit_ids` são opcionais compatíveis. Só atribuições conhecidas entram no ledger; discrepâncias não recebem crédito. Knowledge informa cobertura a partir de sua análise efetiva. Suporte só informa cobertura que seus resultados tipados e restrições reconhecidas sustentam; ambiguidade/condição não compreendida fica unresolved. A resposta global só recebe complete se todas as unidades pertinentes forem cumpridas; caso contrário partial/abstained conservadores. Manter a capacidade de mostrar fatos úteis sem prometer resolver o pedido inteiro.
- Resolver continuação diagnóstica a partir de entidade singular persistida e intenção diagnóstica reconhecida, mesmo sem `last_agent` e sem depender exclusivamente de same/mesmo. Não herdar entidade quando houver novo alvo/modelo/tópico explícito ou escopo documental. Ambiguidade exige esclarecimento; preservar alvo original, tenant e consentimento.
- Offline usa a mesma unidade/contrato para detectar mistura documental/operacional. Não é requisito de paridade semântica aberta, mas parte não reconhecida não pode ser omitida silenciosamente. Não ampliar listas com palavras dos novos probes como solução.

## Frente3 — integrador e demonstração

- Schemas globais: campos opcionais listados acima; `ChatResponse.request_coverage:list[dict]`. Métricas operacionais ganham agrupamento por modo/status, sem chamar sucessoHTTP de resolução semântica. Preservar nomes antigos compatíveis e estatísticas globais.
- Budget/retries continuam30s/geração sob60s totais e zero retries. Medir bytes/tokens e duração por fase. Não ampliar timeout indefinidamente. Modelo padrão continua3.1-flash-lite; piloto3.5-flash-lite de2casos conhecidos não basta para troca nem é novo holdout. Erros/indisponibilidade continuam contando no relatório.
- Walkthrough local: duração aproximadamente3–5min, arquitetura/raciocínio/spec+agentes/fluxo/segurança/limites; saídas e números vêm de recibos existentes. Narrador sintético explicitado, dados simulados, nenhuma chave/header/token/tela privada. Não chamar roteiro de vídeo. Entregar arquivo reproduzível, inspecionar quadros/áudio/duração e vincular README. Preparação pode ocorrer fora do produto sem modificar candidato congelado.

## Aceite e congelamento

TDD: decisões compactas equivalentes às canônicas; hash/keyset/span extra/omitido/duplicado rejeitados; corpo alterado invalida renderer; original/negação/condição/unidade continuam protegidos; diagnóstico sem last_agent com contexto singular e troca explícita; cláusula documental+operacional registrada e não sumida; ledger não aceita IDs do modelo fora do pedido; modos/status agregados sem declarar completo indevidamente; erro por código sem PII.

Suíte completa, checker, harness sem alterar labels, Python3.11/Docker/CI do SHA exato. Pilotos de classes conhecidas são diagnósticos, não nota. Nova auditoria independente depois de freeze, amostra inédita por modo e adjudicação explícita. Main só se95+ sustentado e CI verde; resultados podem continuar abaixo da meta. Nenhum cap artificial ou regra de pontuação foi acrescentado.
