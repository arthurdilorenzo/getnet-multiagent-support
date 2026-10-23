# Evidências V6 — em desenvolvimento

2026-10-23. Baseline `d7e2125`, auditoria independente89,27/100. Este arquivo não atribui nova nota nem autoriza promoção. Plano anterior à implementação: [SPEC_V6](SPEC_V6.md). Main continua672c87e.

## Contratos e observabilidade

- Testes novos primeiro reproduziram ausência de associação de unidades/cobertura e agrupamento de métricas. Schemas opcionais preservam compatibilidade; cada lista tem default_factory independente.
- MetricsCollector agrupa modo/status declarados sob vocabulário limitado. Texto arbitrário vira unclassified; sucesso de execução não é resolução factual. Campos globais anteriores preservados.
- Geração distingue falhas locais tipadas de orçamento/candidato/finish/content via ProviderTurnError.code. Deadline continua TimeoutError; authorizer mapeia etapas separadamente. Não persistir payloads rejeitados ou texto de exceção.
- Estimativa preventiva passa a incluir instrução, tools e schema além de contents; uso real reportado pelo SDK continua preferencial. Telemetria de bytes/estimativa não é contagem faturada. Deadline60s, limite30s por geração, SDK/app sem retry mantidos.
-27 testes focados passaram após implementação: contratos V6/telemetria, provider_budget_v3, operational_contract_v5. Suíte final, integração, Docker, CI e nova auditoria ainda pendentes.

## Frentes delegadas

Protocolo compacto: decisões por IDs canônicos, sem cópia autoritativa de corpos/hashes/offsets; dupla análise/verificação preservada. Cobertura global: unidade/atribuição/evidência e resolução de referência diagnóstica persistida. Walkthrough: artefato local narrado sinteticamente, com evidências reais e limites explícitos.

## Limites vigentes

- V5 teve9/18 falhas503/504 e4 rejeições de protocolo. Protocolo menor é hipótese de melhoria; não foi demonstrado como causa dessas falhas. Preservar resultados e não repetir chamadas para substituir falhas da amostra anterior.
- Corpus é mirror curado por agentes de13 chunks, sem revisão humana ou encoder neural local. RAG lexical/projeção conceitual/RRF e interpretação Gemini são camadas distintas.
- Harness13, smoke6 e teste de componente não substituem avaliação nativa ponta a ponta inédita. Sem95 por checklist, quantidade de testes ou CI verde.

## Walkthrough materializado

`docs/walkthrough/Getnet_Walkthrough_Tecnico.mp4`:280,215s,3.699.599bytes,1280×720,H264/AAC. SHA256207f9cd794bc36db0b5a908e40e2a381afeafae6e2013594219b7ea8c48b105f. Narração local Luciana explicitamente sintética; baseline d7e2125, V6 somente contrato proposto.11 quadros/folha de contato, áudio(mean−15,1dB/max−2,4dB), decodificação integral e manifesto conferidos pelo agente produtor. Bundle inclui snapshots/recibos sanitizados/scripts sem depender do workspace original.198 testes de contrato do baseline exportado passaram com sockets bloqueados; não são execução do novo candidato, nem avaliação Gemini. Ver README/verification/sources do bundle para método e limites.

## Integração e resultados anteriores ao congelamento

-555 testes passaram,1 warning upstream Starlette/httpx,5,49s; checker AST e diffcheck passaram. Novos contratos cobrem diagnóstico referencial PT/EN, trocas explícitas de alvo/tópico, ledger compartilhado, PII, isenção nativa verificável, proposta pendente, metaetapa HITL, protocolo compacto e juiz de footer literal. Fixtures V5 foram adaptadas ao transporte compacto sem aceitar API antiga; assertivas de ação distinguem proposta de execução.
- O harness mantido sem alterar perguntas/labels/thresholds retorna routing/recall/precision/seleção segura1,0 e consistência limitada0,9231. **overall_pass_rate0,8462; CLI exit1.** Comparação Clássica/Smart preserva falso negativo extrativo conhecido; repasse histórico é parcial sustentado, sem cumprimento do pedido atual. Não remover o gate para obter CI verde. A suíte só exige consistência>=0,90; não confundir seus555pass com aprovação do CLI completo.
- Encontrada contaminação entre execuções: IDs fixos do harness acumulavam73turnos e reaproveitavam decline_code05 numa pergunta de conectividade. Teste RED→GREEN exige namespace de sessões único por execução; relatório conserva evaluation_run_id. Registros anteriores permanecem preservados.
- Piloto compacto real de2casos conhecidos: caso comercial análiseSTOP e verificação503; composto análise400. Diagnóstico separado de1call repetiu400 com mensagem genérica INVALID_ARGUMENT, sem diagnóstico específico de complexidade. Nenhum corpo de erro foi persistido.
- Retirado somente maxItems remoto redundante; allowlist+unicidade local impõem mesma cardinalidade. Novo experimento: caso comercial complete20,874s, composto análise50426,980s. Não prova correção da causa400 nem disponibilidade; não substitui recibos anteriores. Esses casos conhecidos são diagnóstico, não holdout ou nota.
- Medida sintética de serialização antes dessa última retirada:1unidade616→276bytes e2unidades1148→457bytes de saída; schemas1154/2000bytes nos testes sintéticos. Não são tokens faturados ou ganho de latência medido.

Recibos detalhados fora do produto em `../getnet_v6_evaluation/`: pytest-final.txt, evaluator-final.txt, development-harness-final-integration.json, compact-pilot-results.json, schema-diagnostic.json, compact-without-maxitems-pilot.json. Congelamento/auditoria/CI do SHA exato serão recibos separados. Candidato ainda não satisfaz promoção95+ nem CI verde.
