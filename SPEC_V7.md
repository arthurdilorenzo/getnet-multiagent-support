# V7 — cumprimento verificável e integração de entrega

> For agentic workers: executar o plano com duas frentes independentes, TDD e revisão final independente; usuário já autorizou execução, sem nova aprovação intermediária.

2026-10-23. Baseline0d5edc2/90,34nominais, CI39986909336failure; main672c87e protegido. Fontes: auditoriaV6, SPEC_V6 e recibosgetnet_v6_evaluation. Janela de implementação16:51:33–17:51:33UTC (60min máximos), testes/auditoria depois. Dois agentes de implementação; revisorfresh somente apósfreeze.

**Goal:** corrigir falsa suficiência operacional, atribuição/protocolo/budget e contratos reais de entrega, sem afrouxar gates para95.
**Architecture:** original mascarado e unidades literais continuam autoridade. Cobertura exige relação/campo/alvo/data/escopo sustentados por observações tipadas, ou análise documental verificada; sucessoHTTP/tool não cumpre necessidade. Status parcial permanece explícito quando não comprovado.
**Tech Stack:** Python3.11+,Pydantic,FastAPI,GoogleGenAI,SQLite,pytest,Docker.
**Spec:** este documento, detalhando achados de ../GETNET_V4_AUDIT_V6_2026-10-23.md.

## Restrições globais

- Não alterar tenant/JWT/consentimento/idempotência para ganhar utilidade. Não usar cloudChatGPT/Work, nem expor chave/.env/PII.
-30s por geração,60s por pedido,32000tokens aproximados,zero retry permanecem; não trocar modelo sem evidência representativa. Não repetir amostra para substituir falhas.
- Manter gates e resultados prévios. Label incorreta só pode ser revisada independentemente com evidência, em registro separado; implementadores não alteram labels para passarCI.
- Não ampliar corpus, vídeo, frontend, dependências ou escopo. Não promover main abaixo95 ou comCIred. Deadline é limite, não promessa de95.

## Review focus

Campo ausente/None/vazio deve ficar pendente; observação relacionada mas de outra relação não cumpre; condição/negação/escopo desconhecido não some; unidade sem atribuição não pode ter fato renderizado e crédito contraditório; veto pré-call e tentativa extra devem ser distinguíveis de falha externa sem guardar conteúdo privado.

## Task1 — cumprimento operacional (agenteA)

Files: novo app/security/fulfillment.py se útil; customer_support_agent.py, operational_intent.py se necessário; tests/test_fulfillment_v7.py. Não editar router/orchestrator/semantic/function_calling/evals/labels.
Interfaces: preservar run(...,delegation_scope) e SpecialistOutput.request_unit_ids/covered_unit_ids/unresolved_unit_ids. Helper puro pode produzir decisão por unidade com reason enum/required fact paths/evidence-backed status, sem texto livre confiado do modelo.
- [ ] RED metamórfico: mesmo pedido/alvo com firmware presente/ausente/None/vazio; relação desconhecida recebendo telemetria; pergunta pública sobre modalidade/conta não ganhafulfilled porPOS; alvo/dataerrados; proposta pending não executa; condições não cumpridas.
- [ ] Implementar contrato delimitado de predicado+campos obrigatórios; dúvida fica unresolved. Campo solicitado precisa existir com valor útil e corresponder ao alvo/escopo. Não aceitar simplesintent+tool.success, nem validar por contagem de evidência. Reconhecimento delimitado declarado; não fingirentailmentuniversal.
- [ ] GREEN novos e contratos V4/V5/V6 pertinentes; nenhumcommit/push pelo agente.

## Task2 — atribuição, protocolo e budget (agenteB)

Files: semantic.py,knowledge_agent.py,router_agent.py,orchestrator.py,function_calling.py; tests/test_semantic_pipeline_v7.py. Não editar suporte/fulfillment/evals/labels/scripts.
Interfaces: units/delegation_scope e outputs atuais preservados; coordenar só leitura comA. Isenção continua prova nativa/hash/verifieraccept, sem crédito factual.
- [ ] RED: unidades pertinentes não atribuídas; facts verificados publicados de unidade fora do escopo; pendinglocal não contradiz fatos publicados; sharedqualifier/negação vetam; erroUNIT_STATE tipado por invariante seguro; vetos orçamento/deadline comfase/código antesnetwork; resposta já coberta não aciona replan sem próximo passo útil.
- [ ] Implementar associação de todas as unidades sem esconderdesconhecidas, reconciliação com prova canônica e parada baseadaemnecessidade restante. Corrigir invariantes de transporte sem aprovar outputs inválidos: tipos/decisões/IDs/corpos permanecemchecados. Diagnóstico só enums/booleans/contagens/unitIDscanônicos, nunca payload,subject/qualifiers,excbodyoupergunta.
- Refinamento prévio ao schema: piloto conhecido com diagnóstico novo retornou503/504, não interpretado como UNIT_STATE nem substituído. Adotar anyOf pequeno por estado(kind/decision/camposnãofactuais vazios), mantendo properties/required/enum/hash/validação local, para reduzir combinações inválidas permitidas pelo transporte anterior. Fonteprimária: https://ai.google.dev/gemini-api/docs/generate-content/structured-output?hl=en (anyOf documentado; complexidade podefalhar400). Não usar muitas alternativas/cópias gigantes nem prometerlatência/compatibilidade sempiloto real separado.
- Segundo experimento observou UNIT_STATE apesar de anyOf mínimo junto a properties comuns. Antes da próxima alteração: trocar para união de quatro objetos completos/autossuficientes (properties/tipos/required/additionalPropertiesfalse por alternativa), conforme o formato documentado, sem depender de interseção de keywords irmãs no decoder. Falhas anteriores preservadas; validar cada ramo localmente e medir bytes/erro400. Não declarar compatibilidade remota apenas por schema JSON válido.
- [ ] GREEN testes próprios+contratos originais; pilotos reais conhecidos só diagnóstico comrecibos novos preservados; semcommit/push.

## Task3 — integração, verificação e auditoria (root)

Files: scripts/docker_smoke.py, README/SPEC/evidence; testesnovos de contrato, semeditar filesdosagentes emparalelo.
- [ ] RED smoke: snapshotparcial precisa evidência/data/pendência exatas; HTTP200/toolname sozinho não basta. Atualizar expectativa obsoleta preservando todoschecksautenticação/tenant/restart.
- [ ] Separar defeito de produto e labeldefasada do harness. Se necessário, revisão independente depois dofreeze; não baixar gates, mascararfulfillment ou substituir label apenas paraficarverde.
- Refinamento anterior à correção do juiz: falso negativoV6 está na atribuição de fonte depois do mesmo predicado literal (“conforme o manual de uso disponível no índice público da fonte primária” vs “como mostra seu manual de uso”). Canonizar somente esses sufixos completos/conhecidos, ancorados no fim, tanto em claims quanto referências; manter o núcleo factual literal, números, papéis, condições e URLs. Cauda extra/nova afirmação não é removida. Testes positivos sintéticos e metamórficos adversariais devem provar que não é entailment livre nem relaxamento por palavras-chave. Labels/gates intactos; declarar escopo dessa normalização e exigir auditoria independente.
- [ ] Integrar dentro60min. pytest-v completo/checker/harness e DockerPython3.11; registrarfailedattempts e warnings.
- [ ] FreezeSHA limpo no branchcandidato; autor/revisor independente depoisfreeze, nova amostra pornecessidade e rodada nativa E2E única. Auditoria11pesos/gates sem caps ou95 porchecklist; CIexato comgh.
- [ ] Main somente se95+ sustentado E CIverde. Caso contrário, preservar main e registrar nota real/pendências.

Self-review: três frentes cobrem os achados;A/B têm ownership distintos e interfaces existentes; original/unknown/campoausente/condição/attribution/budget exercitados por metamórficos. Root limita implementação e preserva critérios, recibos e independência.
