# Correção V4 — evidências e limites (23/10/2026)

Baseline: `672c87e`, auditoria local independente nominal 66,75/100. Esta correção foi planejada em `SPEC_V3.md`, com frentes separadas de segurança/ações, conhecimento/factualidade/avaliação e estado/HITL/supervisor. Um agente separado congelou 76 casos antes de compartilhar feedback; outro revisor independente avalia o candidato e acrescenta probes após o freeze. Labels de agentes não são validação humana nem nota oficial Getnet. A rubrica normativa completa/âncoras não foi disponibilizada.

## Mudanças verificáveis

- Router, Support e Knowledge/Web usam function calling nativo com credencial configurada. Schemas, identidade, escopos, orçamento e execução ficam no servidor. Falha de provedor não ativa planejamento lexical silencioso.
- Respostas factuais usam extratos ou campos tipados, sem aceitar prosa livre do modelo como confirmação de pagamento/ação. Campos ausentes não viram máximas, mínimas ou precipitação zero. Datas e referência diária/histórica ficam visíveis.
- POS: proposta imutável com TTL/hash → confirmação por JWT do titular → recibo SQLite simulado. Chave global não escolhe identidade para confirmar. Scopes são exigidos nas tools e nos endpoints administrativos. Nenhum comando físico é enviado.
- Idempotência compara payload; alvo explícito inexistente não recebe outro terminal. Workers/fila limitados; escrita em voo após timeout é desconhecida/não repetível automaticamente, com operation_id consultável e estado persistido.
- PII natural é redigida antes de persistir/enviar contexto. Regressão obtida ao vivo corrigiu filtro de conta bancária que corrompia IDs opacos de proposta.
- Memória guarda resumo/fatos/entidades; mudança de assunto difere de referência. HITL valida transições e operador em transação; outro ticket pendente mantém pausa. Observações de ferramentas alimentam próximos especialistas e replanejamento limitado.
- Corpus revisto contra fontes oficiais, mantendo limitações de acesso explícitas. Pipeline continua BM25/expansão + hashing/âncoras + RRF; não foi instalado encoder neural. Answerability usa restrições/ontologia local e ainda requer avaliação independente.
- O snapshot financeiro é histórico e sintético; não confirma um depósito atual. Data explícita sem registro retorna ausência. Valores históricos não devem ser apresentados como liquidação no dia da consulta.
- Build/CI instalam lock transitivo. Contêiner roda sem root, com filesystem somente leitura; estado em `/app/state` separado do corpus da imagem. API síncrona é executada fora do event loop. UI aceita JWT em memória e mostra confirmação explícita.

## Verificação anterior ao freeze

| Verificação | Resultado observado |
|---|---|
| `.venv/bin/python -m pytest -v`, Gemini desligado |332 passaram, um aviso de depreciação Starlette/httpx,5,30 s|
| Mesma suíte na imagem Python 3.11, rede desligada |332 passaram, mesmo aviso,5,06 s|
| Harness local 13 regressões |routing 1,0; recall 1,0; precision 1,0; seleção segura 1,0; consistência limitada 1,0; pass rate 1,0|
| Latência harness offline |p50≈13,61 ms; p95≈229,49 ms; não extrapolar para Gemini|
| Docker final `getnet-v4-candidate:local` |build e 6 checks: saúde/offline, non-root/read-only, strict auth, chat autenticado, tenant/role e persistência após restart|
| Checker AST, `git diff --check` |sem violações observadas|
| Credencial/local |`.env` ignorado/mode 600; nenhuma ocorrência da credencial nos arquivos candidatos; sem DB/.DS_Store no status|

O checker e suíte verificam regressões concretas; não provam ausência universal de falhas. O novo juiz automático valida consistência limitada de números/polaridade/extratos, não entailment universal. Os 13 casos antigos tiveram labels documentais corrigidos com fonte/método registrados; não são holdout independente. O conjunto reservado tem 76 casos, manifesto SHA256 `9f6e30f09a809f4d8dfdaf01203bb3410788d72db6969f732c7270a9485fc56b`, fora do repo de entrega; os implementadores não tiveram acesso às perguntas. Feedback preliminar reduz a independência daqueles casos para uma rodada seguinte, por isso o revisor acrescenta inéditos pós-freeze.

## Integração Gemini real: parcial, sem aprovação fictícia

A conectividade foi confirmada usando credencial local, sem expor seu valor. O modelo 2.5 retornou 404 para novos usuários; 3.5 conectou, mas depois retornou 429 (cota gratuita de 20 chamadas/dia) e 503. O modelo configurado passou a `gemini-3.1-flash-lite`, que executou chamadas reais de delegação, tools e continuação.

Uma rodada intermediária completou quatro dos seis checks, incluindo conhecimento, consulta de perfil e handoff de hardware; identificou a corrupção de ID e uma checagem de CVV que confundia dígitos em tokens com dado rotulado. Ambos receberam correção/testes. A rodada posterior não ficou toda verde: dois de seis checks passaram; o provedor retornou 504 em suporte/replanejamento. PII ficou ausente do contexto observado, mas isso não transforma um cenário operacional indisponível em sucesso. Não houve ativação de cobrança, troca de plano ou alegação de disponibilidade garantida.

Recibos completos locais em `../getnet_v3_evaluation/`: `docker-final-smoke.json`, `gemini-flash-lite-smoke.json`, `gemini-final-smoke.json` e diagnóstico de 429/503. Contêm somente fixtures sintéticas/traces redigidos. Os resultados intermediários têm mudanças posteriores e não substituem a auditoria do SHA final.

## Gate de publicação

Nenhuma nota 95+ é atribuída por este registro. Após o freeze, a auditoria independente deve avaliar 11 critérios/4 gates, suíte, holdout e probes novos. O Actions será verificado no branch candidato; verde não autoriza por si só atualizar main. Publicar no main somente se a avaliação sustentar 95+; caso contrário preservar o branch e documentar nota/pendências sem inflar.
