# V5 — interpretação semântica nativa e contratos compartilhados

Data: 2026-10-23. Baseline congelado: `a79c52c`, **85,15/100 nominais**, main preservado em `672c87e`. A [auditoria V4](../GETNET_V4_AUDIT_V4_2026-10-23.md) reprova cobertura/recall e coerência router–store–especialista. Esta spec precede qualquer implementação V5. Decisão autorizada: continuar correções com agentes, sem publicar main antes de95+ sustentado e CI verde.

## Decisão e limites

Adotar a opção A+C do [parecer arquitetural](../GETNET_V5_ARCHITECTURE_OPTIONS_2026-10-23.md): Gemini interpreta/verifica o pedido completo no modo configurado; o renderer continua extrativo; fallback sem chave é uma capacidade delimitada, medida separadamente. Não instalar modelo local sem benchmark próprio. Não resolver a falha adicionando palavras dos48 casos ao parser. Os48 e76 casos anteriores agora são regressão conhecida. Auditor final novo precisa de amostra inédita após congelamento.

Schemas rígidos, IDs e offsets comprovam integridade estrutural, não semântica. Duas passagens do mesmo modelo não são auditoria independente. A nova arquitetura deve ser testada por mutações e com chamadas reais; não anunciar95 por implementação do plano. Falha do provedor com chave não aciona heurística silenciosa.

## Contratos

### Recuperação separada da autorização — agente semântico

- `HybridKnowledgeRetriever.retrieve_candidates(query, top_k=...) -> dict`: retorna candidatos por BM25/hash/RRF com corpo, chunk_id, hash, título/proveniência/URL e scores. NÃO filtra por `parse_requirements` e NÃO afirma suficiência. Escala atual13chunks permite candidatos amplos, com teto explícito de contexto.
- Manter `retrieve_hybrid` como modo local legado/delimitado. A tool documental deve disponibilizar `candidate_chunks` além do resultado local, sem alterar nome `search_getnet_knowledge`, identidade/registro ou promover candidato a evidência aprovada. Expor metadados públicos não autoriza renderer.
- Novo módulo `app/rag/semantic.py` com Pydantic estrito e `SemanticEvidenceAuthorizer`. Usa `generate_turn`/SDK existentes, sem conteúdo factual livre publicável.
- Entrada: mensagem original redigida, hash, contexto de referência autorizado e candidatos do servidor. Saída de análise: unidades textuais literais com IDs/offsets, tipo `requirement/context/courtesy/unresolved`, requisitos/qualificadores e decisões por unidade `supported/contradicted/unknown/ambiguous`, com seleções de IDs/spans/quotes/hash pertencentes aos candidatos. Toda parte substantiva do pedido deve ser representada; gaps, IDs inventados, extratos alterados ou seleção fora do corpo rejeitam o resultado.
- Uma segunda passagem compara a pergunta INTEIRA com a análise e os trechos. Deve conferir todos os IDs de unidades e apontar cláusula omitida, condição classificada como contexto, sujeito/unidade/escopo trocado. Protocolo inválido/falha ou reprovação não autoriza resposta completa. Preservar ressalvas completas nos parágrafos selecionados.
- Máximo duas gerações semânticas adicionais, sem retries de geração; sob deadline/tokens globais existentes. Remover seleção opcional redundante de um único chunk nesse caminho. Nunca usar o gate lexical antigo para rejeitar antes da interpretação nativa.
- Knowledge usa o novo caminho SOMENTE com cliente+chave. Sem chave permanece modo delimitado. Com chave e falha retorna motivo controlado; não chama fallback lexical para fabricar aprovação. Resposta parcial pode mostrar somente extratos verificados, acompanhada das unidades literais não respondidas e `abstained=true`; não declarar cobertura global. Se não houver suporte, abstém sem extratos irrelevantes.

### Sinalização de resultado — integrador

- `SpecialistOutput` e `ChatResponse` ganham campos opcionais compatíveis: `semantic_mode` (`native_semantic`, `limited_offline`, ou vazio) e `answer_status` (`complete`, `partial`, `abstained`, ou vazio).
- O orquestrador conserva esses campos para um especialista; em cadeia agrega conservadoramente: qualquer etapa pertinente incompleta impede status global `complete`. Eles são metadados explicativos, não prova factual.
- Modo offline não alega paridade semântica com Gemini. Um requisito/cláusula reconhecido como desconhecido é explicitado; a parte não compreendida não pode desaparecer com cobertura total. Testar múltiplas cláusulas e elegibilidade/condições ausentes usando fontes sintéticas novas.
- Traces registram modo, fase, cobertura, motivo e duração por geração, sem chaves/PII. Métricas distinguem utilidade em casos cobertos, abstenção segura, falsa suficiência e indisponibilidade por modo.

### Contrato operacional compartilhado — agente operacional

- Reutilizar `TERMINAL_PATTERN`, `TRANSACTION_PATTERN` e `extract_constraints` no store/router/especialista. Não manter regex antiga paralela para persistir/relembrar POS compostos. Referência inequívoca persistida deve alcançar o suporte mesmo após compactação e `last_agent` neutro.
- Centralizar a classificação operacional offline reconhecida em função compartilhada, com intenções liquidação/perfil/diagnóstico/ação e marcas de escopo privado/documental. Datas/IDs explícitos e slots existentes informam a decisão; alias de payout não pode cair silenciosamente em perfil. Negar certeza quando a intenção é ambígua. Não misturar uma pergunta comercial geral com liquidação privada apenas por termo relacionado.
- Ausência de transação deve renderizar seu `requested_transaction_id` e `TRANSACTION_NOT_FOUND`; não chamar a transação de terminal não informado.
- Preservar tenant/scopes, consentimento, imutabilidade, idempotência, data original e confirmação separada. Nenhuma execução remota nova.

### SDK e latência — agente operacional, separado dos demais arquivos

O [diagnóstico](../GETNET_GEMINI_LATENCY_DIAG_2026-10-23.md) mostra que timeout20s também vira `X-Server-Timeout:20`;504 não isola falha externa. Default do modelo3.1-flash-lite já é minimal.

- Tornar thinking minimal explícito somente quando compatível com família/modelo; não usar `thinking_budget=0` como receita para Gemini3. Registrar duração/fase/finish_reason e thoughts tokens quando disponíveis.
- Timeout por geração vira configuração validada, default30s, sempre limitado pelo restante do deadline60s; zero retries de geração. É experimento de budget, sem prometer eliminar falhas. Não aumentar deadline total.
- Support pode encerrar após resultado tipado conclusivo quando não existe chamada dependente/necessidade pendente, evitando pedir prosa final dispensável ao modelo. Não encerrar cedo uma execução que realmente requer segunda tool. Testes devem demonstrar ambos os casos; se não houver critério seguro, preservar loop e registrar limitação.

## Ownership e execução

1. **Semântica/RAG/Knowledge:** `app/rag/*`, `knowledge_agent.py`, somente handler/schema da tool documental em `search_tools.py`, novos testes próprios. Não alterar FX, schemas globais, function_calling, router/store/docs.
2. **Operacional/SDK:** `security/request_constraints.py` e novo módulo de intenção compartilhada se necessário; `customer_support_agent.py`, `router_agent.py`, `memory/store.py`, `agents/function_calling.py`, config e testes próprios. Não alterar Knowledge/RAG/search_tools/schemas globais.
3. **Integrador:** schemas globais, orchestrator, evals, scripts, documentação/recibos, testes de integração. Integrar novos testes nativos às fixtures SDK antigas sem remover expectativas de segurança/protocolo ou alterar labels para obter aprovação.

## Aceite e auditoria

- RED→GREEN com fontes novas sintéticas: duas cláusulas (uma desconhecida), condição classificada como contexto, sujeito invertido, unidade alterada, negação/exceção/escopo, quote/hash/ID/offset inventados, resposta truncada, verificação negativa e provider failure. Renderer não publica prosa livre. Recuperação não pode ter sido pré-filtrada pelo parser antigo.
- Testes de entidades compostas persistidas antes/depois compactação, referência sem last_agent, data por extenso através do chat, intenção privada versus comercial e TXN ausente. Provar zero handler em chamadas proibidas, além de `tools_called`.
- Suíte completa `source .venv/bin/activate && pytest -v`, checker AST, harness local com routing/recall1,0 e consistência>=0,90; não editar rótulos para passar. Integração real separada: medir disponibilidade e fidelidade semântica; respostas simuladas do SDK não comprovam generalização do Gemini.
- Congelar SHA limpo, imagem/CI do mesmo SHA; rodar amostra nova independente por modo e adjudicação humana/por agente explicitamente identificada. Casos antigos apenas regressão. Nenhuma nova nota oficial presumida, nenhum cap artificial, nenhum95 por checklist. Main somente se auditoria independente sustentar95+ e CI verde.
