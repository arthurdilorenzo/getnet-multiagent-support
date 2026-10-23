# SPEC V11 — Addendum M08: anáforas equivalentes e integridade de preflight

## Baseline

HEAD revisado: `ac01841ad5ac1b1b0414d9cc5f1aa9c450543da8`.

## Requisitos

1. Consequências anafóricas usuais após um diagnóstico explicitamente pedido (`If it is`, `If that is the case`, `If so`, `If yes`, `Se estiver`, `Se for esse o caso`, `Se sim`) devem ser associadas à unidade dependente anterior. Como a gramática positiva segue finita, formas sem predicado terminal canônico ficam bloqueadas, mantendo o diagnóstico independente e sem probe local/Web antes ou depois.
2. Para autorizar a condição, todas as chamadas de diagnóstico do único alvo têm que retornar sucesso técnico e payload de negócio íntegro com identidade do terminal exata. `success=False`, exceção, payload `None`/vazio/malformado, `found=false`, alvo ausente ou alvo diferente invalidam o preflight, mesmo se uma chamada anterior foi positiva.
3. O dado completo exige `found is True`, `requested_terminal_id` igual ao alvo, campos tipados esperados de firmware e status. A comparação de firmware/status continua exata conforme o contrato finito existente; não inferir a partir de texto livre.
4. O produtor `diagnose_terminal_issue` deve incluir `requested_terminal_id` nos payloads de sucesso quando recebe um `terminal_id` explícito, ecoando o identificador canônico validado na frota autenticada. O consumidor não pode suprir identidade ausente por default; payloads de erro de domínio permanecem `found=false` e nunca autorizam consequências.

## Aceite

- Testes RED/GREEN para todas as anáforas acima, em PT/EN, com Router/retriever real e registry real; diagnóstico Support pode executar, dependente local/Web não.
- Testes puros, de orquestrador e da ferramenta para sucesso exato positivo (incluindo eco do ID pelo produtor), seguido por cada caso: `success=False/data=None`, `success=True/data=None`, `{}`, `found=false`, terminal ausente, terminal diferente, payload com tipos errados. Nenhum caso de falha/ambiguidade pode liberar Knowledge/Web.
- Regressões anteriores M08, semicolon Pix independente, P05/P06, full suite, evaluator offline e checks estáticos passam.
- Auditoria independente integral depois de implementação; novo freeze/camada nativa só após código aprovado; promoção main exige V4 >=95 e CI verde no SHA exato.
