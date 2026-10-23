# Evidência de implementação V10 — 23/10/2026

Baseline e6dd7dd. SPEC_V10 precede alterações. Dois agentes implementaram frentes separadas e integrador revisou diffs/contratos. Nenhuma nota95+ atribuída nesta fase; main672c87e protegido. Evidências extensas fora do produto em ../getnet_v10_evaluation/.

## Causas e alterações

- O02V9: separador canônico não reconhecia `plus`; gramática de cobertura não reconhecia `Give me`. Fronteiras aditivas agora reconhecem propriedade/verbo delimitados em PT/EN (`plus`, `mais`, `além de`, `as well as`), mantendo substrings/offsets/IDs e os conectores no original. Canonicalização de leitura consome somente conector literal reconhecido; propósito/alvo/tenant/TXN/campo continuam validados. Não pretende gramática universal.
- D04V9: seleção nativa desviou manual Getnet para web genérica. Instruções e descrições das ferramentas agora distinguem manuais Getnet de fatos externos atuais e leituras privadas instaladas. O modelo continua escolhendo; não há override local por palavras-chave nem prova de aderência real por teste de prompt.
- B01/B02V9: perda de utilidade parcial. Mecanismo reproduzido: rejeição de escopo/condição podia apagar todas as unidades aprovadas. Agora a segunda passagem pode declarar `affected_unit_ids`, conferidos contra unidades canônicas. Ausência/null/ambiguidade, condição anexa reconhecida ou condição descartada como contexto conservam veto global; rejeição delimitada pode preservar fato independente. A declaração vem do mesmo modelo e não é prova semântica independente. Não se afirma que esse mecanismo explique todos os tracesV9, que não preservam os checks completos.
- Original/escopo/unidades chegam ao SDK Knowledge como conteúdo de dados canônicos PIImasked: inserir apenas no conversation_context não bastava porque initial_contents filtra chaves. TesteRED adicional reproduziu esse filtro antes da correção. Estado privado/país desconhecido permanece pendente e não ganha crédito factual.

## Verificação antes do freeze

- FrenteB: RED6falhas/9passes;33testes novos,322focados isolados socketsbloqueados/dotenvsuprimido passaram. Fluxo CustomerSupport real offline e SDKtransportfake consulta dois alvos, preserva campoausente e nega alvoextrangeiro/escopo/TXNcruzado. Erro inicial de asserção de harness(found=false/envelopesuccess) preservado e corrigido só no teste.
- FrenteA: RED5falhas/13passes inicial e REDs de roteamento/schema/contexto preservados;25testes novos. Handler documental real com retrieval/corpus/registry/renderer/ledger e transporteSDKfake preserva fatoBrasil e país não suportado pendente; condições dependentes continuam bloqueadas. Não é testeGemini ao vivo.
- Integrador: `pytest -v`:1001passed,1advertência upstream Starlette/httpx. AST no-print e gitdiff--check passaram. CLIevaluator exit0: routing/recall/precision/safe/groundednesslimitada1,0; overall0,9231, modo synthetic_external_api_fixtures identificado. Não é prova de disponibilidade nem qualidade semântica ampla.
- Primeira suíte simultânea do agenteB teve995pass/4falhasAemedição; recibo preservado, não omitido. Full estável do integrador passou após as correções. Nenhuma alteração de corpus/labels/gates/modelo/budgets/retries.

Próxima etapa: freeze limpo, Docker/CI noSHAexato e novo auditor independente criando casos/labels apósfreeze, antes da única rodada nativa. Main somente95+ sustentado ECIverde. Uma nota nominal não é notaoficialGetnet e quantidade de testes não determina pontuação.

## Complemento após revisão independente da seleção documental

Uma revisão de D04 reproduziu que a variante ampla em inglês podia cair abaixo do limiar de retrieval e que o teste anterior provava somente preferência textual no prompt. SPEC_V10 Addendum V10.1 registra o achado antes da correção: a revalidação acontece depois do function calling; somente unidades cujo top candidate é HTTPS no domínio Getnet, `verified_markdown_mirror`, `reviewed_against_official_page` e score no mínimo do limiar existente podem ser reatribuídas de busca web a Knowledge. Expansão lexical bilingue SIM→chip recupera o manual Algar para “Explain the SIM installation steps for my Getnet device” (score0,2793; limiar0,26); a consulta de clima mantém Web (score top0,0428). O especialista continua sendo responsável pela autorização factual. RED/GREEN/metadata não verificada cobertos em `tests/test_documentary_partial_v10.py`; fluxo de Gemini real depois desta mudança não foi executado.

SPEC_V10 Addendum V10.2 acrescenta integração ausente antes da implementação de teste: `POST /api/chat` com prazo Pix coberto e firmware nulo num segundo terminal mantém resposta pública, pendência explícita, status partial e ledger fulfilled/unresolved, sem tomar valor de outro terminal. Novo teste em `tests/test_v10_mixed_pix_firmware_handler.py` passou.

Reverificação local do candidato ainda sem freeze: `GEMINI_API_KEY='' .venv/bin/python -m pytest -v`:1007passed/1 advertência upstream. Evaluator CLI com chave explicitamente vazia: exit0, modo `synthetic_external_api_fixtures`, overall0,9231, métricas limitadas1,0. Smokes Docker seis checks passaram numa imagem `source_dirty=true`, identidade do HEAD pre-freeze; aguarda reexecução no commit candidato limpo. `git diff --check` passou.

Registro de incidente: uma chamada inicial do evaluator sem suprimir `.env` selecionou Gemini ao vivo, executou13 casos, encontrou `ConnectError` sem resposta e terminou exit1; seguido de `CircuitBreakerOpenError` local. Está em `getnet_v10_evaluation/root-recheck-incident.md`, excluído da avaliação e sem uso como holdout. Não registramos segredo. O erro não confirma se houve request remoto/cobrança; nenhuma chamada adicional está planejada nesta rodada.
