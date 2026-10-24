# BCC2 — reparo delimitado A5 e A8

Data:2026-10-23. Baseline bcc235369fd356b7db92d48285f338d49ea21ae0. Branch codex/getnet-v11-conditional-pix, worktree /private/tmp/getnet-v11-worktree. Spec antes do código: SPEC_BCC2_A5_A8.md. Nenhum novo agente, Gemini/rede, push ou modificação de main.

## Origem e escopo

Arthur forneceu avaliação nominal independente93,76/100,6/8casos aprovados, A5/A8pendentes. Relatório no workspace pai: pipeline/GETNET_V11_BCC2353_INDEPENDENT_EVALUATION_2026-10-23.md. Bundle nativo bcc2353 somente lido, oito itens manifesto OK; raw SHA25664a46ce8ed8fe11572dfe063645b6d3b2ab6e9f403ee5bc4877d91d3cc9d3230. A amostra está consumida/conhecida, não foi reexecutada nem editada.

A5: parser anterior reconhecia propriedade 'belongs/owned by' e ausência de 'permission/access', mas não relação afirmativa 'logistics partner maintains terminal', passiva 'maintained by partner business' ou 'no administrative rights over the partner profile'. Estendido o contrato de propriedade vinculada ao alvo e a negativa de permissão. Preflight existente agora recusa antes de provider Support/registry e tools_called fica vazio. Handler tenant-scoped preservado. Testes negativos mantêm menção incidental a parceiro e propriedade negada fora do bloqueio.

A8: pedido explícito 'Gere somente a proposta formal' não pertencia ao vocabulário de preparação aceito. 'Gere/gerar/generate' agora aceitos tanto na preparação quanto no reconhecimento de veto. Mantida referência somente ao pedido imediatamente anterior, com ação/titular/alvo únicos; hipótese/citação/ambiguidade/veto continuam negados. Não alterado endpoint de confirmação, TTL, hash, idempotência ou execução.

Contrato público solicitado por Arthur: ChatResponse serializa action_proposals com status=pending_confirmation e action=remote_restart para remote_reboot. Estado interno SQLite continua PENDING_CONFIRMATION, action_type original e payload_hash inalterados; campos anteriores de confirmação permanecem no JSON. Testes HTTP antigos foram atualizados SOMENTE quanto à caixa pública de status; asserts internos e confirmação autenticada/replay permanecem.

## RED/GREEN e checks

Nove novos casos de teste, com HTTP/SDK FC sintético e registry real/spies, incluem a mensagem exata A5 e variantes; mensagem exata A8 com fixture sintética do alvo; SQL confirma proposta pendente e zero terminal_actions. Negativos de parceiro/veto/hipótese preservados.

RED inicial verification/bcc2/red.txt:4falhas/5passes e3erros de teardown decorrentes do pytest.fail no spy que capturou dispatch proibido, não defeitos adicionais de produção. Após implementação, única falha no novo teste A8 era erro de harness SQL (terminal_id fica em payload_json, não coluna); corrigido sem alterar expectation. Primeira integral teve as3expectativas antigas de statusuppercase HTTP e esse erro SQL; recibos preservados em green_initial/full_initial.

Foco final verification/bcc2/focused.txt: **52passed,1warning,0,96s**.
Integral final verification/bcc2/full_verified.txt: **1565passed,1warning,13,29s**. Comando: Python do .venv do projeto executando verification/v11_e2e/offline_verify.py -q; wrapper roda pytest-q com sockets IP bloqueados antes de imports. O aviso é upstream Starlette/httpx.

Checks pedidos executados literalmente: python scripts/check_no_print.py app tests && git diff --check — exit0, saída vazia. Nenhuma alteração no código das frentes A1/A2/A3/A4/A6/A7; a suíte anterior permanece incluída, sem certificação nativa nova.

## Decisão

Reparo local concluído e apto a nova avaliação. Não é nova nota V4;93,76continua referente ao baseline. Não há promessa de95. Sem CI novo, rede, push, main ou publicação. Futuro teste nativo deve usar o novo SHA e amostra nova, com autorização própria.
