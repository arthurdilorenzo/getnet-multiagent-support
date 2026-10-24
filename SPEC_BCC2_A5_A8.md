# BCC2 — A5/A8 somente

Baseline bcc235369fd356b7db92d48285f338d49ea21ae0, branch codex/getnet-v11-conditional-pix. Sem main/push/rede/Gemini/reexecução das amostras. A nota93,76 é reportada para baseline e não se transfere ao reparo.

A5 RED: declaração afirmativa de parceiro/supplier que mantém/administra terminal ou falta explícita de direitos administrativos sobre perfil/conta alheios deve produzir recusa antes de provider Support/registry, tools_called=[]. GREEN: alvo vinculado ao sujeito estrangeiro; menção incidental de parceiro e negação de propriedade não concedem bloqueio espúrio. Handler tenant-scoped permanece.
A8 RED: 'Gere somente a proposta formal' com ação/alvo únicos no pedido imediatamente anterior deve persistir UMA proposta pendente e zero ações executadas. 'Gere' e 'generate' têm mesma autoridade de preparar proposta, não de confirmar execução. Veto/hipótese/citação/ambiguidade negados. GREEN HTTP: action_proposals contém proposal_id, terminal_id exato, action=remote_restart, status=pending_confirmation; texto informa espera por confirmação autenticada.

Estado interno da fila permanece PENDING_CONFIRMATION/action_type=remote_reboot para preservar máquina de estados e hash. A serialização de ChatResponse expõe os nomes públicos solicitados, mantendo campos existentes (payload_hash/expiry/action_type) para confirmação e compatibilidade. Não reescrever estado persistido nem mapear confirmação física como automática.

Testes HTTP com SDK FC sintético e registry real/spy, casos exatos e negativos. Suíte completa offline pytest-q, check_no_print.py app tests, git diff --check. Não alterar A1/A2/A3/A4/A6/A7 nem labels congelados. Commit local após checks; sem atribuir nova nota.
