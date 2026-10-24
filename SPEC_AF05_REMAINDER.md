# AF05 — correção delimitada de A4/A5/A8

Baseline af050745e6fac8613de7de928ff9aa4692449bc2. Amostra Antigravity AF05 consumida, leitura apenas; sem rede/reexecução/score novo/main.

A4: após FC selecionar diagnose_terminal_issue, completar fanout read-only dos pares explícitos atribuídos, uma vez por alvo, respeitando tenant/constraints e MAX_TOOL_TURNS. Não trocar família de tool nem executar writes por compensação. Prova por campo deve cumprir ledger; formato estrito ID/valor é constraint, não tarefa adicional. Sem headers/conselhos extras quando pedido só células.
A5: contexto afirmativo de conta/negócio alheio e falta explícita de permissão devem negar antes de qualquer consulta operacional, conservando bloqueio tenant no handler. Negação de propriedade alheia não deve ser confundida com falta de permissão. Casos mistos próprios/estrangeiros permanecem por alvo.
A8: registro explícito de proposta pode referenciar ação/alvo inequívocos na frase anterior; criar somente PENDING_CONFIRMATION. Não atribuir execução a uma ferramenta que só prepara proposta. Informação/hipótese/veto de proposta continuam negados; confirmação posterior JWT/hash/idempotência intactos.

Aceitação offline HTTP usando FC sintético que omite segundo alvo, registry real+spy, envelope/ledger/persistência e negativos. Não há nova nota após testes locais; 92 é estimativa reportada pelo Antigravity sem tabela V4.
