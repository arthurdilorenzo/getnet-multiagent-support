# V11 — verificação offline dos contratos ponta a ponta

Data: 2026-10-23. Baseline f519090b24fd44341e3ba5c0733c5c059255714b; branch local codex/getnet-v11-conditional-pix. Spec anterior ao código: SPEC_V11_END_TO_END_REPAIR.md. Histórico e decisões: V11_END_TO_END_REPAIR_PROGRESS.md. Duas frentes e revisor independente, sem rede/Gemini/push/main.

## Resultado entregue

- Unidades canônicas vinculam fontes/formato/contexto à tarefa sem absorver pedido factual ou leitura privada independente. Router e Knowledge recompõem escopo real de consulta. Replan conserva assunto/produto/país e fatos brasileiros já sustentados; SIM conserva tarefa e entrega spans reais em testes de transporte sintético.
- Condição desconhecida permite leitura própria inequívoca, sem permitir conteúdo dependente. Predicado positivo exige alvo/titular/firmware/status tipados, rejeitando atributos adicionais ou incompletude. Aviso usa valores realmente observados.
- Resposta HTTP completa passa por projeção após agregação e antes de persistir. Texto/evidence/intent/trace/citações/fontes/ledger respeitam contrato por alvo/campo; observações internas ricas continuam disponíveis ao supervisor. Ausência de diagnóstico não libera prosa nem crédito de cobertura.
- Propriedade estrangeira explícita, inclusive plural, bloqueia antes do registry. Handler tenant-scoped permanece. Proposta explicitamente solicitada é persistida pendente; confirmação JWT/titular/hash/idempotência permanece separada da preparação. Falha conhecida não herda completed; escrita indeterminada conserva unknown.
- Falha Web de incidente atual comunica não confirmação específica, sem inventar pane ou consultar dados privados.

## Verificação e limites

Comando reproduzível, com sockets IP bloqueados antes de importar a aplicação:

```
/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python verification/v11_e2e/offline_verify.py -q
```

Recibo final verification/v11_e2e/full_release.txt: **1548 passed, 1 warning em 13,83s**, exit 0. O aviso é upstream Starlette/httpx. Runtime local Python 3.13; parsing AST com gramática 3.11 de app/scripts/tests passou (static.txt), mas isso não substitui execução Python 3.11/Docker/CI.

`offline_verify.py eval`: exit 0, evaluator_release.txt. Fixtures externas declaradamente sintéticas; routing/tool recall/precision/safe/groundedness limitada 1.0, overall_pass_rate 0.9231. Não é nota V4, qualidade Gemini nem disponibilidade real.

`check_no_print.py app`, AST e `git diff --check` passaram. Revisão independente em V11_INDEPENDENT_END_TO_END_REVIEW.md; três achados reproduzidos foram corrigidos e rechecados, inclusive resíduo de ledger. Não é certificação de linguagem natural aberta.

Regressões HTTP incluem FC de SDK sintético, registry real com spies, argumento efetivo, envelope serializado integral, pares de alvo/campo, titular estrangeiro, proposta/receipt SQLite, hash errado, confirmação alheia, replay e memória assistente igual à resposta minimizada. Testes semânticos de corpus usam spans reais e decisões de provider sintético; não executar Gemini foi requisito desta rodada.

Os recibos intermediários com falhas permanecem em verification/v11_e2e/: full_initial (213 falhas), full_second (44), full_final (1), full_verified/full_release (verdes). Não se ocultou falha nem alterou labels F519. Relatórios A/B e logs focados preservam RED/GREEN. Mudanças de expectativas legadas documentadas: observações internas não são payload público; leitura própria desconhecida é independente; mocks positivos agora provam titular/alvo; query canônica prevalece sobre subquery genérica. Nenhuma expectativa de consulta dependente, tenant ou execução foi afrouxada.

## Decisão operacional

Nenhuma nova nota atribuída. **88,18 nominal continua sendo a última nota nativa conhecida de f519**, não avaliação deste código. Sem CI exato ou Docker novo. Nenhuma promoção autorizada por testes offline.

F519 preservada: 10 itens do manifesto OK antes/depois; raw SHA256 b9c30b32f4eebde559de2b81aef66b8f795a46e0851a28e17fa3f7608953727d. Sem reexecução do runner, sem alteração de orçamento/modelo/retries.

V9 e6dd7dd (93,79 nominal) foi comparada somente em leitura após pergunta de Arthur; não houve rollback/troca de base. Amostras e notas diferentes não medem causalmente melhora/piora. A versão antiga já tinha falhas de multialvo/Algar/misto/país; superfície menor pode orientar simplificação futura, sem apagar ganhos comprovados.

Publicação continua condicionada a futura amostra nova expressamente autorizada, nota independente real >=95, gate semântico aprovado e CI verde no SHA exato. Main local permanece 672c87e; nenhuma chamada de push foi feita.
