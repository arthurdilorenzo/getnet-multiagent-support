# SPEC V11 — Addendum M08: marcadores sem prefixo e probe do Router

## Baseline

HEAD revisado: `de1f750f25b8c766eecb67e9eb56c22183d3ff5d`.

## Contratos

1. Postfixes condicionais reconhecidos (`If`, `Provided that`, `Since/As long as`, `Se`, `Desde que`, e os já suportados com conectivo) vinculam ao conteúdo imediatamente anterior quando o texto pede uma divulgação dependente. A ligação lexical não amplia a gramática autorizadora: condição não parseável ou não provada permanece bloqueada e não busca conteúdo.
2. Perguntas diagnósticas diretas continuam permitidas; `if/se` interrogativo não se converte em condição de divulgação.
3. Nenhum retriever local nem ferramenta Web recebe conteúdo dependente antes do pré-requisito. Em particular, o probe de pertinência do Router (`retrieve_candidates`) deve pular unidades/documento condicionais até validação tipada. Planejamento sem retrieval pode continuar, desde que as unidades bloqueadas não sejam executadas.
4. A condição autorizadora só abre a consulta documental depois de `diagnose_terminal_issue` concluído, sucesso tipado, terminal exato e firmware/status esperados.

## Aceite

- RED/GREEN com Router real: `Tell me the eligible benefits. Provided that firmware ...`, equivalente `If ...`, e `Desde que ...` não podem chamar retriever antes do diagnóstico nem Knowledge/Web em condição desconhecida/falsa.
- Spy no `retriever.retrieve_candidates` comprova zero consultas com texto dependente; usar plano que realmente chama `_prefer_local_documentary_evidence`, sem substituir `router.plan` inteiro.
- Positivo autorizado mostra ordem: diagnóstico real primeiro, retrieval depois com conteúdo dependente canonical.
- Diagnósticos `Check if my terminal ... is Online` e `Verifique se o terminal ... está Online` continuam executando Support.
- Regressões M08/P05/P06, testes do Router, suíte completa, avaliador offline, `check_no_print.py` e `git diff --check` passam. Nova revisão independente antes de novo freeze nativo.
