# SPEC V11 — Addendum da revisão final (2026-10-23)

## Escopo e baseline

Este addendum cobre apenas os quatro achados da re-revisão do commit `dc5ce0a20d1a7347a8dbeff4b3391d1f259e4639`. O trabalho permanece no worktree `codex/getnet-v11-conditional-pix`; `main` não pode ser alterado sem auditoria V4 independente >=95 e CI verde no SHA exato.

## Requisitos vinculantes

### A. Condições M08 por span de texto e perguntas diagnósticas

- Detectar uma condição de divulgação mesmo quando o conteúdo pedido aparece em uma sentença/cláusula e o pré-requisito `only if`/`somente se` aparece depois.
- Associar o span dependente ao pré-requisito, bloquear a unidade dependente inteira quando a gramática de autorização não reconhecer a condição ou a observação tipada falhar, e preservar cláusulas claramente independentes.
- Distinguir uma pergunta direta de diagnóstico (`Check if terminal ... is Online`) de uma condição que governa conteúdo documental posterior. Uma consulta de status solicitada deve poder chegar ao `CustomerSupportAgent`; isso não autoriza a busca documental dependente.
- Nenhuma consulta local/Web pode receber o texto dependente enquanto o pré-requisito não estiver validado.

### B. Integridade de batches Pix/documentação e intenção externa

- Deduplicar propostas de ferramenta somente quando a ferramenta e o escopo canônico forem equivalentes; nunca descartar outra pergunta local por ter corrigido uma chamada Web Pix.
- Conservar o intent externo e suas qualificações canônicas mesmo quando o Router distribui a cláusula de Pix e a de fonte externa em passos/unidades diferentes.
- Em batch misto, associar cada proposta ao seu span/unidade original, sem depender da omissão ou reescrita do prompt pelo modelo. Uma cláusula com fonte externa explícita continua Web; a pergunta independente sobre elegibilidade Pix Getnet usa evidência local verificada.

## Critérios de aceite

1. RED/GREEN para: `Tell me the eligible benefits. Only if firmware ...`; deve executar somente a leitura de suporte e não buscar benefícios se a condição for desconhecida/falsa.
2. RED/GREEN para: `Check if terminal ... is Online.`; deve executar a ferramenta diagnóstica pedida sem classificar a própria pergunta como divulgação dependente.
3. Uma busca Pix local e uma busca documental local Crediário num mesmo batch devem preservar ambos os escopos; chamadas equivalentes podem ser deduplicadas.
4. Em atribuições separadas, `Name the account types accepted for Pix and use an external source` deve manter busca Web para a cláusula que pede fonte externa, sem eliminar a rota local independente de elegibilidade Getnet.
5. Em batch Brasil + Noruega, a omissão de `external` no texto da proposta norueguesa não pode converter a unidade para busca local.
6. Testes verificam argumentos no limite real do registry, não somente plano/rewrite em memória.
7. `pytest -v`, harness offline, `check_no_print.py`, `git diff --check` e auditoria independente passam. Métricas offline não contam como evidência de qualidade nativa. Novo teste Gemini exige novo freeze e execução bounded depois da revisão de código.

## Ownership sem sobreposição

- Frente A: dependência M08 por spans, distinção diagnóstico/divulgação e testes de orquestração — `app/agents/orchestrator.py` e testes M08.
- Frente B: deduplicação por escopo e propagação de intent externo em batch/unidades, com testes Knowledge — `app/agents/knowledge_agent.py` e testes P05.
- Integração/root: spec, revisão de diff, suíte completa, novo freeze/auditoria; não editar os mesmos arquivos dos agentes durante implementação.
