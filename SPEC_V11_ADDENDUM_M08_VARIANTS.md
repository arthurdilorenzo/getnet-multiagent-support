# SPEC V11 — Addendum M08: conectivos e perguntas diagnósticas

## Baseline

HEAD revisado: `0745d6ccbc054d3be4a4f2f854b314607936a776`. Este addendum cobre os dois achados independentes reproduzidos contra o registry real.

## Contratos

1. Uma condição pós-fixada que restringe uma resposta anterior continua a controlar o span anterior quando introduzida por conectivo de contraste, incluindo `But only if`, `However, only if` e equivalentes em português. Se o gate não autorizar a condição por gramática finita e leitura tipada do alvo exato, nenhuma busca factual dependente é executada.
2. Uma pergunta direta de diagnóstico de estado do terminal permanece elegível ao `CustomerSupportAgent` em formulações comuns, incluindo possessivo inglês (`my terminal`) e português (`Verifique se o terminal ... está Online`). Isso não libera documentação dependente que também esteja no pedido.
3. A distinção é baseada no propósito da frase e no vínculo com divulgação consequente, não no token `if/se` isoladamente. Em caso de mistura ou ambiguidade, preservar diagnóstico permitido quando separável e bloquear apenas disclosure dependente; se inseparável, abster sem alegar consulta.

## Aceite

- Testes RED/GREEN no limite `ToolRegistry.execute` para os quatro exemplos de conectivo, falsos/desconhecidos e autorização positiva por diagnóstico tipado.
- Testes RED/GREEN para `Check if my terminal ... is Online` e `Verifique se o terminal ... está Online`, afirmando execução Support e ausência de Knowledge/Web.
- Regressões existentes de M08, P05, P06, `test_replan_scope_v8.py` e integridade de ações passam.
- Nenhuma chamada de API/rede; o próximo teste nativo só após revisão independente limpa e novo freeze.
