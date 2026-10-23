# SPEC V11 — Addendum M08: anáfora, cláusulas independentes e reads contraditórios

## Baseline

HEAD revisado: `ceffaf92ce17202f6a323420ab3947aa02e24617`.

## Contratos vinculantes

1. `If so` / `Se sim` / anáforas equivalentes após diagnóstico explícito associam a divulgação subsequente ao resultado diagnóstico. A gramática positiva continua fechada: se a anáfora não puder ser autorizada pela gramática finita, o agente pode responder a unidade de diagnóstico, mas deve bloquear qualquer busca de benefícios/documentação dependente antes e depois do resultado.
2. Pergunta diagnóstica separada por ponto e vírgula de uma pergunta documental independente não transforma a segunda em dependente. Ex.: diagnóstico do terminal + categorias aceitas para Pix no Brasil deve permitir Support e a busca local independente Pix, sem atribuir a Pix a condição de firmware.
3. Em preflight de alvo único, qualquer erro/falha de leitura `diagnose_terminal_issue`, mesmo sem payload interpretável e ainda que outra chamada tenha retornado um valor positivo, torna a condição não verificada. Não aceitar sucesso parcial contraditório; fail closed.
4. Nenhuma consulta local/Web de unidade dependente pode ocorrer antes de qualquer preflight pertinente nem após condição falsa, ambígua ou não autorizada.

## Testes de aceite

- RED/GREEN com Router real, retrieval probe real, registry real e handlers sintéticos para inglês e português com anáfora; demonstrar Support permitido, zero retrieval/delegação dependente.
- RED/GREEN para diagnóstico + Pix independente unidos por ponto e vírgula; diagnosticar terminal e responder Pix local sem condicioná-lo ao status do terminal.
- Teste do avaliador puro e do orquestrador com uma leitura exata positiva seguida de erro `data=None` no mesmo terminal; condição deve ser falsa/desconhecida e nenhum factual dependent retrieval executa.
- Regressões M08/P05/P06, Task1 helper, Router, replan e ação segura; suíte completa e evaluator offline; diff check limpo.
- Nova revisão independente integral, novo freeze para qualquer rodada nativa; main só com nota >=95 e CI verde no SHA exato.
