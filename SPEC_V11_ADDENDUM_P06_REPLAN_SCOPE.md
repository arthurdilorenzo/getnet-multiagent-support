# SPEC V11 — Addendum P06: replan com escopo de fonte explícito

## Baseline

Candidate: `c8b32ba3f9f5f9deac8752097b54160551fd77e7`.

## Falha observada

`_observe_and_replan` suprime replanejamento quando a mensagem contém Crediário, Brasil, Noruega e documentação e a etapa Knowledge retorna suporte parcial do Brasil. A condição também captura uma unidade da Noruega que pede explicitamente regras atuais via fonte externa, impedindo a etapa Web solicitada.

## Contrato

1. Preservar resposta parcial documentada: fatos sustentados do Brasil devem ser entregues e a Noruega pode ser declarada sem suporte documental interno, sem replanejamento redundante, quando o pedido for somente sobre a documentação disponível.
2. Não suprimir uma unidade pendente que peça explicitamente fonte externa, busca atual ou autoridade externa. Intenção explícita de fonte deve prevalecer sobre a heurística de “comparação documental sem ganho”. O replanejamento deve manter-se limitado ao escopo Noruega e não duplicar a unidade brasileira já cumprida.
3. A regra de evitar replanejamento sem ganho deve depender de evidência de cobertura e escopo da unidade pendente, e não apenas da presença de termos como Crediário, Brasil, Noruega e documentação na mensagem inteira.
4. Falha, ausência ou resultado parcial da fonte externa não autoriza afirmar fatos não sustentados; responder parcialmente e explicitar a lacuna.

## Aceite

- Regressão do pedido P06 original: Brasil sustentado + Noruega documentalmente sem cobertura -> preserva fatos brasileiros, informa limitação norueguesa e não faz busca externa/replano redundante.
- Caso contrastante: mesma composição temática mas unidade Noruega pede explicitamente fonte externa/regras atuais -> a unidade pendente chega ao plano Web uma única vez; a unidade brasileira cumprida não é refeita nem perde seus fatos.
- Cobrir inglês e português, cobertura parcial/abstained e fonte externa indisponível; nenhuma condição pode fabricar evidência ou repetir unidade cumprida.
- TDD focado, regressões P06/M08 e suíte completa; auditoria independente do SHA novo. Não atribuir ganho de nota V4 sem freeze nativo vinculado ao SHA exato.
