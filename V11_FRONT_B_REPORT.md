# V11 frente B — segurança pública, tenant e proposta

Baseline f519090; trabalho local compartilhado sem commit, push, main, Gemini ou rede. O pacote congelado F519 e seus labels não foram modificados nem reexecutados. Nenhuma nota foi atribuída.

## Causas e RED anterior ao produto

`tests/test_v11_public_safety_e2e.py` foi criado antes das alterações de produto. A primeira execução teve cinco falhas reais: proposta explicitamente solicitada em PT/EN era recusada pela gramática de imperativo de execução; declaração de propriedade estrangeira em PT/EN passava por `constrain_call`; `ToolExecutionResult(success=False)` herdava `completed`. Um sexto RED separado mostrou que cláusula de apresentação (“uma linha curta”) desabilitava o minimizador inteiro e expunha firmware sentinela. O integrador também preservou RED HTTP próprio de sentinelas em envelope/memória antes da integração.

## Implementação

- `app/security/public_projection.py`: contrato determinístico por alvo/campo nas unidades originais; qualificadores desconhecidos não desligam a projeção. União de campos para o mesmo alvo é projetada. `response`, `structured_data` e `evidence` do Support são reconstruídos a partir de resultados verificados com titular/ID correspondentes. Profile explicitamente solicitado e outros resultados independentes continuam; diagnóstico amplo explicitamente solicitado de outro alvo continua. Observações `tool_results` ricas permanecem INTERNAS e intactas para prova tipada do supervisor. `ChatResponse` não serializa esse campo.
- Projeção final recompõe texto/evidence, remove citações/sources diagnósticos arbitrários, substitui intent livre e details de todos os traces sob contrato de campos, restringe annotations de coverage e reconstrói propostas de resultados reais. Mantém estados de abstention/partial e metadados do envelope. A escolha conservadora de não publicar details livres reduz detalhes de observabilidade pública nesses pedidos; eventos/status/timing continuam. Não transforma o helper em certificado semântico.
- `customer_support_agent.py`: removeu o minimizador anterior e sua lista de palavras permissivas. O gate da ferramenta que SOMENTE propõe agora reconhece criação/preparação explícita de proposta para ação/alvo, distinta do consentimento de execução. Informação, hipótese, citação e veto de proposta continuam recusados; veto de execução não cancela proposta expressamente pedida. JWT/hash/tenant/TTL/idempotência do endpoint permanecem.
- `request_constraints.py`: propriedade estrangeira afirmativa por alvo bloqueia antes de registry/handler, inclusive fallback de perfil que tentaria contornar a negativa. Menção de fornecedor em assunto independente e declaração negada não são tratadas como propriedade estrangeira. Handler continua tenant-scoped como segunda barreira.
- `schemas.py`: validador impede falha conhecida com default `completed`. Estados explícitos `pending`/`unknown` de operação iniciada/indeterminada são preservados para reconciliação.

## Interface exata para integração

```python
from app.security.public_projection import project_public_response
projected = project_public_response(
    response,                       # ChatResponse já agregado
    original_message,               # texto original com máscara de PII
    outputs,                        # list[SpecialistOutput], internos ricos
    authenticated_user_id,          # principal autenticado
    trusted_segments=trusted_segments,  # keyword opcional list[str] | None
)
```

Retorna cópia `ChatResponse`; chamar ANTES de `append_message(role='assistant', content=projected.response)`. `trusted_segments` recebe somente notices realmente produzidos pelo servidor (condição, limite/replan); nunca rationale/texto livre do modelo. É idempotente. Sem diagnóstico efetivo nos outputs, preserva a resposta agregada/notice existente e sanitiza intent/trace. O integrador pode reaplicar depois de anexar trace final, desde que o texto persistido já seja o projetado. Não usar `project_specialist.tool_results` como payload público: são deliberadamente observações internas ricas.

## Verificação focada GREEN

Python usado: `/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python`.

```
python -m pytest tests/test_v11_public_safety_e2e.py tests/test_operational_scope_v8.py -q
112 passed, 1 warning
```

Os 26 casos novos incluem HTTP serializado `/api/chat` com principal JWT e fixture sintética própria, registry REAL com spy de argumentos efetivos, provider FC nativo de SDK inteiramente sintético para bloqueios/propostas, preparação PT/EN, hipótese/informação/citação/veto negativos, alvo estrangeiro sem dispatch, multimáquina PT/EN em delegação única/separada e igualdade da memória assistente com texto HTTP. Proposta é realmente persistida PENDING_CONFIRMATION; zero `terminal_actions` antes da confirmação; titular alheio e hash errado negados; confirmação correta/replay produz exatamente um receipt. Teste adicional valida idempotência de projeção e preservação simultânea de perfil/proposta/Knowledge/notices independentes.

A expectativa antiga que mandava desabilitar projeção quando o mesmo alvo tinha dois campos foi substituída pelo contrato vinculante V11 de união por alvo. Asserts antigos que serializavam SpecialistOutput interno como se fosse resposta pública agora verificam `response/evidence/structured_data`; há regressões HTTP integrais equivalentes e `tool_results` bruto segue disponível para condição tipada. Não houve alteração de labels congelados.

Uma execução adicional não centralizada de boundary+conditional durante edição concorrente e ANTES da integração final produziu falhas transitórias (134 failed/111 passed). Não foi declarada como GREEN nem substituída por alegação de suíte integral. Suíte integral, teste de sentinelas do integrador e revisão final pertencem ao integrador/revisor; resultados offline não demonstram qualidade ou disponibilidade Gemini nem autorizam promoção.

## Ajuste de integração posterior

A integração expôs dois contratos adicionais sem mudar assertions: metadados tipados do gate devem continuar observáveis e firmware ausente deve ser nomeado como tal. A projeção de trace passou a preservar somente por action/agent e tipo: `predicate_authorized` booleano, alvo presente no pedido, reason enum do gate, `success`/replay booleanos de calls registradas, listas de agentes validados, contadores e stopping_reason enum. Details arbitrários/freeform continuam removidos e um teste novo verifica sentinelas e tipos/IDs inválidos. A mensagem de campo ausente agora explicita `Firmware: não informado`, sem criar evidence de firmware; a observação de alvo próprio preserva `partial`/pendência.

Comando focado posterior (novos B + boundary integral do integrador + dois testes de ausência firmware + escopo legado): **116 passed**, um aviso upstream. O teste A `test_exact_predicate_does_not_depend_on_consequence_wording` também foi executado depois do ajuste para verificar diretamente o contrato `predicate_authorized` (**2 passed**, um aviso upstream). Nenhuma assertion foi alterada para esconder estas falhas de integração.

## Última onda conhecida — no-diagnostic, plural e coerência

Dois RED HTTP antes do fix demonstraram o bypass sem `diagnose_terminal_issue`: prose/evidence de perfil não solicitado e prefixo livre de operador humano sobreviveram tanto ao envelope quanto à memória. Probe preservado em `/private/tmp/v11-b-no-diagnostic-red.txt`. Corrigido: qualquer output Support sob contrato de campo é projetado, inclusive zero resultados diagnósticos; a composição final sempre usa outputs seguros e notices server-owned fornecidos pelo integrador. Conhecimento independente continua preservado. Ausência é apresentada por par; perfil não pedido não serve como substituição. Observações internas seguem intactas.

Atribuição plural afirmativa explícita (“both terminals belong to my supplier” / “ambos os terminais pertencem ao fornecedor”) agora vincula a declaração aos dois alvos nomeados e nega ambos antes de registry. Testes HTTP PT/EN verificam spy sem chamadas; menção independente de fornecedor e relação negada continuam permitidas.

A projeção final também rebaixa unidades de campo sem prova para unresolved e resposta para partial/abstained; nunca promove estado. Fulfilled de trabalho independente permanece intacto, com teste dedicado de envelope complete inconsistente. Resposta parcial com conhecimento verificado independente permanece partial.

Verificação estável desta onda:

```
python -m pytest tests/test_v11_public_safety_e2e.py tests/test_v11_public_boundary_integration.py tests/test_v11_canonical_e2e.py::test_exact_predicate_does_not_depend_on_consequence_wording tests/test_operational_scope_v8.py tests/test_fulfillment_v7.py::test_run_missing_firmware_preserves_unknown_in_text_and_pending_state tests/test_v10_mixed_pix_firmware_handler.py::test_chat_keeps_pix_fact_when_requested_firmware_is_absent -q --tb=short
126 passed, 1 warning
```

`git diff --check` passou. A suíte integral final fica centralizada no integrador. Sem alterações em orchestrator/A/labels, rede ou execução Gemini nesta frente.

Último resíduo do mesmo finding: o recheck do revisor forçou `covered=true` sem `tool_results` em duas unidades, a segunda começando por `; informe`. A classificação por gramática de verbos omitia essa segunda unidade. Agora a determinação usa o próprio contrato alvo/campo canônico (com prefixo literal removido), e a prova vem exclusivamente de resultados diagnósticos completed/success, found, principal/ID exatos e campos úteis. Não depende de `covered_unit_ids` alegado pelo especialista. O teste HTTP B agora também força complete/covered sem diagnóstico e exige todas as linhas não fulfilled. Recheck do revisor + novos B + boundary + condição exata: **39 passed**, um aviso upstream. Sem alteração dos asserts do revisor ou escopo novo.
