# V11 — contratos ponta a ponta após F519

Baseline: f519090b24fd44341e3ba5c0733c5c059255714b. Data: 2026-10-23.
Autoridade: handoff e adjudicação GETNET_V11_F519090_NATIVE_ADJUDICATION_2026-10-23.md no workspace pai. Nenhuma nova amostra nativa autorizada.

## Diagnóstico antes do código

A1/A2/A3/A6 compartilham perda de representação: request_units quebra conectores e frases sem preservar a relação entre tarefa, contexto e qualificadores; Router/Knowledge tentam reparar somente formas literais. A gramática condicional acopla o predicado à frase da conclusão. Assim o alvo existe mas a leitura não acontece; modificadores tornam-se trabalho; o argumento final conserva contexto sem tarefa ou país sem produto.
A4 tem outra causa: minimização local depende de todos os fragmentos e de vocabulário permissivo finito; delegações separadas e qualificadores desligam a projeção. O agregador depois copia evidence e texto sem uma barreira pública por alvo/campo.
A5: vínculo de ID ao pedido não é autorização quando o próprio pedido declara proprietário estrangeiro. Handler tenant-scoped continua segunda barreira.
A8: consentimento para proposta passa pelo parser de imperativo de execução; recusas usam o default completed de ToolExecutionResult.
A7: fallback de falha Web não comunica especificamente a não confirmação do incidente pedido.

## Contratos vinculantes

1. Unidades canônicas conservam offsets/texto literal, não geram whitespace/pontuação como necessidade e vinculam qualificadores de fonte/formato ao pedido que qualificam. Não promover cláusulas substantivas a qualificadores. Contexto e tarefa documental ficam ligados no argumento efetivo, sem incorporar uma unidade privada independente.
2. Router e Knowledge consomem o mesmo escopo canônico. O argumento real de tool inclui tarefa + assunto + geografia/restrições pertinentes após replan, mesmo se o modelo propõe apenas contexto ou qualifier. Preservar fatos Brasil após Web falhar; sem inferir Noruega. Fonte externa explícita continua externa. Não creditar qualifier separado como fato.
3. Condição: separar identificação segura da leitura do reconhecimento exato do predicado. Alvo próprio inequívoco permite read-only mesmo se condição não reconhecida; dependente só após todos predicados reconhecidos e provados por observação tipada do alvo/tenant exatos. Não relaxar disjunções, negacões, ranges, atributos extras ou ambiguidade. Comunicar valores realmente observados/requisito divergente; jamais inventar. Condição desconhecida bloqueia dependente.
4. Projeção pública após agregação: campos pedidos por terminal são a allowlist para texto/evidence/trace/citações/intent e memória pública persistida. Para sinal A e conexão B, só pares exatos; sem status/firmware/modelo/sinal B/conexão A/instruções/hardware extra. Delegações separadas equivalem a agregadas. Pedidos independentes legítimos continuam intactos; não usar ausência de reconhecimento de um fragmento como liberação de dados. Observações internas podem ser ricas.
5. Tenant: rejeitar antes de registry/handler terminal explicitamente de fornecedor/terceiro, em PT/EN. Não substituir alvo por próprio, não persistir proposta, não consultar perfil para contornar. Preservar handler tenant-scoped e principal autenticado.
6. Proposta: autorização explícita de criar/preparar proposta para ação/alvo próprios permite somente PENDING_CONFIRMATION persistida, nunca execução. Pergunta informativa/hipótese/citação/veto de proposta não autoriza. Veto de execução mantém-se e não veda proposta expressamente pedida. Confirmar continua via endpoint JWT do titular + hash + isolamento/idempotência existentes. Toda recusa/erro conhecido anterior à execução é failed, nunca completed. Escrita já iniciada com resultado indeterminado continua unknown e exige reconciliação; nunca converter unknown em failed para permitir retry.
7. Web indisponível para incidente atual: informar especificamente que incidente/alvo/local não está confirmado, sem inventar causa/incidente/citação ou consultar dados privados.

## Aceitação e plano

Criar regressões RED antes de produto e variantes PT/EN, inspecionando JSON HTTP integral via ASGI /api/chat com fixtures sintéticas e spies de argumentos reais. Não basta chamar helper. Verificar evidence/trace, ledger, isolamento e persistência de proposta/memória. Exercitar transporte FC com provider sintético explicitamente identificado, sem Gemini/rede. Não alterar labels/artefatos F519 ou relaxar gates.

Frente A (agente): request_units/documentary_intent, conditional_dependencies, router_agent, knowledge_agent, orchestrator; testes próprios de A1/A2/A3/A6/A7. Sem editar suporte/constraints/schema/egress.
Frente B (agente): customer_support_agent, request_constraints, fulfillment, schemas/registry se necessário e novo app/security/public_projection.py; testes próprios A4/A5/A8. Fornecer função de projeção final e contrato para integrador. Sem editar orchestrator/router/knowledge/unidades documentais.
Integrador: ligar projeção final depois agregação e antes persistência, validar integração e suíte completa offline/AST/diff. Alterar orchestrator somente após frente A encerrar.
Revisor independente: sem implementar; revisar diff inteiro versus spec/segurança e testes públicos; registrar gaps reproduzíveis. Correções voltam aos donos, depois revisão focada.

## Restrições globais

Tudo local no worktree /private/tmp/getnet-v11-worktree, branch codex/getnet-v11-conditional-pix. Sem rede/Gemini/chaves/env privado, push/merge/publicação. Main intocado. F519 consumida e imutável; testes offline novos são regressões, sem nova nota. Publicação só com futura nota independente real >=95, gate semântico e CI verde no SHA exato, todos fora desta verificação offline.
