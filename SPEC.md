# Getnet V4 — especificação e plano de implementação

> Este plano descreve a correção anterior ao commit `672c87e`. Os ciclos seguintes estão em [SPEC_V3.md](SPEC_V3.md) e [SPEC_V4.md](SPEC_V4.md). A execução atual, derivada da auditoria de `a79c52c` (85,15/100) e condicionada a95+ antes de main, está em [SPEC_V5.md](SPEC_V5.md).

**Data:** 23/10/2026 · **Baseline:** `b6c8b09` · **Meta:** corrigir os cinco gaps auditados e fornecer evidência para reavaliação 95+. A nota só pode ser atribuída pelo avaliador externo.

## Objetivo e restrições

Preservar FastAPI, Google GenAI, SQLite, ToolRegistry e a interface existente. Implementar function calling real, fechar acessos entre tenants, remover ruído do corpus e alinhar documentação ao comportamento verificável. Nenhum dado externo ou capacidade será inventado. Execução autorizada por Arthur: spec → agentes em paralelo → testes/evals → commit e push em `main` → Actions verde. Trabalho no checkout local existente; sem ChatGPT/Work/cloud e sem compartilhamento de arquivos de carreira. Não versionar credenciais, bancos SQLite ou arquivos temporários.

## Arquitetura alvo

```text
HTTP → autenticação/role/tenant → guardrails → memória por proprietário
    → RouterAgent (Gemini function calls de delegação; RoutingPlan validado)
    → supervisor limitado a MAX_ORCHESTRATION_ITERATIONS
        → CustomerSupportAgent (Gemini → ToolRegistry → FunctionResponse → Gemini)
        → Knowledge/Web (retrieval/APIs e abstention)
        → EscalationAgent (fila SQLite; operador autenticado responde)
    → sanitização → persistência → resposta/trace
```

Sem `GEMINI_API_KEY`, o roteamento e o planejamento local determinísticos continuam explícitos e testáveis. Com chave, erros/saídas inválidas do provedor devem produzir erro controlado/abstenção; não podem cair silenciosamente na seleção por palavras-chave. Loops têm limite de turnos e chamadas. Não existe promessa de entendimento neural no fallback.

### Contratos

- `RouterAgent.plan(message, user_id, conversation_context=None, tracer=None) -> RoutingPlan`: usa ferramentas de delegação com `sub_query` e `rationale`; nomes conhecidos, argumentos estritos e número limitado de passos; sem identidade escolhida pelo LLM. Conversão validada para `PlanStep` e `RoutingPlan`.
- `CustomerSupportAgent.run(message, security_context, session_id='', conversation_context=None, tracer=None) -> SpecialistOutput`: apresenta apenas tools operacionais permitidas via `ToolRegistry.get_gemini_function_declarations(names=...)`; desativa execução automática do SDK; executa as chamadas pelo registry; envia `types.Part.from_function_response(...)` com resultados reais na conversa seguinte. Diagnóstico de hardware mantém sinalização de escalação; nenhuma tool executa fora do registry.
- `ToolRegistry.execute(tool_name, raw_arguments, security_context, session_id='', trace_id='') -> ToolExecutionResult`: valida schema, vincula `user_id`/sessão no servidor, valida propriedade e mascara resultados. Calls não autorizadas ou argumentos malformados não alcançam handlers.
- Endpoints HITL de listagem **e detalhe**, métricas e execução de evals: autenticação administrativa quando houver dados/operações administrativas. Escolha de implementação: HITL só para operador (401 sem credencial, 403 para merchant), preservando filtros opcionais após autorização.
- Produção exige autenticação estrita e segredos externos fortes; configuração JWT ativa impede downgrade para sessão anônima. Credenciais inválidas nunca caem no modo demo. Desenvolvimento sem segredos pode usar modo demo explicitamente identificado, com segredos efêmeros em memória quando necessários; não é autenticação de produção.
- Sessões são autorizadas por proprietário persistido no SQLite, nunca apenas por prefixo do identificador; protege leitura, escrita, exclusão e resposta HITL.
- Corpus: conteúdo útil de produto/serviço, URL oficial, título, hash atualizado e identificação da proveniência. Limpeza reproduzível na ingestão e aplicada ao JSON versionado; índice coerente com o corpus. Não reescrever fontes como se fossem novos fatos.
- RAG real: BM25 com expansão léxica + projeção vetorial por hashing com âncoras conceituais + RRF + threshold e abstention. Não há encoder neural denso instalado. Nomes legados podem ser preservados por compatibilidade, desde que documentados.

## Matriz de requisitos × Régua V4

| Critério (peso) | Evidência alvo / limite |
|---|---|
| 1. Business & Challenge Fit (8%) | 10 cenários existentes, corpus oficial útil, falhas externas honestas |
| 2. Agentic Reasoning & Orchestration (15%) | delegação Gemini nativa; loop observa FunctionResponse e escolhe próximo passo; limites explícitos |
| 3. Context, Knowledge & RAG (10%) | corpus higienizado, URLs/hashes, retrieval documentado sem encoder fictício |
| 4. Tool & Action Engineering (10%) | schemas do registry consumidos pelo Gemini; identidade protegida, escrita idempotente, erros estruturados |
| 5. Security & Identity (13%) | HITL protegido, JWT validado, produção estrita, sem segredo default fixo, isolamento de sessões |
| 6. Memory, State & Resilience (8%) | memória SQLite por tenant, regressões multiturno e falhas externas |
| 7. Human Oversight & Risk Management (7%) | fila/reply reais, operador autenticado; sem promessa de SLA ou entrega externa não confirmada |
| 8. Evaluation & Quality Engineering (14%) | testes de regressão e SDK simulado no protocolo real; suíte completa + harness com limites declarados |
| 9. Observability, Reliability & Cost (6%) | logs estruturados, trace de decisões/falhas, nenhuma chamada de console em app; tokens/custo estimados declarados |
| 10. Production Engineering (5%) | CI confiável, verificação sintática Python, configuração segura e higiene Git |
| 11. Engineering Judgment (4%) | README auditável, trade-offs preservados, evidências distinguem testes offline de execução real do Gemini |

### Gates

- **Critical Security:** nenhuma listagem/detalhe HITL anônima, nenhum acesso de merchant a outro tenant, nenhuma identidade privilegiada escolhida pelo LLM.
- **Challenge Fit:** manter implementação local do núcleo e corpus com fontes oficiais; sem substituir por SaaS pronto.
- **Evidence:** toda claim aponta a código/teste; métricas do harness local não são nota V4 nem qualidade semântica universal.
- **Semantic Quality:** sem cotações/clima inventados, sem ruído administrativo no corpus, sem alegar atendimento humano realizado quando apenas enfileirado.

## Testes de aceite por gap

| Gap | Aceite verificável |
|---|---|
| 1. CI | docstrings/comentários/strings com `print(` passam; chamadas Python reais `print(...)`, inclusive multilinha e `builtins.print`, falham; verificar apenas `.py`; executar a mesma checagem usada no Actions |
| 2. Function calling | SDK simulado retorna call → registry executa → resposta de função segue ao SDK → texto final; duas rodadas dependentes; roteamento por call com mensagem sem keywords; tools desconhecidas, argumentos inválidos, spoof de identidade, falha de SDK e limite de loop têm resultados controlados; fallback só sem chave |
| 3. Segurança | HITL list/detail anônimos=401, merchant=403, operador autorizado=200; JWT forjado/expirado/malformado rejeitado; produção sem segredos fortes falha na configuração; JWT não permite downgrade; sessões não leem/escrevem/apagam dados de outro tenant |
| 4. Corpus/RAG | JSON sem CPF solicitado, SAC/telefones, Black Friday, menus e cookies; limpeza idempotente, preserva fatos de produtos e URL/hash; testes retrieval/OOD/citações continuam passando sem afrouxar threshold para acomodar avaliação |
| 5. README | cada claim revisada contra código; pipeline local descrito literalmente; modo offline, mock de operações, SQLite, limitações de guardrails, heurística de eval e ausência de teste Gemini ao vivo documentados; trade-offs mantidos |

## Execução paralela e propriedade dos arquivos

1. **Segurança/Auth:** `app/config.py`, `app/security/auth.py`, `app/main.py` (autorização), `app/memory/store.py`, configurações `.env.example`/compose se necessário e novos testes de segurança.
2. **Gemini:** `app/agents/router_agent.py`, `app/agents/customer_support_agent.py`, `app/tools/registry.py`, utilitário de function calling, novos testes; coordenar qualquer alteração ao orchestrator com integrador.
3. **CI/Corpus/README:** `.github/workflows/ci.yml`, checker de console, `app/observability/logger.py`, `app/rag/`, corpus/índice, README e testes próprios. Não alterar contratos de auth sem coordenação.
4. **Integrador:** evidências/evals, correções de integração, revisão independente, SPEC, WORKLOG, execução final e Git. Nenhum agente faz commit/push ou modifica arquivos de outra frente sem avisar.

Em cada frente: escrever regressões, observar falha da baseline, implementar, verificar testes focados e relatar resultados. Revisar diff integrado antes de publicar.

## Verificação e entrega

- Baseline reproduzida e SPEC criada antes de código de produto.
- Três frentes implementadas e revisadas; nenhum achado crítico/alto pendente.
- `source .venv/bin/activate && pytest -v` passa integralmente.
- `python -m app.evals.evaluator`: routing accuracy=1.0; mean tool recall=1.0; mean groundedness>=0.90. O CLI deve retornar código diferente de zero se critérios falharem.
- Checker do Actions passa; arquivos temporários/segredos ausentes do staging.
- Registrar comandos, resultados, limitações e próximos passos em registro detalhado e WORKLOG.
- Commit claro em `main`, `git push origin main`, confirmar SHA e resultado verde com `gh run list`/`gh run view`.

Resultados executados e revisão independente: [V4_REFACTOR_EVIDENCE_2026-10-23.md](V4_REFACTOR_EVIDENCE_2026-10-23.md). O SHA publicado e o run do Actions ficam registrados no WORKLOG operacional local e no histórico GitHub.

## Limites da evidência

A suíte offline testa o protocolo Gemini por respostas SDK simuladas, não comprova execução ao vivo sem chave/rede. O harness local tem dataset pequeno e juiz lexical baseado em termos/citações; não prova ausência universal de alucinação. Sem mensuração externa, não afirmar 95/100, produção certificada, cobertura completa nem qualidade semântica perfeita.
