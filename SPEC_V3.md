# Getnet V4 — plano de correção da auditoria 672c87e

**Baseline:** `672c87e1a7a85a8eab95a36cfc5cc29fce58017f`, nota independente nominal 66,75/100. **Data:** 23/10/2026. **Autoridade:** pedido de Arthur para corrigir a auditoria com spec-driven/multiagentes, começar pelos P0, testar e reauditar. Publicar em `main` somente se uma avaliação independente sustentar 95+ sem inflação; caso contrário preservar o candidato em branch local e informar evidências/pendências. A rubrica normativa completa não está disponível; não inventar caps ou nota oficial de Marcos.

## Objetivo e arquitetura

Corrigir os onze achados A1–A11, manter o núcleo local FastAPI/GenAI/SQLite e demonstrar comportamento correto além dos exemplos oficiais. O LLM decide delegações/ferramentas, mas identidade, consentimento, campos factuais, estados e permissões são contratos do servidor. Respostas financeiras/operacionais devem ser derivadas de evidências tipadas; redação livre não pode confirmar fatos/ações ausentes. Retrieval precisa verificar se a evidência responde à pergunta, não só medir proximidade.

```text
HTTP autenticado → PII/input limits → sessão/fatos/resumo → supervisor
  → Router function calls → especialistas → observações → replanejamento limitado
  → Support: tools de consulta / proposta de ação → renderização de dados reais
  → Knowledge/Web: retrieval/APIs → answerability/evidência → extratos seguros
  → HITL: transições validadas + pausa de mutações + retomada transacional
  → resposta com evidências, estado e trace; nenhum texto inventa recibo

Ação POS: proposta imutável → confirmação autenticada por API → recibo SQLite
```

## Restrições globais

- Não alterar `main` durante implementação; trabalhar em `codex/getnet-v4-remediation`.
- Não expor/configurar credenciais em código, logs, fixtures, relatórios ou Git. `.env` local ignorado; somente o integrador usa chave real em testes opt-in. Testes comuns executam com `GEMINI_API_KEY=''` para não chamar o provedor inadvertidamente.
- Sem ChatGPT/Work/cloud, contatos externos ou dados reais de clientes. APIs públicas/documentação oficial podem fornecer evidência técnica; SDK real recebe apenas fixtures sintéticas redigidas.
- Preservar interface visual, acrescentando somente controles necessários de autenticação/confirmação para uso correto do modo strict.
- Testes de regressão antes dos fixes (RED→GREEN); não alterar expectativas apenas para silenciar falhas. Mudanças de contrato de segurança devem ser explícitas nos testes/documentação.
- Não declarar qualidade semântica pelo score lexical antigo, nem ajustá-lo para perseguir 95. Métricas por categoria, contraexemplos e limitações são parte da entrega.
- Evidência gerada com mocks é rotulada; Gemini e Docker reais são verificações separadas. Falta de chave/rede/daemon não vira aprovação fictícia.

## Contratos de integração

### Respostas e ferramentas

- `CustomerSupportAgent.run(...) -> SpecialistOutput`: mantém loop nativo e resultados de função, mas compõe a resposta final de campos retornados pelas tools. Texto livre do LLM não é autoridade para saldo, depósito, execução ou protocolo. Falhas, pending confirmation e resultado incerto ficam visíveis. Nenhum cliente/terminal explícito inexistente pode ser substituído silenciosamente.
- `KnowledgeAgent.run(...)`: saída por extratos/fatos sustentados e citações; nunca devolver diretamente prosa do provedor como fato validado. Suporta evidência explícita de answerability e abstém quando país, operação, data, taxa ou outra restrição da pergunta não estiver coberta. Cabeçalho/URL sozinhos não são resposta. Confiança não recebe piso arbitrário de 0,65.
- `ToolExecutionResult` recebe campos opcionais compatíveis: `operation_id: Optional[str]`, `outcome_state: Literal['completed','pending','unknown','failed'] = 'completed'`. Timeout de escrita em execução retorna resultado desconhecido, `retryable=False`, ID consultável. Erro de validação/ação negada nunca vira sucesso por prosa.
- `ToolRegistry.get_operation(operation_id: str, security_context: SecurityContext) -> dict`: consulta autenticada e escopada do estado da operação; resultado terminal ou incerto descrito honestamente. Fila de workers limitada e sem retry cego de escrita.
- `RegisteredTool.required_scopes`: tools de cliente exigem `customer:read_own`; proposta POS exige `terminal:action_own`; chat/HITL usam permissões apropriadas. Tools públicas não recebem permissão administrativa implícita. Role/tenant não podem ser escolhidos pelo LLM.

### Proposta e confirmação de ação

- Toda ação remota solicitada pelo agente prepara uma proposta persistida e imutável; não grava `terminal_actions` como executada nesse turno. O nome legado `execute_terminal_remote_action` pode ser mantido, com descrição/status explícitos de proposta, por compatibilidade.
- `action_policy.propose_action(user_id, session_id, terminal_id, action_type, reason='', idempotency_key='') -> dict`: valida frota, estado HITL e permissão na fronteira, retorna `proposal_id`, alvo/tipo, `payload_hash`, `status='PENDING_CONFIRMATION'`, expiração e instrução de confirmação. Não fornecer ao LLM um mecanismo de autoconfirmação.
- `action_policy.confirm_action(proposal_id, user_id, expected_payload_hash) -> dict`: somente chamada por endpoint autenticado de usuário; valida owner, sessão, hash, expiração, estado de intervenção humana e idempotência; persiste recibo/estado em transação. Repetição da mesma confirmação retorna recibo anterior, nunca nova ação. Mesmo idempotency key com payload divergente retorna conflito.
- `POST /api/actions/{proposal_id}/confirm`: body `{user_id, expected_payload_hash, confirmation:'CONFIRM'}`; identidade vinculada à credencial, sem modo demo não verificado para confirmar. Nenhuma tool de confirmação é disponibilizada ao LLM.
- `GET /api/tool-operations/{operation_id}?user_id=...`: estado de execução do registry somente ao principal autorizado.
- Autorização condicional, adiada, citada, negada ou ambígua não gera execução. A proposta pode solicitar confirmação futura, preservando o texto/intenção para auditoria. Nova intenção legítima recebe identidade própria; não usar hash permanente de user+terminal+tipo como idempotência universal.

### Memória, HITL e supervisor

- `StateStore.get_conversation_context` mantém campos atuais e acrescenta fatos/entidades de sessão quando necessário. O resumo é efetivamente consumido pelos agentes, rotulado como dados, com redaction e limites. Continuação e mudança de assunto têm decisões diferentes; não concatenar toda pergunta curta à anterior.
- `StateStore.has_pending_human(session_id, user_id) -> bool`: verifica owner e presença de tickets `OPEN/CLAIMED/RESPONDED` que impedem mutações automáticas. Consultas seguras podem continuar; confirmação de escrita exige sessão liberada.
- HITL: `OPEN→CLAIMED→RESPONDED/RESOLVED`; reply direto em OPEN pode claim+reply atomicamente pelo operador. Ticket resolvido não pode ser reivindicado/respondido silenciosamente. Outro operador não substitui responsável sem reatribuição explícita. Resolver um ticket não libera sessão com outro pendente. Reabertura, se implementada, é ação explícita auditada.
- Supervisor propaga observações entre especialistas; router pode replanejar por resultados, sempre validado e limitado. Demonstrar ao menos dois gatilhos além de simples fila fixa (hardware e evidência insuficiente/falha ou consulta dependente). Respostas de múltiplas intenções preservam falhas parciais e usam resultados reais do passo anterior.
- `RouterAgent.replan(original_message, user_id, observations, conversation_context=None, tracer=None) -> RoutingPlan`: contrato recomendado para feedback nativo; sem chave usa política explícita segura. Evitar ciclos de agentes repetidos sem progresso.

### APIs públicas e oráculo

- Câmbio: extremos ausentes ficam nulos/omitidos; referência diária é rotulada como referência, não cotação intradiária atual. Data/hora/fonte acompanham valores. Precipitação ausente é desconhecida, nunca 0. Campos numéricos não finitos/intervalos inválidos são rejeitados.
- Novo avaliador mede evidência/corretude de afirmações/abstenção e precisão de seleção de ferramentas, além de recall. Contradições com palavras corretas, números não suportados e flags de abstention com fatos fabricados devem falhar.
- Dataset independente congelado, autoria separada dos implementadores, ≥30 paráfrases PT/EN e ≥20 perguntas com lacunas plausíveis, com expectativas e trechos de suporte ou motivo de abstenção. Não alegar revisão humana quando labels foram produzidos por agente. Nova auditoria acrescenta casos inéditos depois da implementação.

## Matriz de correções e aceites

| Prioridade/achado | Aceite verificável | Critérios V4 principais |
|---|---|---|
| P0 A1 campos inventados | Fixtures de FX/clima completos, incompletos, nulos, antigos e inválidos: nada ausente vira fato; rótulo referência correto | 1,3,8,9,11 |
| P0 A2 texto sem evidência | Fake SDK diz reboot/depósito após profile: saída final não confirma ação/número; idem Knowledge prosa falsa com URL | 2,3,4,8 |
| P0 A3 consentimento | Pedido adiado/citado/negado não executa; confirmação autenticada owner+hash+TTL; replay único; cliente/sessão divergente rejeitados | 4,5,7 |
| P0 A4 PII/scopes | CPF/senha/CVV em linguagem natural ausentes de DB/log/trace/payload; role/scopes/tenant negativos 401/403 | 5,8 |
| P0/P1 A5/A10 RAG/oráculo | Perguntas senha/Pix país/cancelamento/prazo/chargeback sem evidência abstêm; paráfrases cobertas respondem com excertos; contradição não pontua 1 | 1,3,8 |
| P1 A6 memória | Troca antecipação→Pix, terceira referência, entidade além da janela, restart e resumo consumido | 2,6 |
| P0/P1 A7/A8 tools | ID inexistente não troca alvo; mesma key+payload diferente conflito; timeout write unknown sem retry e reconciliável | 4,6,9 |
| P1 A9 HITL | Transições inválidas rejeitadas; operador atribuído; dois tickets não liberam sessão prematuramente; confirmação POS bloqueada em pausa | 5,6,7 |
| P1 A11 supervisão | Observações reais alimentam especialista/plano seguinte, loop limitado e falhas parciais explícitas | 2,6,7 |
| P1/P2 produção | Lock instalado, Docker build/smoke/restart em CI, strict acessível pela UI/cliente, deadlines e concorrência medidos | 9,10,11 |

Pesos mantidos: Fit8; Orchestration15; RAG10; Tools10; Security13; Memory8; Human7; Evaluation14; Observability6; Production5; Judgment4. A soma é100. Nota independente não pode ser retrocalculada pela meta.

## Divisão de trabalho e sequência

1. **Agente Segurança/Ações (P0):** `app/security/pii.py`, novo `app/security/action_policy.py`, `app/tools/{registry,customer_tools}.py`, `app/agents/customer_support_agent.py`, testes novos próprios. Não editar main/config/schemas/store sem coordenar.
2. **Agente Conhecimento/Factualidade/Evals (P0/P1):** `app/agents/knowledge_agent.py`, `app/tools/search_tools.py`, `app/rag/`, corpus/índice se necessário, `app/evals/evaluator.py`, testes próprios. Não ler dataset reservado do revisor durante implementação.
3. **Agente de avaliação independente inicial:** cria e congela dataset reservado fora do repo, com fontes/labels/limites. Depois libera slot para agente Memória/HITL/Supervisor (`app/memory/store.py`, `app/hitl/queue.py`, `app/agents/{orchestrator,router_agent}.py`, testes próprios).
4. **Integrador:** config/schemas/SDK helper/deadlines, main/endpoints/UI, Docker/CI, scripts de smoke/live, README/evidências, revisão de integração e publicação condicionada. Não compartilhar .env/chaves com agentes; testes offline por padrão explícito.
5. **Revisor novo final:** executa suite/harness/reproducers antigos e novos, avalia os11 critérios/4gates com nota independente. Resultado <95 ou gates/evidência insuficientes impede atualização de main. Corrigir achados concretos e reauditar sem direcionar a nota.

## Verificação e publicação

- Testes completos, checker, harness por categoria e regressões de A1–A11; nenhum P0/P1 funcional pendente.
- Prova Gemini opt-in com fixtures sintéticas redigidas; nenhuma chave nos artefatos. Falha de autenticação/quota/rede é registrada e não recebe nota de sucesso.
- Docker local via runtime existente, build/smoke/restart e versão Python3.11; medir concorrência e timeout. Não destruir VM/volumes de outros trabalhos.
- Revisão e auditoria independente do candidato exato (SHA local), sem inflar pelo número de testes. 95+ precisa ser sustentado nos critérios e gates; ausência da rubrica completa é declarada.
- Somente após gate: integrar em main, push autorizado, conferir SHA e Actions success. Se necessário validar workflow em branch de candidato, explicar e não tratá-lo como publicação aprovada em main.
- Atualizar registro detalhado e WORKLOG com resultados, limitações, decisões e pendências; manter .env, DB, caches e fixtures sensíveis fora do Git.
