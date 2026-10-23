# Integridade semântica e answerability — plano de implementação

> Execução autorizada por Arthur com spec-driven e multiagentes; retomada no medium. Usar TDD, frentes com ownership separado e revisão independente após congelamento. Não pedir nova aprovação para executar este escopo já autorizado.

**Goal:** corrigir as falhas da auditoria de `e7a0f10` (78,30/100), preservando a necessidade original através de delegação, ferramenta, evidência e resposta.

**Architecture:** o pedido original é autoridade sobre alvo/data/restrições; o modelo propõe chamadas, sem poder alterar essas restrições. Retrieval propõe candidatos; requisitos estruturados e fatos vinculados a spans autorizam somente extratos pertinentes. Similaridade e presença de palavras não são prova factual.

**Tech Stack:** Python/FastAPI/GenAI/SQLite existentes. Não adicionar biblioteca neural nesta rodada; reconsiderar somente com evidência de necessidade. Não alegar compreensão aberta universal pelo parser local.

**Spec e baseline:** sucede `SPEC_V3.md`; auditoria detalhada em `../GETNET_V4_AUDIT_V3_2026-10-23.md`. `main` permanece `672c87e`; trabalhar no branch `codex/getnet-v4-remediation`.

## Restrições globais

- `.env` não será lido/exibido pelos agentes nem versionado. Testes comuns com `GEMINI_API_KEY=''`; chamadas reais somente opt-in pelo integrador, sem ativar cobrança.
- Nenhuma resposta pronta por pergunta/case_id. O conjunto de 76 casos anterior agora é regressão conhecida, não novo teste cego; um revisor deve criar amostra inédita depois do novo freeze.
- O conteúdo factual apresentado continua extrativo/derivado de campos tipados. Campos, país, sujeito, unidade e condições precisam corresponder à evidência.
- Não mudar labels apenas para aprovar métricas. Indisponibilidade e abstenção correta são diferentes de resposta útil; publicar métricas separadas.
- Publicação em main continua condicionada a uma auditoria independente sustentando 95+, sem cap/âncora oficial inventados. CI verde é necessário, insuficiente.

## Contratos de interface

### Pedido operacional imutável

Criar `app/security/request_constraints.py`, sem dependências externas:

- `RequestConstraints` dataclass imutável: `original_message: str`, `terminal_ids: tuple[str,...]`, `transaction_ids: tuple[str,...]`, `sales_dates: tuple[str,...]`, `ambiguous_date: bool`.
- `extract_constraints(message: str, conversation_context: dict | None = None) -> RequestConstraints`: IDs POS com múltiplos segmentos, IDs de transação, datas ISO e PT/EN por extenso; referências inequívocas ao mesmo terminal usam entidade persistida. Texto é redigido antes de log/persistência.
- `constrain_call(tool_name: str, arguments: dict, constraints: RequestConstraints) -> dict`: produz cópia validada ou lança `RequestConstraintError(code, message)`. Não substituir alvo explícito por primeiro terminal. ID/data omitidos podem ser vinculados pelo servidor quando há exatamente um valor; valor divergente do modelo é rejeitado antes do handler. Ambiguidade exige esclarecimento.
- Support usa `authorization_message` original existente para extrair o contrato; nenhuma subquery do Router é autoridade para mudar alvo/data. O mesmo contrato vale no modo nativo e offline.
- `diagnose_terminal_issue` recebe `transaction_id: Optional[str]`; se explícito, filtra a transação exata e retorna ausência em vez de outra transação.
- `get_sales_settlement` recebe a data reconhecida e não retorna dinheiro de outro dia. Datas por extenso devem produzir o mesmo resultado que ISO; formato ambíguo não escolhe silenciosamente um dia.

### Preservação de necessidade documental

- `KnowledgeAgent.run(..., request_message: Optional[str] = None)` mantém assinatura compatível. O orquestrador fornece a mensagem original; chamadas diretas usam `message` como original.
- Uma query escolhida pelo SDK pode melhorar retrieval, mas a autorização factual verifica requisitos da mensagem original, não apenas a query reformulada. País, elegibilidade, operação e atributos originais não podem desaparecer.
- Para múltiplas intenções, verificar requisitos pertinentes ao especialista, preservando os qualificadores aplicáveis. Intenção não resolvida deve ficar explícita; não aprovar cobertura global por um único item respondido.

### Requisitos, fatos e extratos RAG

- Substituir o residual universal de palavras por requisitos explícitos: sujeito, relação/operação, valor solicitado, tipo de resposta e qualificadores. Cortesia/contexto incidental não vira atributo obrigatório; meio de pagamento, propriedade de hardware e restrição territorial não viram scaffolding descartável.
- Implementar catálogo de fatos/relações ancorados no corpus, sem perguntas/respostas prontas. Cada fato contém ID, sujeito, relação, valor/unidade, polaridade, condições/escopo quando documentados, chunk/hash e span literal verificável. O loader rejeita spans ou hashes inválidos.
- Sujeito e relação precisam coincidir: Link aceita Pix não estabelece Pix aceita cartão; hardware da Smart não se transfere à Clássica. Ausência é `unknown`, nunca negação inventada.
- O renderer recebe somente evidências aprovadas, preservando sentenças/ressalvas completas, URLs e a identificação de resumo curado. A saída registra cobertura por requisito e limitações do parser.
- Manter retrieval BM25/expansão + hashing/âncoras + RRF documentado. O parser local é delimitado; o score não será convertido em probabilidade factual.

### Integridade cambial

- A identidade do payload deve corresponder ao par solicitado. Validar `code/codein` quando fornecidos pelo primário e `base`, `amount`, moeda de destino no secundário. `base` incompatível, `amount` inválido ou unidade não verificável produzem erro/abstenção, nunca outro par com novo rótulo.
- Valores ausentes, extremos e proveniência preservam os contratos V3. Não inventar metadados para aceitar fixture incompleta.

### Avaliação e evidência

- O juiz limitado precisa rejeitar inversão de sujeito/autoridade, não só números/negação. Preferir referências estruturadas ou extratos verificáveis; alegação material não verificável deve exigir revisão, sem receber `passed=true` por termos corretos.
- Medir separadamente resposta útil em casos cobertos, abstenção em lacunas, associação sujeito–relação, preservação de restrições e integridade de unidades/spans. Anexos do próprio candidato provam rastreabilidade, não pertinência independente.
- Smoke Docker deve registrar SHA do código e image ID/digest para vincular recibo ao candidato. Gemini: preservar falhas reais, usar tentativas limitadas somente em geração sem execução automática, dentro de deadline, se justificadas; não promover retry a sucesso fictício.

## Review focus e testes de aceite

| Classe | Aceite |
|---|---|
| Contexto/cortesia e paráfrases | Adicionar cortesia ou contexto não muda uma resposta suportada; pelo menos variantes PT/EN novas por relação documentada. |
| Propriedade/relação ausente | Trocar sujeito, atributo, país, operação ou meio de pagamento não pode obter o mesmo extrato genérico como resposta suficiente. |
| SDK adversarial | Query sem restrição original, alvo/data divergentes e seleção inventada são rejeitados antes de publicar fato incompatível. |
| Datas/referentes | ISO e datas por extenso equivalentes; “mesmo terminal” após compactação mantém o ID; ID de múltiplos segmentos e TXN chegam à tool correta. |
| Unidade/payload externo | EUR solicitado + USD observado, amount diferente/inválido, unidade trocada e valores ausentes não viram cotação EUR válida. |
| Oráculo | Contradição de quem determina limite/taxa falha mesmo com número correto; extrato íntegro com ressalva passa; prosa não verificável não ganha aprovação. |

## Tarefas e ownership

### 1. Integridade operacional e unidades — agente A

Arquivos: novo `app/security/request_constraints.py`; `app/agents/customer_support_agent.py`; `app/tools/customer_tools.py`; somente funções cambiais em `app/tools/search_tools.py`; testes novos próprios.

- [x] Escrever regressões para alvo alterado pelo SDK, IDs multi-segmento, transação explícita, referência persistida, datas PT/EN/ambíguas e par/amount cambial divergente; confirmar RED.
- [x] Implementar os contratos acima sem alterar auth/consentimento/tenant existentes.
- [x] Executar testes próprios e regressões antigas; reportar mudanças de contrato sem editar expectativas compartilhadas para ocultar falha.

### 2. Answerability relacional — agente B

Arquivos: `app/rag/`; `app/agents/knowledge_agent.py`; catálogo de fatos em `data/`; somente função `search_getnet_knowledge`/seu schema em `app/tools/search_tools.py` se necessário e coordenado; testes novos próprios.

- [x] Escrever testes com corpus sintético: cortesia/paráfrase/contexto, cash ausente, sujeito trocado, unidade, negação/ressalva, dupla exigência e span adulterado; confirmar RED.
- [x] Implementar requisitos/catálogo/gate/renderer e verificar contra pedido original; não ampliar listas com palavras do conjunto reservado.
- [x] Executar testes próprios e regressões; documentar linguagem/cobertura delimitadas e evidência desconhecida.

### 3. Integração, router, oráculo e evidências — integrador

Arquivos: `app/agents/orchestrator.py`, `router_agent.py`; `app/evals/evaluator.py`; schemas/config somente se necessário; scripts/README/evidências/WORKLOG.

- [x] Regressões de roteamento PT/EN para transação, IDs compostos e clima por formulações diferentes; depois corrigir seleção com fallback apenas sem chave.
- [x] Passar mensagem original a Knowledge e validar integrações sem perder requisitos de multi-intenção.
- [x] Regressão da contradição de autoridade; reforçar juiz limitado e referência por relação, mantendo limites explícitos.
- [ ] Suíte completa, checker, harness/regressões conhecidas, imagem Python3.11 e smoke com image ID/SHA; Gemini opt-in quando o provedor permitir.
- [ ] Congelar candidato, pedir auditoria nova com amostra inédita, conferir CI no branch; main somente se 95+ sustentado.

## Revisão do plano

Todos os achados finais da auditoria V3 estão ligados a contrato/teste acima. Nenhum novo catálogo pode usar perguntas/case IDs como lookup. A meta não determina a nota. O roteiro de vídeo existente continua roteiro; não contabilizar vídeo como entregue até existir e ser inspecionado.
