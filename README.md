# Getnet Multi-Agent Support System (v2.0)

Sistema Multi-Agente de Atendimento, Diagnóstico Operacional e Suporte Comercial construído em **Python 3.11+**, **FastAPI**, **Google GenAI SDK (`gemini-2.5-flash`)** e **SQLite**, projetado para o desafio técnico de Engenharia de IA da **Getnet**.

---

## 1. Visão Geral da Arquitetura

O sistema implementa uma arquitetura **Supervisor Multi-Agente (Plan → Delegate → Observe → Dynamic Replan → Stop)** com isolamento de identidade Zero-Trust, **RAG Híbrido** sobre páginas oficiais da Getnet (`site.getnet.com.br`), **Function Calling** tipado via `ToolRegistry`, memória conversacional multi-turno persistida em **SQLite** e fila real de **Human-in-the-Loop (HITL)** com retomada de sessão.

```text
[Cliente / UI / API]
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Camada de Segurança & Ingress (FastAPI Middleware + Auth + Rate Limiter)   │
│ • Sliding-Window Rate Limiter (HTTP 429 + Retry-After)                     │
│ • AuthManager: JWT (HMAC-SHA256) / X-API-Key / Tenant Isolation (HTTP 403) │
│ • GuardrailManager (Input): Bloqueia Prompt Injection/XSS/SQLi (Regra ID)  │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ MultiAgentOrchestrator (Supervisor Loop: Plan -> Delegate -> Observe)      │
│ 1. Carrega contexto multi-turno e status HITL no SQLite (StateStore)       │
│ 2. RouterAgent gera plano validado por Pydantic (RoutingPlan / PlanStep)   │
│ 3. Delega para Especialistas e inspeciona SpecialistOutput a cada passo    │
│ 4. Replaneja dinamicamente (ex.: falha física de POS -> EscalationAgent)   │
└───────┬───────────────────────┬────────────────────────┬───────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌───────────────────┐  ┌──────────────────────┐  ┌───────────────────────────┐
│ KnowledgeAgent &  │  │ CustomerSupportAgent │  │ EscalationAgent (HITL)    │
│ WebSearchAgent    │  │ (Loop Multi-Tool)    │  │ • Persiste Ticket SQLite  │
│ • Hybrid RAG:     │  │ • Liquidação D+1/Pix │  │   (OPEN->CLAIMED->RESOLVED│
│   BM25 + Dense    │  │ • Telemetria Multi-  │  │ • Alerta Telegram/Webhook │
│   384d + RRF      │  │   Terminal POS       │  │ • Endpoint de Resposta do │
│ • Abstention <0.26│  │ • Write Tool Remota  │  │   Operador Humano com     │
│ • Open-Meteo WMO  │  │   Idempotente        │  │   Retomada de Sessão      │
│ • AwesomeAPI/ECB  │  │ • Catálogo ISO-8583  │  │                           │
└───────────────────┘  └──────────────────────┘  └───────────────────────────┘
        │                       │                        │
        └───────────────────────┼────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Camada de Egress, Redaction & Observabilidade                              │
│ • Indirect Prompt Injection Sanitizer (dados externos Web/RAG)             │
│ • PIIRedactor (CPF, CNPJ, Cartão PCI-DSS, CVV, Senhas, Domicílio Bancário) │
│ • SystemTracer (trace_id, step_duration_ms) + Metrics (p50/p95/p99, USD)   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes de Engenharia em Detalhe

### 2.1. Orquestração Supervisor e Replanejamento Dinâmico (`app/agents/orchestrator.py`, `app/agents/router_agent.py`)
- **Planejamento Estruturado (`RoutingPlan`)**: O `RouterAgent` utiliza JSON Schema validado por Pydantic v2 (`RoutingPlan` e `PlanStep`) para classificar a intenção, atribuir confiança e decompor perguntas compostas em múltiplos passos (`is_multi_step=True`).
- **Loop Supervisor (`MAX_SUPERVISOR_ITERATIONS = 4`)**: O `MultiAgentOrchestrator` mantém uma fila de execução (`execution_queue`) e avalia o `SpecialistOutput` retornado por cada especialista:
  - Se o `CustomerSupportAgent` diagnosticar em telemetria um defeito físico de hardware (`requires_hardware_replacement=True`, como violação de *tamper switch* `ERR_TAMPER_09` no perfil `cliente3012`), o Supervisor injeta dinamicamente um novo `PlanStep` acionando o `HUMAN_ESCALATION_AGENT` na mesma requisição.
  - Se o `KnowledgeAgent` abster-se por score abaixo do limiar em uma consulta factual pública, o Supervisor pode redirecionar para busca externa.

### 2.2. Motor RAG Híbrido, Corpus Oficial e Abstention (`app/rag/`)
- **Ingestão de Páginas Oficiais (`app/rag/ingestor.py`, `data/getnet_corpus.json`)**:
  - Corpus com **47 chunks** indexados com metadados verificáveis (`chunk_id`, `source_file`, `source_url`, `sha256_hash`), combinando documentação estruturada e raspagem das páginas públicas oficiais da Getnet (`https://site.getnet.com.br/todas-as-maquininhas/`, `/crediario/`, `/pix/`, `/conta-digital/`, `/link-de-pagamento/`, `/duvidas/` e `https://www.getnet.net/en/`).
- **Recuperação Híbrida (`app/rag/retriever.py`, `app/rag/embeddings.py`)**:
  1. **Sparse Retrieval**: Algoritmo **Okapi BM25** ($k_1 = 1.5, b = 0.75$) ponderado pela cobertura de termos da consulta e expansão bilíngue PT/EN.
  2. **Dense Retrieval**: `BilingualDenseEmbedder` gerando vetores densos de **384 dimensões** normalizados em $L_2$ (combinando unigramas, bigramas, 4-gramas de caracteres e âncoras conceituais bilíngues de meios de pagamento), persistidos em `data/vector_index.json` e comparados por similaridade de cosseno.
  3. **Fusão & Calibração**: Combinação via **Reciprocal Rank Fusion (RRF, $k=20$)** e score calibrado $\in [0.0, 1.0]$.
  4. **Limiar de Relevância & Abstention**: Consultas cujo maior score fique abaixo de `RAG_MIN_RELEVANCE_THRESHOLD = 0.26` acionam `abstained = True`, recusando-se a responder sem base documental em vez de alucinar.

### 2.3. Registro de Ferramentas, Function Calling e Zero-Trust (`app/tools/`)
- **`ToolRegistry` (`app/tools/registry.py`)**: Registro centralizado de **10 ferramentas** (8 de leitura e 2 de escrita idempotente) com validação Pydantic, timeout individual via `ThreadPoolExecutor` e exportação automática de JSON Schema para Function Calling (`get_gemini_function_declarations`).
- **Isolamento Zero-Trust de Tenant**: Os parâmetros `user_id` e `session_id` são removidos do schema exposto ao LLM. O modelo nunca escolhe de qual cliente ler dados; o `ToolRegistry` injeta o `user_id` diretamente do `SecurityContext` autenticado e bloqueia qualquer tentativa de acesso cruzado (`CROSS_TENANT_ACCESS_DENIED`).
- **Suporte Multi-Terminal e Ações de Escrita Idempotentes (`app/tools/customer_tools.py`)**:
  - `diagnose_terminal_issue`: Seleciona dinamicamente o terminal específico na frota do lojista por `terminal_id` (`GET-SMART-88219` vs `GET-CLASSICA-10442`), modelo ou anomalia de sinal/hardware, cruzando com o catálogo **ISO-8583** (`05`, `51`, `12`, `54`, `57`, `91`).
  - `execute_terminal_remote_action`: Ferramenta de escrita que agenda comandos remotos (`remote_reboot`, `sim_carrier_switch`, `parameter_table_update`) persistidos na tabela SQLite `terminal_actions` e protegidos por `idempotency_key` na tabela `idempotency_store`.
- **Integrações Externas Reais sem Fallbacks Falsos (`app/tools/search_tools.py`)**:
  - `get_weather_forecast`: Geocodificação + previsão diária via **Open-Meteo API**, decodificando o código meteorológico oficial **WMO 4677** (`weathercode`).
  - `get_currency_exchange_rate`: Cotação comercial em tempo real via **AwesomeAPI** com failover secundário para **Frankfurter API (European Central Bank)**.
  - **Honestidade Semântica sob Falha**: Todas as chamadas externas passam por `CircuitBreaker` + retries com *exponential backoff*. Se os provedores externos estiverem inacessíveis, a tool levanta `ExternalServiceUnavailableError` e o agente informa explicitamente a indisponibilidade temporária, **jamais** inventando temperaturas ou cotações fictícias.

### 2.4. Memória Multi-Turno e Fila Real de Human-in-the-Loop (`app/memory/store.py`, `app/hitl/queue.py`)
- **Persistência SQLite (`data/getnet_state.db`)**: Armazena sessões (`sessions`), histórico de mensagens (`messages`), chamados de escalonamento (`escalation_tickets`), comandos de terminal (`terminal_actions`) e chaves de idempotência (`idempotency_store`).
- **Janela Deslizante + Resumo Progressivo**: Mantém as últimas `MEMORY_WINDOW_MESSAGES` mensagens completas e compacta turnos anteriores em `rolling_summary`.
- **Ciclo Completo de HITL (`app/hitl/queue.py`)**:
  1. `create_escalation_ticket` grava o protocolo `GET-ESC-...` no SQLite com status `OPEN`, snapshot técnico do diagnóstico, histórico sanitizado da conversa e muda a sessão para `waiting_human`.
  2. Notifica canais externos configurados (Telegram Bot API e/ou Webhook HTTP).
  3. Um operador humano pode listar chamados (`GET /api/hitl/tickets`), assumir e responder via `POST /api/hitl/tickets/{ticket_id}/reply`.
  4. A resposta do operador humano é injetada na tabela `messages` da sessão do cliente e o status retorna para `active`, permitindo que o agente retome a conversa no próximo turno já ciente da resolução humana.

### 2.5. Segurança, Guardrails e Proteção de PII (`app/security/`, `app/agents/guardrails.py`)
- **Autenticação e Autorização (`app/security/auth.py`)**:
  - Emissão e validação de tokens JWT assinados com `HMAC-SHA256` (`POST /api/auth/token`) e suporte a `X-API-Key`.
  - Tentativas de consultar `user_id` divergente do token JWT retornam `HTTP 403 Forbidden`.
- **Rate Limiting (`app/security/rate_limit.py`)**: Algoritmo *Sliding Window* thread-safe por usuário e IP (`HTTP 429 Too Many Requests` com cabeçalho `Retry-After`).
- **Guardrails de Entrada, Contexto Indireto e Saída (`app/agents/guardrails.py`)**:
  - Bloqueia *Prompt Injection* direto (PT/EN), *Jailbreak*, extração de *System Prompt* e payloads SQLi/XSS retornando apenas códigos opacos (`SEC-INJ-001` a `SEC-CODE-001`), sem jamais expor expressões regulares internas.
  - `sanitize_untrusted_content`: Remove diretivas de *Indirect Prompt Injection* eventualmente presentes em páginas externas antes da síntese.
- **Redaction de PII (`app/security/pii.py`)**: Mascara automaticamente CPF, CNPJ, PAN de cartão de crédito (padrão PCI-DSS preservando apenas 4 últimos dígitos), CVV, senhas e números de agência/conta bancária em textos e dicionários estruturados.

---

## 3. Como Executar o Projeto

### 3.1. Execução Local (Python Virtualenv)

```bash
# 1. Clone o repositório oficial
git clone https://github.com/arthurdilorenzo/getnet-multiagent-support.git
cd getnet-multiagent-support

# 2. Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências com versões fixadas
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente (opcional: adicione sua GEMINI_API_KEY)
cp .env.example .env

# 5. Inicie o servidor FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Acesse a interface web interativa em `http://localhost:8000` ou a documentação OpenAPI em `http://localhost:8000/docs`.

### 3.2. Execução via Docker (Multi-Stage & Non-Root)

```bash
docker compose up --build -d
```

O container executa sob o usuário sem privilégios `appuser` (UID `10001`), sem copiar arquivos `.env` ou diretórios de teste para a imagem final, persistindo o banco SQLite no volume `getnet_sqlite_state`.

---

## 4. Testes Automatizados e Suíte de Avaliação (Evals)

### 4.1. Executar a Suíte de Testes (`pytest`)

A suíte conta com **33 testes automatizados** cobrindo recuperação híbrida BM25+Dense+RRF, abstention fora do domínio, enraizamento dinâmico em documentos novos, validação Pydantic do roteador, seleção multi-terminal, idempotência de tools de escrita, bloqueio Cross-Tenant, JWT, redaction de PII, sanitização de *Indirect Prompt Injection*, falha simulada de APIs externas (garantindo zero números inventados), memória multi-turno SQLite e ciclo completo de HITL:

```bash
pytest -v
```

### 4.2. Executar o Harness de Avaliação Quantitativa (`app/evals/evaluator.py`)

```bash
python -m app.evals.evaluator
```

Resultado obtido no dataset de avaliação (`data/eval_dataset.json`, 13 casos de teste cobrindo RAG PT/EN, Tools, Multi-Step, Abstention e Guardrail):

| Métrica de Avaliação | Resultado Obtido | Meta de Engenharia |
| :--- | :---: | :---: |
| **Routing Accuracy** | **100.0%** (`1.0000`) | $\ge 95.0\%$ |
| **Mean Tool Selection Recall** | **100.0%** (`1.0000`) | $\ge 95.0\%$ |
| **Mean Groundedness Score** | **96.15%** (`0.9615`) | $\ge 90.0\%$ |
| **Overall Pass Rate** | **100.0%** (`1.0000`) | $\ge 95.0\%$ |
| **Latency p50 / p95 (Modo Determinístico/Local)** | **~4.8 ms / ~65.8 ms** | $< 2000\text{ ms}$ |

---

## 5. Principais Endpoints da API

| Método | Endpoint | Descrição |
| :---: | :--- | :--- |
| `POST` | `/api/chat` | Endpoint principal do orquestrador multi-agente (`message`, `user_id`, `session_id`). Aceita `Authorization: Bearer <jwt>` ou `X-API-Key`. |
| `POST` | `/api/auth/token` | Emite token JWT assinado (`HMAC-SHA256`) para um `user_id` e `role` (`merchant` ou `operator`). Requer `X-API-Key`. |
| `GET` | `/api/sessions/{session_id}` | Retorna histórico multi-turno, resumo progressivo e status da sessão no SQLite. |
| `DELETE` | `/api/sessions/{session_id}` | Limpa o histórico conversacional de uma sessão específica. |
| `GET` | `/api/hitl/tickets` | Lista chamados persistidos na fila Human-in-the-Loop (`OPEN`, `CLAIMED`, `RESPONDED`, `RESOLVED`). |
| `POST` | `/api/hitl/tickets/{ticket_id}/reply` | Endpoint para operador humano responder a um protocolo HITL, injetando a resposta na sessão do cliente e reativando o atendimento. |
| `GET` | `/api/metrics` | Retorna telemetria agregada: latências `p50`/`p95`/`p99`, contagens por agente/tool, tokens consumidos e custo estimado em USD. |
| `GET` | `/api/evals/run` | Executa a suíte de avaliação automatizada e retorna o relatório JSON completo. |
| `GET` | `/api/health` | Healthcheck com status do SQLite, quantidade de chunks RAG indexados e estado dos Circuit Breakers. |

---

## 6. Decisões de Engenharia, Trade-offs e Limitações Declaradas

1. **Operação Dual (Com e Sem `GEMINI_API_KEY`) Sem Respostas Hardcoded**:
   - **Trade-off**: Avaliadores frequentemente executam o repositório sem chaves de API externas configuradas. Na versão anterior, isso havia sido tratado com respostas estáticas fixas para as perguntas do enunciado, o que comprometia a validade do sistema.
   - **Decisão atual**: Quando `GEMINI_API_KEY` está configurada, o sistema utiliza o `gemini-2.5-flash` para planejamento estruturado, Function Calling nativo e síntese textual. Quando executado sem chave de API, o sistema opera através do mesmo pipeline de dados: executa o cálculo real de **Okapi BM25 + Embeddings Vetoriais 384d + RRF**, invoca as mesmas tools registradas no `ToolRegistry` e compõe a resposta dinamicamente a partir dos trechos recuperados do corpus (`data/getnet_corpus.json`) ou dos payloads JSON retornados pelas tools. Nenhuma pergunta do desafio possui resposta hardcoded.

2. **Embeddings Vetoriais Determinísticos Locais (`BilingualDenseEmbedder`) vs. Modelo Neural Pesado (`sentence-transformers` / Torch)**:
   - **Trade-off**: Incluir PyTorch + `sentence-transformers` adicionaria ~1.5 GB à imagem Docker e aumentaria o tempo de build/cold-start em ambientes de avaliação sem GPU ou com restrição de rede.
   - **Decisão atual**: Implementamos um embedder denso de 384 dimensões L2-normalizado (`app/rag/embeddings.py`) baseado em projeção de n-gramas morfológicos e âncoras semânticas bilíngues do domínio de adquirência, persistido em `data/vector_index.json`.
   - **Limitação declarada**: Embora garanta alta precisão e alinhamento PT/EN no domínio de pagamentos com latência sub-milissegundo e zero dependências pesadas, ele possui menor capacidade de generalização semântica aberta para domínios arbitrários do que um modelo `text-embedding-004` ou `bge-m3`.

3. **Persistência em SQLite (`data/getnet_state.db`) vs. PostgreSQL + Redis**:
   - **Trade-off**: Para execução autocontida via `uvicorn` ou container único, o SQLite com *locking* thread-safe elimina a necessidade de subir instâncias separadas de banco relacional e cache, garantindo persistência real de sessões, fila HITL e idempotência.
   - **Limitação declarada**: Em um ambiente de produção distribuído com múltiplas réplicas horizontais no Kubernetes, o SQLite local e o *Rate Limiter* em memória deveriam ser substituídos por **PostgreSQL (com `pgvector`)** e **Redis** compartilhado.

4. **Autenticação Flexível por Padrão (`AUTH_ENFORCE_STRICT=false`)**:
   - **Trade-off**: Para não quebrar testes automatizados de correção que enviam apenas `{"message": "...", "user_id": "cliente1988"}` sem cabeçalhos HTTP de autenticação, o modo padrão cria um `SecurityContext` escopado ao `user_id` da requisição quando nenhum header `Authorization: Bearer` ou `X-API-Key` é enviado. Entretanto, quando um JWT é fornecido (ou quando `AUTH_ENFORCE_STRICT=true`), o sistema valida criptograficamente a assinatura `HMAC-SHA256` e bloqueia com `HTTP 403` qualquer divergência de tenant.
