# Getnet Multi-Agent Support System

Projeto de demonstração do desafio Getnet, em Python 3.11+, FastAPI, Google GenAI SDK e SQLite. Inclui interface de conversa, roteamento entre especialistas, recuperação documental, ferramentas operacionais simuladas e fila local de atendimento humano.

O sistema consulta dados públicos externos quando a rede está disponível. Cadastros, vendas, telemetria e comandos de maquininhas usam `data/mock_customers.json`; não existe conexão com sistemas internos ou terminais da Getnet. A fila HITL e o histórico são persistidos de fato no SQLite, mas isso não representa atendimento por uma equipe da Getnet.

## Arquitetura e modos de execução

```text
HTTP → autenticação/tenant → guardrails → memória SQLite
     → RouterAgent → plano validado (RoutingPlan)
     → supervisor com fila e limite de iterações
         ├─ CustomerSupportAgent → ToolRegistry → dados mock / escrita SQLite
         ├─ KnowledgeAgent       → retrieval local → trechos e citações
         ├─ Web Search           → APIs públicas / busca de snippets
         └─ EscalationAgent      → ticket SQLite → resposta de operador
     → mascaramento de PII → persistência → resposta e trace
```

| Componente | Com `GEMINI_API_KEY` | Sem chave |
| --- | --- | --- |
| Router | Gemini solicita delegações por function calling; nomes e argumentos são validados e convertidos em `RoutingPlan`. | Regras léxicas determinísticas produzem o plano. |
| Suporte operacional | Gemini recebe schemas das tools permitidas; o aplicativo executa cada call no registry, devolve `FunctionResponse` ao modelo e permite rodadas dependentes até o limite. | Um extrator de intenção/slots escolhe as mesmas tools registradas. |
| Knowledge / Web | Seleção de ferramentas por regras e extração de slots; Gemini pode redigir a resposta a partir dos resultados. | Mesma seleção, com composição extrativa/estruturada dos resultados. |
| Escalação | Enfileiramento local por ferramenta e supervisão de resultados. | Mesmo caminho local. |

A execução automática de tools pelo SDK fica desativada: as chamadas de suporte passam pelo `ToolRegistry`, que valida schemas, permite apenas ferramentas registradas e vincula identidade e sessão no servidor. Router e suporte não caem silenciosamente no planejador léxico após erro do Gemini; falhas de protocolo/provedor têm resultado controlado. Knowledge pode voltar à síntese local se sua redação Gemini falhar.

O supervisor executa até `MAX_ORCHESTRATION_ITERATIONS` passos. Ao observar `requires_escalation` no diagnóstico de suporte, pode acrescentar uma etapa humana. Esse replanejamento é uma política explícita de código; não é um planner geral autônomo. Chamadas de suporte também têm limites de turnos e de quantidade de tools.

## Retrieval local: o pipeline implementado

O código está em `app/rag/`. O nome legado `BilingualDenseEmbedder` é mantido por compatibilidade; **não há encoder neural, sentence-transformers ou chamada à API de embeddings** neste caminho. `GEMINI_EMBEDDING_MODEL` não é consumido pelo retriever.

1. O texto indexado repete o título duas vezes e acrescenta o conteúdo. A normalização remove acentos/pontuação e stopwords.
2. O **Okapi BM25**, com `k1=1.5` e `b=0.75`, usa expansão léxica manual EN→PT da consulta.
3. Uma **projeção por hashing em 384 coordenadas** combina unigramas, prefixos de cinco caracteres, bigramas, 4-gramas de caracteres e âncoras conceituais PT/EN escritas à mão. O vetor é normalizado em L2; a comparação usa cosseno limitado a `[0, 1]`.
4. Cada documento recebe um score heurístico: `0.5 × BM25_normalizado_por_cobertura + 0.5 × cosseno`. O **Reciprocal Rank Fusion**, com `k=20`, combina as posições dos dois rankings e serve de **desempate**; o score heurístico é a ordenação principal.
5. Resultados abaixo de `RAG_MIN_RELEVANCE_THRESHOLD=0.26` são descartados. Se nenhum passar, o retorno tem `abstained=true`, sem trechos/citações. Cada resultado aceito carrega URL, título, arquivo, `chunk_id`, excerto e score.

O score não é probabilidade de correção. As listas bilíngues são específicas do domínio e os buckets podem colidir. A escolha mantém dependências e execução local pequenas, com generalização semântica menor que a de um encoder treinado. O limiar não é prova de abstention correta para qualquer pergunta.

### Corpus, proveniência e reprodução

`data/getnet_corpus.json` reúne duas origens distintas:

- `verified_markdown_mirror`: seções dos seis arquivos `data/getnet_*.md`, curados localmente com URLs de referência. Esse rótulo legado não significa transcrição literal nem validação independente de cada afirmação.
- `live_web_scrape`: trechos de páginas públicas capturados por HTTP. `source_url` registra a URL final, `secondary_url` a URL solicitada e `scraped_at` a data da captura.

As URLs configuradas incluem catálogo de maquininhas, Crediário, Pix, conta digital, link de pagamento, dúvidas e página global em inglês. `canonical_sources` é a lista de URLs de entrada, não uma garantia de que todas responderam ou geraram chunks. O snapshot é datado; preços, regras e disponibilidade devem ser conferidos na fonte antes de uso operacional.

A mesma higiene determinística é aplicada a Markdown, HTML novo e snapshot existente. Ela remove formulários, menus, contatos públicos, cookies e boilerplate conhecido, preservando instruções e fatos de produto. Chaves Pix do tipo CPF/CNPJ e passos de menus do terminal são conteúdo útil, não formulários do site. O filtro é heurístico e precisa de revisão se o HTML mudar.

Cada `content_hash` contém o SHA-256 completo do conteúdo higienizado; o identificador usa um prefixo desse hash. A limpeza do snapshot preserva URLs e datas de captura. Os vetores são regenerados a cada indexação e persistidos em `data/vector_index.json`; esse arquivo é um artefato derivado, sem confiar em cache anterior.

Para reproduzir a limpeza do snapshot versionado e seu índice, sem HTTP:

```bash
python -m app.rag.ingestor --clean-existing
python -c 'from app.rag.retriever import retriever'
pytest -q tests/test_corpus_hygiene.py
```

A limpeza repetida é idempotente. Para reconstruir **somente dos mirrors locais**, use `python -m app.rag.ingestor`; isso substitui o snapshot e remove os trechos web anteriores. Para uma nova captura, use `python -m app.rag.ingestor --live` e depois regenere o índice com o comando acima. A captura depende de rede/layout e registra avisos ao pular URLs; não é byte a byte reproduzível nem uma validação editorial das fontes.

## Ferramentas, persistência e integrações

O `ToolRegistry` registra ferramentas de vendas, perfil, diagnóstico, ação remota simulada, criação/consulta de ticket, retrieval, clima, câmbio e busca geral. Schemas Pydantic, controle de identidade, erros estruturados e timeout cercam sua execução.

- **Operações simuladas:** liquidação e diagnóstico leem clientes mock; a escolha de terminal considera ID/modelo/sinais de falha. `execute_terminal_remote_action` grava uma intenção idempotente no SQLite. Mesmo quando o payload usa `SCHEDULED_ON_TERMINAL`, nenhum comando é enviado a hardware real.
- **APIs públicas:** clima usa geocodificação e previsão Open-Meteo, interpretando códigos WMO; câmbio tenta AwesomeAPI e, em falha, Frankfurter. Retries e circuit breakers tratam indisponibilidade; erros não são substituídos por números inventados.
- **Busca geral:** consulta snippets da Wikipedia PT/EN e tenta DuckDuckGo HTML como alternativa. Não lê integralmente cada fonte; resultados e URLs de busca têm limitações de proveniência.
- **Memória:** SQLite guarda sessões, mensagens, resumo textual de turnos antigos, tickets, ações e idempotência. O resumo é uma compactação textual local, não um modelo de memória semântica. A janela recente é configurável.
- **HITL:** ticket começa `OPEN`; o gestor também suporta `CLAIMED`, `RESPONDED` e `RESOLVED`. A API de reply grava a resposta humana, acrescenta mensagem à sessão e retorna seu estado para `active`; o próximo turno carrega essa resposta. Telegram/webhook são opcionais e dependem de configuração. Falha de notificação não desfaz o ticket nem comprova que um humano o recebeu. Não há SLA de atendimento.

O timeout do registry limita a espera pelo resultado; não interrompe à força uma thread que já iniciou o handler. As escritas locais não constituem uma transação distribuída com terminais ou serviços externos.

## Executar localmente

```bash
git clone https://github.com/arthurdilorenzo/getnet-multiagent-support.git
cd getnet-multiagent-support
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interface: `http://localhost:8000`; OpenAPI: `http://localhost:8000/docs`.

O exemplo inicia em `APP_ENV=development`, com segredos vazios e sem chave Gemini. Nesse modo demo, o chat aceita o `user_id` declarado e marca a identidade como `demo_unverified`; **não autentica o lojista**. Segredos necessários são efêmeros por processo. Use apenas dados fictícios. Clima/câmbio/busca podem consultar a rede mesmo sem Gemini; sem rede, retornam falha/abstenção.

Para usar Gemini, preencha `GEMINI_API_KEY` no `.env`; o modelo padrão é `gemini-2.5-flash`. Isso habilita o caminho de SDK descrito acima e envolve envio do contexto necessário ao provedor. Os testes offline não comprovam qualidade ou disponibilidade desse modelo ao vivo.

### Autenticação estrita e Docker

Para testar acesso autenticado, gere **dois segredos independentes** e coloque-os em `AUTH_API_KEY` e `JWT_SECRET` no `.env`:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Repita o comando para o segundo segredo. Produção exige ambos externos e fortes: pelo menos 32 caracteres, diversidade mínima de caracteres e ausência de marcadores previsíveis. `APP_ENV=production` força autenticação estrita e falha ao iniciar se a configuração for insegura. Configurar segredo JWT externo, emitir ou verificar um JWT também liga o modo estrito no processo; credenciais inválidas nunca caem em demo.

`X-API-Key` é uma credencial compartilhada de backend/bootstrap, capaz de emitir tokens e executar operações administrativas; não a distribua a lojistas. Emita JWTs de merchant vinculados ao `user_id` e JWTs pessoais de operador para atribuição das respostas HITL. A API key administrativa atribui ações ao principal `operator_system`.

Exemplo com `AUTH_API_KEY` e `MERCHANT_TOKEN` definidos no ambiente do shell:

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "X-API-Key: $AUTH_API_KEY" -H 'Content-Type: application/json' \
  -d '{"user_id":"cliente1988","role":"merchant"}'

# Copie access_token para MERCHANT_TOKEN antes desta chamada.
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer $MERCHANT_TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_id":"cliente1988","session_id":"minha-sessao","message":"Quando caem minhas vendas?"}'
```

A UI atual demonstra o chat sem inserir tokens; no modo estrito, use um cliente HTTP autenticado. Uma sessão pertence ao usuário persistido no SQLite; não basta conhecer ou fabricar um prefixo de `session_id`.

Com os dois segredos no `.env`, execute:

```bash
docker compose up --build -d
```

O Compose força produção e autenticação estrita. A imagem tem build em dois estágios, roda como `appuser` (UID 10001), copia aplicação/dados e não copia `.env` nem fontes de testes. `requirements.txt` fixa dependências diretas; `requirements-lock.txt` registra a resolução completa, mas o build/CI atual instala `requirements.txt`, portanto não impõe todas as versões transitivas do lock. O volume `getnet_sqlite_state` persiste `/app/data`, incluindo o banco; volumes já existentes também preservam seu corpus anterior e precisam de atualização explícita ao trocar o snapshot.

## API

| Método | Caminho | Acesso / comportamento |
| --- | --- | --- |
| POST | `/api/chat` | Chat, JWT ou API key no modo estrito; `user_id` e `session_id` vinculados no servidor. |
| POST | `/api/auth/token` | Operador autenticado; emite JWT merchant/operator. |
| GET / DELETE | `/api/sessions/{session_id}?user_id=...` | Histórico ou limpeza com verificação do proprietário. |
| GET | `/api/hitl/tickets` | Operador; filtros opcionais por status e cliente. |
| GET | `/api/hitl/tickets/{ticket_id}` | Operador; detalhe do chamado. |
| POST | `/api/hitl/tickets/{ticket_id}/reply` | Operador; identidade da resposta vem da credencial. |
| GET | `/api/metrics` | Operador; métricas do processo. |
| GET | `/api/evals/run` | Operador; executa harness e pode gravar sessões/tickets de avaliação. |
| GET | `/api/health` | Diagnóstico básico e configuração Gemini; não prova disponibilidade de todos os provedores. |
| GET | `/api/scenarios` | Exemplos de perguntas para a demonstração. |

Listagem/detalhe de HITL, métricas e evals rejeitam acesso anônimo com 401 e JWT merchant com 403. JWTs verificam assinatura HMAC-SHA256, emissor, expiração e formato dos claims. A proteção não substitui um provedor corporativo de identidade, revogação e gestão de credenciais.

## Testes, avaliação e evidência

Execute a mesma sequência principal do Actions:

```bash
python scripts/check_no_print.py app
pytest -v
python -m app.evals.evaluator
```

O checker analisa AST somente de arquivos `.py`: ignora comentários, docstrings e strings; rejeita chamadas diretas de console, multilinha e `builtins.print`, incluindo aliases importados. Não é análise de fluxo completa para reflexão/aliases dinâmicos.

Os testes exercitam retrieval, abstention/citações, higiene/idempotência/hash/índice, identidade e sessões, HITL, memória, ferramentas e falhas externas. Testes de function calling usam respostas simuladas do SDK para verificar o ciclo **call → registry → FunctionResponse → nova call/texto**, incluindo chamadas dependentes, argumentos inválidos e limites. Eles verificam o protocolo de integração; **não são um teste Gemini ao vivo**. Testes de cenário podem acessar APIs públicas e aceitam indisponibilidade honesta; a suíte não é inteiramente isolada da rede.

O harness usa `data/eval_dataset.json` (13 casos locais) e calcula routing accuracy, tool selection recall, score de grounding, pass rate e latência. O CLI retorna código diferente de zero quando routing ou tool recall ficam abaixo de 1.0, grounding médio abaixo de 0.90 ou o dataset está vazio. Esse gate pertence ao dataset local e não mede desempenho fora dele.

`GroundednessJudge` é uma **heurística lexical**, baseada em presença de termos esperados, URLs e ausência de algumas strings de fallback conhecidas. Não verifica entailment, correção de cada afirmação nem autenticidade externa das fontes. Os resultados não constituem avaliação semântica independente, nota V4 ou garantia contra alucinações. Latências sem Gemini não representam latência de uma chamada ao provedor; execuções do harness também podem depender de rede e estado SQLite prévio.

O plano, critérios e limites da correção estão em [SPEC.md](SPEC.md). Não há resultado de validação Gemini ao vivo publicado nesta entrega.

## Trade-offs e Limitações Declaradas

- **Segurança em camadas:** padrões de injection/XSS/SQLi, sanitização de conteúdo indireto, redaction de PII e headers reduzem riscos conhecidos; não garantem bloquear todos os ataques nem representam certificação PCI-DSS. Os filtros podem ter falsos positivos e negativos.
- **Persistência autocontida:** SQLite e locks locais facilitam execução e revisão em uma instância. PostgreSQL e um serviço compartilhado de limitação/cache seriam necessários para escalar o estado entre réplicas. Rate limiter, métricas e circuit breakers atuais residem em memória por processo e reiniciam com ele.
- **Observabilidade local:** logs JSON e trace IDs registram etapas e tempos. Router e suporte usam `usage_metadata` do SDK quando disponível; os outros registros podem estimar tokens por tamanho do texto. Custo usa coeficientes ilustrativos no código e não é conciliação de faturamento. Não há exportação de tracing distribuído configurada.
- **Qualidade documental:** limpeza e citações tornam a proveniência inspecionável; não substituem revisão editorial, atualização periódica ou confirmação de regras comerciais na fonte.
- **Ambiente de demonstração:** ainda seriam necessários identidade corporativa, revogação/rotação de tokens, aprovação e execução real das ações, gestão de dados e validação operacional antes de conectar clientes e dispositivos reais.
