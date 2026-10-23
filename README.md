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
| Knowledge / Web | Gemini seleciona ferramentas e argumentos por function calling; para RAG pode selecionar um ID de trecho já aprovado. A resposta factual usa extratos ou campos tipados, sem prosa livre do modelo. | Mesma seleção, com composição extrativa/estruturada dos resultados. |
| Escalação | Enfileiramento local por ferramenta e supervisão de resultados. | Mesmo caminho local. |

A execução automática de tools pelo SDK fica desativada: as chamadas de suporte passam pelo `ToolRegistry`, que valida schemas, permite apenas ferramentas registradas e vincula identidade e sessão no servidor. Router e suporte não caem silenciosamente no planejador léxico após erro do Gemini; falhas de protocolo/provedor têm resultado controlado. A resposta de suporte é renderizada de campos retornados pelas ferramentas; texto livre do provedor não confirma ações ou pagamentos. A seleção opcional de um trecho RAG pode voltar ao extrato local aprovado se o provedor falhar; isso não cria evidência nova.

O supervisor executa até `MAX_ORCHESTRATION_ITERATIONS` passos. Ao observar `requires_escalation` no diagnóstico de suporte, pode acrescentar uma etapa humana. Observações e resultados estruturados também alimentam etapas seguintes e um replanejamento limitado do router. A política offline é explícita; o sistema não é um planner irrestrito. Chamadas de suporte também têm limites de turnos e de quantidade de tools.

## Retrieval local: o pipeline implementado

O código está em `app/rag/`. O nome legado `BilingualDenseEmbedder` é mantido por compatibilidade; **não há encoder neural, sentence-transformers ou chamada à API de embeddings** neste caminho. `GEMINI_EMBEDDING_MODEL` não é consumido pelo retriever.

1. O texto indexado repete o título duas vezes e acrescenta o conteúdo. A normalização remove acentos/pontuação e stopwords.
2. O **Okapi BM25**, com `k1=1.5` e `b=0.75`, usa expansão léxica manual EN→PT da consulta.
3. Uma **projeção por hashing em 384 coordenadas** combina unigramas, prefixos de cinco caracteres, bigramas, 4-gramas de caracteres e âncoras conceituais PT/EN escritas à mão. O vetor é normalizado em L2; a comparação usa cosseno limitado a `[0, 1]`.
4. Cada documento recebe um score heurístico: `0.5 × BM25_normalizado_por_cobertura + 0.5 × cosseno`. O **Reciprocal Rank Fusion**, com `k=20`, combina as posições dos dois rankings e serve de **desempate**; o score heurístico é a ordenação principal.
5. Resultados abaixo de `RAG_MIN_RELEVANCE_THRESHOLD=0.26` são descartados. A gramática local PT/EN extrai requisitos de sujeito, relação, valor, unidade e qualificadores reconhecidos. O catálogo `data/knowledge_facts.json` liga fatos a parágrafos literais, `chunk_id`, offsets e SHA-256; hashes/spans divergentes não autorizam resposta. Consultas com corpus personalizado usam extração por padrões locais explícitos. A cobertura pode combinar chunks: todas as relações reconhecidas precisam de suporte no conjunto selecionado, caso contrário o retorno tem `abstained=true`, sem trechos/citações. Cabeçalhos e URLs não bastam. Cada resultado aceito carrega URL, título, arquivo, excerto, proveniência, score e cobertura dos requisitos. A resposta preserva os parágrafos aprovados e suas ressalvas.

O score não é probabilidade de correção. As listas bilíngues são específicas do domínio e os buckets podem colidir. O parser relacional reconhece construções delimitadas; uma condição não reconhecida ainda pode produzir erro. Spans e hashes verificam identidade do resumo curado local, sem provar sua verdade externa. A escolha mantém dependências e execução local pequenas, com generalização semântica menor que a de um encoder treinado. O limiar e o gate local não são prova de abstention correta para qualquer pergunta.

### Corpus, proveniência e reprodução

`data/getnet_corpus.json` contém o snapshot dos mirrors locais revistos contra páginas oficiais por agente em 23/10/2026. Capturas web antigas e afirmações sem suporte foram removidas nesta revisão. O ingestor suporta duas origens:

- `verified_markdown_mirror`: seções dos oito arquivos `data/getnet_*.md`, curados localmente com URLs, localização e método de revisão. São resumos locais; o rótulo legado não significa transcrição literal nem validação humana independente de cada afirmação.
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

A limpeza repetida é idempotente. Para reconstruir **somente dos mirrors locais**, use `python -m app.rag.ingestor`; isso substitui o snapshot e remove os trechos web anteriores. Para uma nova captura, use `python -m app.rag.ingestor --live` e depois regenere o índice com o comando acima. Alterar o conteúdo exige revisar os fatos e spans em `data/knowledge_facts.json`: o catálogo padrão não aprova automaticamente um novo texto, mesmo após reindexação. A captura depende de rede/layout e registra avisos ao pular URLs; não é byte a byte reproduzível nem uma validação editorial das fontes.

## Ferramentas, persistência e integrações

O `ToolRegistry` registra ferramentas de vendas, perfil, diagnóstico, ação remota simulada, criação/consulta de ticket, retrieval, clima, câmbio e busca geral. Schemas Pydantic, controle de identidade, erros estruturados e timeout cercam sua execução.

O servidor preserva o pedido original durante a delegação. IDs explícitos de POS/transação e datas reconhecidas limitam as chamadas de suporte; uma omissão inequívoca pode ser vinculada pelo servidor, enquanto divergências ou ambiguidade exigem esclarecimento. O parser aceita ISO, datas numéricas inequívocas e meses PT/EN por extenso com ano; datas relativas mantêm os avisos de snapshot histórico. Slots públicos reconhecidos também limitam cidade, horizonte e moedas selecionados pelo modelo. A cobertura dessas gramáticas é delimitada.

- **Operações simuladas:** liquidação e diagnóstico leem clientes mock; a escolha de terminal considera ID/modelo/sinais de falha. `execute_terminal_remote_action` cria uma proposta imutável com `PENDING_CONFIRMATION`, hash e expiração. Somente `POST /api/actions/{proposal_id}/confirm`, com JWT do titular e escopo `terminal:action_own`, registra um recibo simulado. Idempotência rejeita a mesma chave com payload diferente. Mesmo o recibo `SCHEDULED_ON_TERMINAL` não envia comando a hardware real.
- **APIs públicas:** clima usa geocodificação e previsão Open-Meteo, interpretando códigos WMO; câmbio tenta AwesomeAPI e, em falha, Frankfurter. Retries e circuit breakers tratam indisponibilidade; erros não são substituídos por números inventados. Extremos e precipitação ausentes continuam desconhecidos; a referência diária do Frankfurter é rotulada como referência, sem extremos intradiários fabricados.
- **Busca geral:** consulta snippets da Wikipedia PT/EN e tenta DuckDuckGo HTML como alternativa. Não lê integralmente cada fonte; resultados e URLs de busca têm limitações de proveniência.
- **Memória:** SQLite guarda sessões, mensagens, resumo textual de turnos antigos, tickets, ações e idempotência. Resumo, fatos e entidades extraídos por regras persistem além da janela e são consumidos como dados não confiáveis. Mudanças explícitas de produto substituem o tópico; turnos referenciais podem usar o anterior. Não é um modelo de memória semântica. A janela recente é configurável.
- **HITL:** ticket começa `OPEN`; o gestor também suporta `CLAIMED`, `RESPONDED` e `RESOLVED`. Claim/reply validam estado e operador atribuído em transação. Ticket resolvido não aceita alteração silenciosa. A sessão só volta a `active` quando não restam tickets pendentes; propostas/confirmações POS ficam bloqueadas durante a intervenção. O próximo turno carrega a resposta humana. Telegram/webhook são opcionais e dependem de configuração. Falha de notificação não desfaz o ticket nem comprova que um humano o recebeu. Não há SLA de atendimento.

O registry limita workers/fila e persiste operações consultáveis pelo titular. Seu timeout limita a espera; não interrompe à força uma thread que já iniciou o handler. Escrita em voo retorna `unknown`, `retryable=false` e `operation_id`, exigindo consulta/reconciliação antes de repetir. O SDK tem timeout HTTP, limite de saída, orçamento aproximado de tokens e deadline por requisição; esses limites não prometem cancelamento físico de efeitos externos. As escritas locais não constituem uma transação distribuída com terminais ou serviços externos.

## Executar localmente

```bash
git clone https://github.com/arthurdilorenzo/getnet-multiagent-support.git
cd getnet-multiagent-support
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interface: `http://localhost:8000`; OpenAPI: `http://localhost:8000/docs`.

O exemplo inicia em `APP_ENV=development`, com segredos vazios e sem chave Gemini. Nesse modo demo, o chat aceita o `user_id` declarado e marca a identidade como `demo_unverified`; **não autentica o lojista**. Segredos necessários são efêmeros por processo. Use apenas dados fictícios. Clima/câmbio/busca podem consultar a rede mesmo sem Gemini; sem rede, retornam falha/abstenção.

Para usar Gemini, preencha `GEMINI_API_KEY` no `.env`; o modelo padrão é `gemini-3.1-flash-lite` (conectividade verificada nesta revisão; disponibilidade depende da conta). Isso habilita o caminho de SDK descrito acima e envolve envio do contexto necessário ao provedor. Os testes offline não comprovam qualidade ou disponibilidade desse modelo ao vivo.

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

Na interface, cole somente o JWT do cliente no campo **Acesso**. Ele permanece em memória na página, sem armazenamento persistente; não cole a chave Gemini nem a API key administrativa. Propostas mostram um botão de confirmação com alvo e tipo. Uma sessão pertence ao usuário persistido no SQLite; não basta conhecer ou fabricar um prefixo de `session_id`.

Com os dois segredos no `.env`, execute:

```bash
docker compose up --build -d
```

O Compose força produção e autenticação estrita. A imagem tem build em dois estágios, roda como `appuser` (UID 10001), copia aplicação/dados e não copia `.env` nem fontes de testes. `requirements.txt` fixa dependências diretas; build e CI instalam a resolução transitiva de `requirements-lock.txt`. O volume `getnet_sqlite_state` persiste somente `/app/state` (banco/índice derivado); o corpus em `/app/data` acompanha a imagem. O Compose usa filesystem somente leitura e `/tmp` efêmero. Volumes antigos montados em `/app/data` exigem migração explícita do banco antes de adotar o novo layout.

## API

| Método | Caminho | Acesso / comportamento |
| --- | --- | --- |
| POST | `/api/chat` | Chat, JWT ou API key no modo estrito; `user_id` e `session_id` vinculados no servidor. |
| POST | `/api/auth/token` | Operador autenticado; emite JWT merchant/operator. |
| POST | `/api/actions/{proposal_id}/confirm` | JWT do titular + hash/CONFIRM; sem autoconfirmação pelo LLM. |
| GET | `/api/tool-operations/{operation_id}?user_id=...` | JWT do titular + escopos originais; reconciliação de resultado. |
| GET / DELETE | `/api/sessions/{session_id}?user_id=...` | Histórico ou limpeza com verificação do proprietário. |
| GET | `/api/hitl/tickets` | Operador; filtros opcionais por status e cliente. |
| GET | `/api/hitl/tickets/{ticket_id}` | Operador; detalhe do chamado. |
| POST | `/api/hitl/tickets/{ticket_id}/claim` | Operador; transição válida e atribuição autenticada. |
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
GEMINI_API_KEY='' python -m app.evals.evaluator
python scripts/docker_smoke.py --output /tmp/docker-smoke.json
```

O checker analisa AST somente de arquivos `.py`: ignora comentários, docstrings e strings; rejeita chamadas diretas de console, multilinha e `builtins.print`, incluindo aliases importados. Não é análise de fluxo completa para reflexão/aliases dinâmicos.

Os testes exercitam retrieval, abstention/citações, higiene/idempotência/hash/índice, identidade e sessões, HITL, memória, ferramentas e falhas externas. Testes de function calling usam respostas simuladas do SDK para verificar o ciclo **call → registry → FunctionResponse → nova call/texto**, incluindo chamadas dependentes, argumentos inválidos e limites. Eles verificam o protocolo de integração; **não são um teste Gemini ao vivo**. Testes de cenário podem acessar APIs públicas e aceitam indisponibilidade honesta; a suíte não é inteiramente isolada da rede.

O harness usa `data/eval_dataset.json` (13 regressões locais com labels revistos por agente) e calcula routing accuracy, recall/precision de ferramentas, extras perigosos, consistência de evidências, pass rate e latência. A revisão de labels removeu termos sem suporte (touchscreen/menu de link) e corrigiu referências de origem, preservando as perguntas. Essa revisão não transforma o conjunto em avaliação independente. O CLI retorna código diferente de zero quando routing, recall/precision ou seleção segura de ferramentas ficam abaixo de 1.0, consistência média ou pass rate abaixo de 0.90 ou o dataset está vazio. Esse gate pertence ao dataset local e não mede desempenho fora dele.

`GroundednessJudge` é um **validador limitado de consistência de evidências**. Rejeita números não suportados, conflitos explícitos de polaridade, extratos alterados e abstenções com alegações materiais. Quando existe referência independente, só certifica alegações extrativas dessa referência: paráfrases não verificáveis ficam reprovadas para revisão semântica, mesmo que possam estar corretas. Isso impede aprovar inversões de autoridade apenas por repetir termos e números. Uma URL ou termos esperados sozinhos não aprovam a resposta. Ainda não prova entailment universal, cobertura ou autenticidade externa; pode haver falsos positivos/negativos. Métricas automáticas não são nota V4 nem garantia contra alucinações.

A suíte comum desliga Gemini mesmo com `.env` configurado. A integração real é opt-in, usa dados sintéticos e estado temporário:

```bash
GETNET_LIVE_TESTS=1 python scripts/live_smoke.py --output /tmp/gemini-smoke.json
```

O teste envolve consumo do provedor; nunca coloque a chave na linha de comando ou nos artefatos. O smoke Docker constrói a imagem e verifica saúde, usuário não root, strict auth, chat, tenant e persistência após restart. O recibo registra SHA do checkout, indicador de alterações locais, ID imutável da imagem e revisão OCI. Reutilizar imagem com `--skip-build` exige revisão correspondente e checkout limpo. O Actions executa esse smoke e publica seu recibo como artefato. Uma avaliação semântica independente reservada, separada dos implementadores, é necessária além desses checks.

O plano atual e os critérios de aceite estão em [SPEC_V4.md](SPEC_V4.md); os ciclos anteriores estão em [SPEC_V3.md](SPEC_V3.md) e [SPEC.md](SPEC.md). O score 95+ depende da auditoria do candidato exato, sem inferência pelo número de testes. Não há nota oficial nova do avaliador Getnet.

## Trade-offs e Limitações Declaradas

- **Segurança em camadas:** padrões de injection/XSS/SQLi, sanitização de conteúdo indireto, redaction de PII e headers reduzem riscos conhecidos; não garantem bloquear todos os ataques nem representam certificação PCI-DSS. Os filtros podem ter falsos positivos e negativos.
- **Persistência autocontida:** SQLite e locks locais facilitam execução e revisão em uma instância. PostgreSQL e um serviço compartilhado de limitação/cache seriam necessários para escalar o estado entre réplicas. Rate limiter, métricas e circuit breakers atuais residem em memória por processo e reiniciam com ele.
- **Observabilidade local:** logs JSON e trace IDs registram etapas e tempos. Router e suporte usam `usage_metadata` do SDK quando disponível; os outros registros podem estimar tokens por tamanho do texto. Custo usa coeficientes ilustrativos no código e não é conciliação de faturamento. Não há exportação de tracing distribuído configurada.
- **Qualidade documental:** limpeza e citações tornam a proveniência inspecionável; não substituem revisão editorial, atualização periódica ou confirmação de regras comerciais na fonte.
- **Ambiente de demonstração:** ainda seriam necessários identidade corporativa, revogação/rotação de tokens, aprovação e execução real das ações, gestão de dados e validação operacional antes de conectar clientes e dispositivos reais.
