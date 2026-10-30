# Getnet Multi-Agent Support

FastAPI + Gemini Developer API, RAG neural/BM25, SQLite, ferramentas de suporte e confirmação de ações em duas fases.

## Executar

Use Python 3.11+ e as dependências fixadas em `requirements-lock.txt`. Configure `GEMINI_API_KEY` do projeto gratuito existente, `GEMINI_MODEL=gemini-3.5-flash-lite` e, para consultas neurais inéditas, `GETNET_LIVE_EMBEDDINGS=1`. Não há fallback para provedor pago. Em produção forneça `AUTH_API_KEY` e `JWT_SECRET` independentes e aleatórios (32+ caracteres); `APP_ENV=production` força autenticação estrita.

```sh
python -m pip install -r requirements-lock.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Acesse `/` para a UI, `/docs` para os contratos e `/api/health` para saúde. Sem chave, o chat informa HTTP503; não simula atendimento nativo. Docker: configure os mesmos segredos externamente e use `docker compose up --build`. A imagem roda como UID10001, filesystem read-only e estado no volume `/app/state`. A verificação local de container exige Docker daemon disponível.

## Arquitetura e comportamento

`POST /api/chat` autentica o titular e chama o núcleo em `app/v2/`: Router propõe necessidades/etapas; Support executa pelo registry; Knowledge recupera fontes e compõe uma resposta validada. As responsabilidades compartilham um modelo configurado e não dependem de gramáticas lexicais por cenário. `/api/v2/chat` é alias do mesmo runtime.

O modelo recebe um transporte plano com enum de blocos; o servidor converte para tipos internos e confere cobertura, origem, recurso, campo, unidade e disponibilidade. Campos privados e valores públicos estruturados são renderizados pelo servidor. Citação válida não prova, sozinha, fidelidade semântica de prosa documental.

RAG: corpus local com URLs/hashes → índice/cache neural compatível (`gemini-embedding-001`,384D) + BM25 → trechos recuperados → composição. Sem vetor neural compatível, a recuperação declara sparse_only; não mistura espaços. Corpus curado e métricas de regressão não equivalem a avaliação humana independente. Hit@K e MRR existentes medem o denominador declarado; Hit@K não é recall de múltiplos relevantes.

Tools: perfil, diagnóstico de terminal, recebíveis, tickets, clima Open-Meteo, câmbio AwesomeAPI/Frankfurter e snippets web. Preserve data/fonte/unidade e distinga cotação observada de referência diária. Fonte indisponível produz limitação explícita, sem número inventado.

## Ações e atendimento humano

`execute_terminal_remote_action` apenas cria proposta vinculada ao titular/sessão, com hash e expiração. Requer `terminal:action_own`. A confirmação usa `POST /api/actions/{proposal_id}/confirm` com JWT atual do titular e hash esperado; repetição retorna o mesmo recibo. A execução POS é simulada, não integração com equipamento físico.

HITL: operador autenticado reivindica/responde/resolve ticket pelos endpoints `/api/hitl/tickets/*`; `resume` retoma o pedido congelado com observação atribuída. Resposta humana não concede scopes nem confirma ação. Restart/replay não deve gerar nova chamada, ação ou publicação.

## Testes e evidências

```sh
python scripts/run_offline_tests.py -q
python scripts/check_no_print.py app
```

Execute testes com rede do modelo bloqueada, dotenv desabilitado e DB/cache sintéticos isolados via `scripts/run_offline_tests.py`. Testes legados preservam cobertura de componentes históricos; não representam, automaticamente, a qualidade semântica do caminho atual. `/api/evals/run` retorna 409 para impedir que uma avaliação legada seja apresentada como medição do runtime integrado.

Os recibos de execução nativa ao vivo no Gemini Free Tier (requisições HTTP, consumo de tokens e respostas estruturadas do runtime V2) estão versionados em `docs/eval_results/v2_live_integration/`.

## Demonstração curta

1. Inicie o app, autentique um lojista e pergunte sobre Crediário: compare afirmações e ressalvas com a evidência.
2. Consulte perfil/terminal: confira que os dados pertencem ao titular e que informação ausente permanece desconhecida.
3. Peça revisão humana, responda como operador e retome; repita resume e confira idempotência.
4. Solicite ação: confira proposta sem execução, confirme como titular e repita para obter o mesmo recibo simulado.
5. Consulte clima/câmbio/web; confira fonte, referência temporal e ausência de valores inventados.

O walkthrough técnico audiovisual atualizado do Núcleo V2 (`docs/walkthrough/Getnet_Walkthrough_Tecnico.mp4`, acompanhado de `narration.txt`, `subtitles.srt`, `contact-sheet.jpg` e `verification.json`) apresenta os contratos, trechos reais de código e evidências da entrega atual.
