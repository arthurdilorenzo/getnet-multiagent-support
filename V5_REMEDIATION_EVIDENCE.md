# V5 — evidências e limites da implementação

2026-10-23. Baseline `a79c52c`:85,15/100 nominais na auditoria independente, sem autorização de main. A [SPEC_V5](SPEC_V5.md) precedeu o código; o parecer externo ao produto compara opções nativa/local/offline e rejeita tratar hashing ou catálogo como compreensão universal.

## Execução com agentes

- Frente semântica: candidatos separados da autorização, `SemanticEvidenceAuthorizer` com análise e revisão completas, unidades/spans enumerados pelo servidor, validação estrutural rígida e renderer extrativo. Handoff:146 testes focados, incluindo41 novos. Não alterou fontes Markdown ou labels.
- Frente operacional/SDK: intenção compartilhada entre router/suporte, POS/TXN compostos na memória, referências após compactação, datas e ausência de TXN coerentes; timeout30s limitado ao deadline60s, zero retries e telemetria segura. Handoff:107 testes focados,18 novos.
- Integrador: schemas/status/mode, agregação conservadora, métricas por modo e avaliação parcial separando consistência de cumprimento. RED→GREEN cobriu também snippets web que antes eram marcados como resposta completa. Documentação/scripts e recibos preservam diferenças entre simulação, componente nativo e chat completo.

## Diagnóstico real de desenvolvimento

Dois casos de classe conhecida foram executados com Gemini3.1-flash-lite, somente no componente semântico, antes do freeze e com checkout alterado. Quatro gerações concluíram; nenhuma504 nessa execução. A pergunta de antecipação antes recusada recebeu decisão `complete` com extratos; a pergunta sobre senha/sessões ficou `abstained`, sem inventar política de sessão. Durações aproximadas por caso:11,91s e10,57s.

Esse piloto não é holdout, não mede chat/routing nem comprova o candidato final. Não atribuir a ausência de timeout isoladamente ao novo budget/thinking: houve várias mudanças e a amostra é pequena. Os recibos finais devem apontar SHA limpo e ficar fora do produto, em `../getnet_v5_evaluation/`.

## Limites e decisões explícitas

- Não há encoder neural de embeddings instalado. A interpretação semântica nativa usa Gemini sobre candidatos BM25/hash/RRF; duas passagens do mesmo modelo não são revisão independente e podem compartilhar erro.
- IDs/hashes/offsets e partição literal provam integridade, sem garantir interpretação ou verdade externa do resumo curado. Não publicar prosa factual livre do modelo.
- Sem chave, a gramática é composicional e delimitada; não promete paridade com Gemini. Cláusula/condição fora da linguagem não pode receber `complete`. Snippets de busca geral são apresentados como resposta parcial.
- Não há ledger semântico completo entre especialistas. Perguntas que combinam documentação e suporte podem resultar em parcial conservador. Manter essa limitação visível é preferível a declarar completude sem prova.
- O loop nativo de suporte não foi abreviado após a primeira tool: falta critério confiável para excluir necessidades dependentes. Isso preserva o protocolo mas pode exigir outra geração e aumentar latência.
- Casos48/76 anteriores agora são regressões conhecidas. Métricas do harness13 e testes SDK simulados não provam generalização real. Labels independentes por agente não são validação humana ou nota oficial.
- Main permanece condicionado a auditoria independente sustentando95+, além de CI verde; quantidade de testes e execução deste plano não satisfazem sozinhas essa condição.
