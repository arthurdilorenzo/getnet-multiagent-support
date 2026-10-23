## Retrieval local: o pipeline implementado

O código está em `app/rag/`. O nome legado `BilingualDenseEmbedder` é mantido por compatibilidade; **não há encoder neural, sentence-transformers ou chamada à API de embeddings** neste caminho. `GEMINI_EMBEDDING_MODEL` não é consumido pelo retriever.

1. O texto indexado repete o título duas vezes e acrescenta o conteúdo. A normalização remove acentos/pontuação e stopwords.
2. O **Okapi BM25**, com `k1=1.5` e `b=0.75`, usa expansão léxica manual EN→PT da consulta.
3. Uma **projeção por hashing em 384 coordenadas** combina unigramas, prefixos de cinco caracteres, bigramas, 4-gramas de caracteres e âncoras conceituais PT/EN escritas à mão. O vetor é normalizado em L2; a comparação usa cosseno limitado a `[0, 1]`.
4. Cada documento recebe um score heurístico: `0.5 × BM25_normalizado_por_cobertura + 0.5 × cosseno`. O **Reciprocal Rank Fusion**, com `k=20`, combina as posições dos dois rankings e serve de **desempate**; o score heurístico é a ordenação principal.
5. **Candidatos nativos:** `retrieve_candidates` entrega ranking sem gate lexical nem limiar de answerability, limitado a48chunks/48.000 caracteres. O corpus atual pequeno cabe integralmente. Candidatos não são respostas aprovadas. **Modo offline:** `retrieve_hybrid` mantém limiar0,26 e catálogo `data/knowledge_facts.json`, com sujeito/relação/valor/unidade, parágrafos, offsets e SHA-256. Corpus personalizado pode usar extração por padrões explícitos. A gramática composicional precisa consumir o pedido completo para declarar `complete`; formulação fora da linguagem pode retornar extrato parcial identificado ou abstenção.

No modo nativo, `SemanticEvidenceAuthorizer` recebe o original e unidades/trechos enumerados pelo servidor. A primeira geração classifica cada unidade e liga requisitos a spans; a segunda revê a pergunta inteira, condições, papéis, unidades e suficiência. Hash, IDs, offsets, quotes e presença de todas as unidades são validados localmente. Apenas parágrafos integrais aprovados chegam ao renderer, preservando ressalvas. Há no máximo duas gerações semânticas adicionais por atendimento documental. Contexto pode resolver referências, sem autorizar ações.

`semantic_mode` distingue `native_semantic` de `limited_offline`; `answer_status` distingue `complete`, `partial` e `abstained`. Uma resposta parcial mostra literalmente as partes não respondidas e preserva `abstained=true`. O orquestrador não promove etapas incompletas a cobertura global. Esses campos são metadados de execução, não certificação de verdade.

O score não é probabilidade de correção. As listas bilíngues são específicas do domínio e os buckets podem colidir. O parser offline é delimitado e não promete paridade com Gemini. A interpretação semântica nativa é feita pelo LLM, sem mudar a natureza do índice por hashing. Spans/hashes provam identidade do resumo local, não sua verdade externa. Duas passagens do mesmo modelo podem compartilhar erros; schemas válidos e partição completa do texto não provam que um qualificador foi entendido corretamente. Auditoria independente continua necessária.

