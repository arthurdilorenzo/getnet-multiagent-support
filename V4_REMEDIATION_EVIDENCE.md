# Ciclo V4 — integridade do pedido e evidência relacional

Data: 2026-10-23. Baseline: `e7a0f10`, auditoria independente **78,30/100 nominais**. Plano anterior à implementação: [SPEC_V4.md](SPEC_V4.md). Este registro não atribui nova nota nem autoriza publicar main.

## Frentes e contratos

1. Agente de integridade operacional: `RequestConstraints` imutável extraído da mensagem original, vinculação de POS/data/TXN, referência persistida e validação da identidade/unidade cambial. 33 testes novos próprios confirmaram RED→GREEN; 194 testes direcionados passaram no handoff.
2. Agente RAG/Knowledge: catálogo de 37 fatos em 13 chunks, sujeito/relação/valor/unidade/escopo, verificação de offsets/hash e cobertura agregada entre chunks. 32 testes próprios; 99 testes focados passaram no handoff. Revisão final corrigiu transferência de atributo entre sujeitos e polaridade entre cláusulas.
3. Integrador: roteamento de IDs compostos/TXN e clima, preservação da mensagem original na delegação, juiz conservador para referências independentes e recibo Docker com identidade da imagem/revisão. Testes de regressão confirmaram a falha do juiz antes da correção.
4. Autor independente: nova amostra reservada de 48 casos, 24 PT/24 EN, fora deste repositório. SHA-256 do dataset `b92d668728ccb54d5a786182df49b74e8b3a1aa8e19f973a5106491436ff94d2`. Implementadores não receberam perguntas/labels antes de congelar o candidato. Os 76 casos do ciclo anterior são regressão conhecida, não novo holdout.

## Limites preservados

- BM25/expansão léxica, hashing/âncoras e RRF continuam sem encoder neural. A gramática relacional é local e delimitada; não prova entailment universal ou compreensão de qualquer qualificador. Países arbitrários sem construção reconhecida, múltiplas cidades/pares e lógica contratual aberta permanecem limites.
- Spans/hashes comprovam identidade do resumo local curado, não autenticidade ou verdade externa. Nenhum Markdown fonte foi alterado para adequar o novo catálogo às perguntas.
- Datas relativas não se tornam automaticamente consultas atuais; snapshots operacionais continuam identificados como históricos. Ações POS continuam propostas e recibos simulados, sem hardware real.
- O juiz limitado pode rejeitar uma paráfrase correta da referência. A comparação Clássica/Smart tem texto fonte e referência abreviada diferentes; exige revisão semântica independente. Não alterar label para obter 100% local.
- Labels independentes são de agente, sem validação humana. A nova amostra tampouco representa todo o domínio ou a nota oficial da Getnet.
- O smoke Gemini anterior foi parcial por falhas do provedor; não contabilizar como teste ao vivo integralmente aprovado. Qualquer nova execução deve ter recibo próprio.

## Verificação e decisão de publicação

Recibos finais de suíte/harness, Docker/Gemini, SHA exato, Actions e auditoria serão registrados fora do produto em `../getnet_v4_evaluation/` e no WORKLOG. O branch de trabalho permanece `codex/getnet-v4-remediation`. `main` só pode receber candidato que sustente 95+ em auditoria independente, além de CI verde. Testes locais, quantidade de testes ou checklist completo não substituem esse gate.
