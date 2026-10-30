# Getnet — Walkthrough Técnico (Núcleo V2 Integrado)

Vídeo técnico gravado e validado sobre a arquitetura V2 integrada: **`Getnet_Walkthrough_Tecnico.mp4`** (**4min09,88s**, 1280×720, H.264/AAC, **3.759.169 bytes**, SHA-256 `bbf3ff11e72ddcc67cd19dcfea4febbae1348e7781b0d8ac8c6d744cd08fee56`).

## Conteúdo Entregue

- `Getnet_Walkthrough_Tecnico.mp4`: walkthrough de 11 cenas cobrindo o entrypoint unificado (`app/main.py`), contratos tipados (`app/v2/contracts.py`), RAG híbrido com embeddings neurais 384D (`gemini-embedding-001` + BM25), `ToolGateway` isolado por tenant, ação remota em duas fases, fila HITL com retomada ativa pelo agente (`resume`), recibos nativos do Gemini Free Tier e suíte offline de 1.820 testes.
- `narration.txt`, `subtitles.srt`: locução completa e legendas sincronizadas das 11 cenas.
- `contact-sheet.jpg`: prancha de contato com os 11 quadros extraídos do MP4 final.
- `verification.json`: recibo de verificação de streams, duração (249,88s), volume de áudio (-15,0 dB médio / -2,8 dB pico) e decodificação integral sem erros.
- `sources.json`: mapeamento de cenas, timestamps e arquivos-fonte do repositório.
- `render_walkthrough.py`, `verify_video.py`, `run_validation.py`: scripts locais reprodutíveis (Pillow + macOS `say` + FFmpeg, zero chamadas de rede).
- `manifest.sha256`: hashes SHA-256 de todos os artefatos deste diretório.

## Verificação de Integridade

Para auditar os hashes SHA-256 de todos os arquivos entregues neste diretório:

```sh
shasum -a 256 -c manifest.sha256
```

Para regerar o vídeo localmente em macOS (requer Pillow e FFmpeg):

```sh
python3 render_walkthrough.py
python3 verify_video.py
```
