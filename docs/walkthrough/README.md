# Getnet — walkthrough técnico portable

Vídeo final em português: **4min40,215s**,1280×720,H.264/AAC,**3.699.599bytes**. Narração sintética Luciana (macOS) explicitamente identificada; dados de clientes e ações de terminal são simulados. O vídeo apresenta cenas gráficas de arquitetura, código e recibos reais; não é uma gravação de interface ao vivo.

Baseline auditado: `d7e2125129d5806ee38730f12868aeaaba469112`. A cena V6 mostra o contrato proposto em23/10/2026, sem atribuir resultados ainda não demonstrados. A auditoria V5 produziu89,27/100 nominais, não uma nota oficial; não sustentou95+. O vídeo preserva essa conclusão e distingue484testes anteriores de198contratos locais novos. O caminho RAG é BM25+expansão léxica+hashing/ngramas/âncoras384d, com RRF como desempate; não há encoder neural local. O modo nativo interpreta/verifica o pedido original via Gemini.

## Conteúdo

- `Getnet_Walkthrough_Tecnico.mp4`: vídeo final; SHA-256 `207f9cd794bc36db0b5a908e40e2a381afeafae6e2013594219b7ea8c48b105f`.
- `narration.txt`, `subtitles.srt`: locução completa e legendas externas aproximadas.
- `contact-sheet.jpg`:11quadros extraídos do MP4 final.
- `verification.json`: duração, streams, áudio médio−15,1dB/pico−2,4dB e decodificação integral aprovada. Não houve escuta humana.
- `sources.json`: fontes/timestamps das cenas, resultado local e CI arquivada.
- `local-validation.json`: recibo original de198contratos aprovados,1warning,1,76s; rede bloqueada/dotenv suprimido.
- `snapshots/`: código exato do baseline usado nos três excertos, specsV3/V5, propostaV6, recibos agregados sanitizados e excertos documentais. `snapshots/provenance.json` registra origem/hash. Os arquivos são suficientes para renderizar sem Git, sem diretórios irmãos e sem o workspace original.
- `manifest.sha256`: hashes de todos os arquivos estáticos deste bundle, exceto o próprio manifesto.

A seleção de fontes foi explícita: código/specs, métricas agregadas e URLs públicas. Não inclui arquivos de configuração, credenciais, headers de sessão, perguntas de clientes ou payloads privados. Os recibos sanitizados mantêm apenas os campos necessários às alegações do vídeo. As strings que mencionam arquivos de origem em `sources.json` são referências históricas; o renderer lê somente os snapshots deste bundle.

## Reprodução local, sem rede

Requisitos para render: macOS com a voz **Luciana** já instalada,Python3 com **Pillow**,FFmpeg e ffprobe. Fontes locais Arial e Menlo precisam estar disponíveis nos caminhos padrão do macOS. FFmpeg/ffprobe são resolvidos pelo PATH (fallback Homebrew); se necessário,configure as variáveis `FFMPEG` e `FFPROBE` com os executáveis locais. Nenhuma instalação ou download é executado pelos scripts.

A partir desta pasta, independentemente de onde ela foi copiada:

```sh
python3 render_walkthrough.py --check-inputs
python3 render_walkthrough.py
python3 verify_video.py
shasum -a 256 -c manifest.sha256
```

A síntese usa `/usr/bin/say -v Luciana -r178`, seguida de `atempo=1.5` sem alteração de pitch. O serviço local de fala do macOS pode precisar de permissão fora do sandbox: na produção original,sem esse acesso,`say` retornava sucesso mas gerava apenas cabeçalho sem amostras;ffprobe detectou o problema. Não se usa serviço externo de voz. ÁudioAIFF,PNGs e segmentos são regenerados em `audio/`,`frames/`,`segments/` e não estão duplicados neste bundle.

AIFFs já existentes são reutilizados. Se editar a locução,remova os AIFFs gerados antes de renderizar de novo. A saída pode variar em bytes com outras versões de voz/FFmpeg/fontes; o manifesto é do vídeo entregue, não promessa de reprodução binária. `verify_video.py` exige180–300s,menos de25MB,streams de áudio/vídeo,áudio médio acima de−45dB e decode completo; extrai11quadros. A verificação não equivale a avaliação semântica do sistema.

## Repetir contratos do baseline (opcional)

A reprodução do vídeo não precisa do produto. Para repetir os testes,é necessário um checkout Git com o commit baseline e um Python com as dependências de teste já instaladas. Se a pasta estiver em `docs/walkthrough/` dentro desse checkout,o script detecta o repositório e usa sua `.venv` quando disponível:

```sh
python3 run_validation.py
```

Fora do checkout,declare as duas localizações:

```sh
python3 run_validation.py --repo /caminho/checkout --python /caminho/venv/bin/python
```

O script faz `git archive` do SHA congelado para uma pasta temporária local,suprime dotenv,bloqueia sockets e executa seis módulos de contratos:ações,HTTP,segurança,function calling eHITL. Remove a cópia temporária e grava `local-validation-rerun.json`,preservando o recibo original. Não altera o checkout,configurações ou banco de desenvolvimento,nem acessa Gemini ou hardware. A ausência do commit/dependências causa erro explícito,sem fazer fetch ou instalar pacotes.

Nenhum commit,push,publicação ou alteração de produto foi feito para gerar este bundle. Limites restantes:corpus datado/curado,estado local,ausência de hardware real e operação distribuída não comprovada. Uma gravação UI futura pode complementar esta entrega.
