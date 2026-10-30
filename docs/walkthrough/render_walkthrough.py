"""Render local reprodutível do Walkthrough V2: Pillow + macOS say + FFmpeg. Zero rede."""
import hashlib, json, os, pathlib, shutil, subprocess, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

O = pathlib.Path(__file__).resolve().parent
R = O.parents[1]
SHA = subprocess.check_output(['git', '-C', str(R), 'rev-parse', '--short', 'HEAD'], text=True).strip()
FF = os.environ.get('FFMPEG') or shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
FP = os.environ.get('FFPROBE') or shutil.which('ffprobe') or '/opt/homebrew/bin/ffprobe'
W, H = 1280, 720
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
MONO = '/System/Library/Fonts/Menlo.ttc'


def font(n, b=False, m=False):
    return ImageFont.truetype(MONO if m else BOLD if b else FONT, n)


def run(a):
    subprocess.run(a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def code(rel_path, start, end):
    s = (R / rel_path).read_text(encoding='utf-8').splitlines()
    return '\n'.join(f'{i:03}  {s[i-1]}' for i in range(start, min(end, len(s)) + 1))


scenes = [
    dict(
        title='Suporte Multiagente Getnet — Núcleo V2',
        tag='01 / ARQUITETURA INTEGRADA E ESCOPO',
        big='Runtime semântico unificado, RAG neural 384D e segurança em 2 fases',
        lines=[
            f'Walkthrough técnico atualizado • revisão {SHA} • 29/10/2026',
            'FastAPI + Gemini Developer API (gemini-3.5-flash-lite) no entrypoint principal.',
            'RAG híbrido com embeddings neurais 384D (gemini-embedding-001) + Okapi BM25.',
            'Fila HITL com retomada ativa da sessão, ação em 2 fases e 1.820 testes verdes.',
        ],
        snippet=code('app/main.py', 37, 51),
        source='app/main.py:37–51 • get_integrated_runtime()',
        voice='Este walkthrough apresenta a arquitetura integrada do sistema multiagente de suporte Getnet na versão dois. O ponto de entrada principal da aplicação opera diretamente sobre o runtime semântico unificado com o modelo Gemini três ponto cinco Flash Lite. O sistema combina recuperação híbrida com embeddings neurais de trezentas e oitenta e quatro dimensões, ferramentas operacionais isoladas por lojista, escalonamento humano com retomada ativa pelo agente e confirmação de ações críticas em duas fases.',
    ),
    dict(
        title='Entrypoint Único e Autenticação Estrita',
        tag='02 / SEGURANÇA E IDENTIDADE',
        big='JWT HS256, isolamento de tenant e verificação de escopos no servidor',
        lines=[
            'POST /api/chat valida rate limit, credencial JWT e escopo chat:write.',
            'assert_tenant_access impede que um lojista consulte outra conta.',
            'Em produção, segredos fracos são rejeitados no boot e modo demo é desativado.',
            'Dados sensíveis (CPF, CNPJ, cartões, e-mail) passam por redator de PII.',
        ],
        snippet=code('app/main.py', 150, 169),
        source='app/main.py:150–169 • chat_endpoint()',
        voice='Toda requisição ao endpoint principal de chat passa primeiro pelo controle de taxa de requisições e pela resolução estrita do contexto de segurança. O servidor verifica a assinatura do token JWT, exige o escopo de escrita de chat e valida o vínculo de propriedade do lojista antes de acionar o runtime. Em ambiente de produção, segredos fracos interrompem a inicialização e todo conteúdo sensível é mascarado contra vazamento de dados pessoais.',
    ),
    dict(
        title='Contratos Tipados e Decomposição Canônica',
        tag='03 / RACIOCÍNIO E COMPOSIÇÃO',
        big='Planejamento estruturado e validação rigorosa de blocos de resposta',
        lines=[
            'Router decompõe o pedido em requisitos canônicos documentais, operacionais ou humanos.',
            'TransportAnswerBlock valida discriminadores compatíveis com OpenAPI 3.0.',
            'O servidor confere cobertura de requisitos, citações, unidades e disponibilidade.',
            'Fatos ausentes produzem resposta parcial ou abstenção limpa, sem alucinação.',
        ],
        snippet=code('app/v2/contracts.py', 120, 136),
        source='app/v2/contracts.py:120–136 • TransportAnswerBlock',
        voice='Em vez de depender de gramáticas manuais por cenário, o núcleo dois utiliza contratos Pydantic estritos para planejamento e composição. O pedido do cliente é decomposto em necessidades explícitas, e cada bloco de resposta gerado pelo modelo precisa declarar seu tipo e apontar para evidências ou fatos verificados pelo servidor. Quando uma informação não existe nas fontes ou nas ferramentas, o validador rebaixa o bloco ou declara resposta parcial de forma transparente.',
    ),
    dict(
        title='RAG Híbrido com Embeddings Neurais Reais',
        tag='04 / CONTEXT, KNOWLEDGE & RAG',
        big='Vetores 384D (gemini-embedding-001) + Okapi BM25 + RRF',
        lines=[
            'Embeddings reais de 384 dimensões gerados pela API Gemini e persistidos em cache.',
            'Recuperação combina similaridade cosseno neural com escore léxico BM25.',
            'Proveniência por SHA-256, URL oficial e preservação de ressalvas contratuais.',
            'Métricas formais de retrieval aferidas em suíte: Hit@1, Hit@3, Hit@5 e MRR@5.',
        ],
        snippet=code('app/v2/knowledge.py', 35, 52),
        source='app/v2/knowledge.py:35–52 • KnowledgeService',
        voice='O módulo de conhecimento executa busca híbrida real combinando vetores densos de trezentas e oitenta e quatro dimensões do modelo Gemini Embedding com o ranking léxico BM vinte e cinco. Os trechos recuperados carregam a URL oficial da Getnet e hash criptográfico de integridade. A qualidade da recuperação é monitorada pelas métricas Hit at K e Mean Reciprocal Rank, garantindo que prazos, taxas e ressalvas do emissor cheguem íntegros à etapa de síntese.',
    ),
    dict(
        title='ToolGateway e Isolamento de Estado',
        tag='05 / TOOL ENGINEERING',
        big='O modelo nunca executa ferramentas diretamente nem escolhe o titular',
        lines=[
            'Oito ferramentas registradas: perfil, diagnóstico POS, liquidação, tickets, ação, clima, câmbio e web.',
            'IsolatedToolRegistry vincula todas as operações ao SQLite e ao tenant autenticado.',
            'Resolução simbólica de equipamentos: vincula exatamente o terminal retornado pela tool.',
            'Fatos operacionais preservam tipo de recurso, ponteiro de campo e unidade.',
        ],
        snippet=code('app/v2/adapters.py', 142, 160),
        source='app/v2/adapters.py:142–160 • ToolGateway',
        voice='As chamadas de ferramentas propostas pelo planejador atravessam obrigatoriamente o Tool Gateway. O modelo nunca injeta o identificador do cliente por conta própria: a identidade vem sempre da sessão autenticada no servidor. Cada leitura de perfil, liquidação financeira ou diagnóstico de maquininha é transformada em fatos canônicos tipados com ponteiro de campo exato, impedindo que dados de um terminal sejam atribuídos a outro.',
    ),
    dict(
        title='Escrita Segura em Duas Fases no Terminal POS',
        tag='06 / ACTION POLICY & IDEMPOTÊNCIA',
        big='Proposta imutável ≠ execução imediata',
        lines=[
            'Fase 1: execute_terminal_remote_action exige terminal:action_own e cria proposta pendente.',
            'Gera proposal_id único, confirmation_hash SHA-256 e janela de expiração.',
            'Fase 2: apenas POST /api/actions/{id}/confirm com JWT do titular efetiva a ação.',
            'Idempotência estrita: repetição devolve o mesmo recibo; payload divergente retorna 409.',
        ],
        snippet=code('app/security/action_policy.py', 96, 115),
        source='app/security/action_policy.py:96–115 • propose_action()',
        voice='Comandos remotos em maquininhas seguem um protocolo estrito de duas fases. Na primeira fase, a ferramenta apenas registra uma proposta pendente de confirmação com hash criptográfico e prazo de expiração, exigindo o escopo de ação do titular. Nenhuma alteração é executada pelo modelo sozinho. A efetivação só ocorre na segunda fase, quando o próprio lojista envia uma confirmação explícita via endpoint dedicado, com proteção completa de idempotência.',
    ),
    dict(
        title='Fila HITL e Retomada Ativa pelo Agente',
        tag='07 / HUMAN OVERSIGHT & RESUMPTION',
        big='Ticket OPEN → CLAIM → REPLY → RESUME com memória e escopos preservados',
        lines=[
            'Escalonamento grava pedido original e escopos do lojista na fila SQLite e envia alerta.',
            'Operador autenticado assume o chamado (claim) e registra a orientação (reply).',
            'Na retomada (resume), o agente recebe a observação atribuída ao especialista humano.',
            'O runtime conclui o atendimento sem reabrir ticket duplicado em caso de replay.',
        ],
        snippet=code('app/v2/runtime.py', 475, 495),
        source='app/v2/runtime.py:475–495 • Runtime.resume()',
        voice='Quando um atendimento exige análise humana, o sistema abre um ticket persistente em banco de dados, congela os requisitos da sessão e dispara notificação externa opcional. Assim que o operador autenticado assume o ticket e registra seu parecer, o endpoint de retomada acorda o agente semântico. O agente incorpora a decisão do especialista humano como fato verificado e sintetiza a resposta final ao lojista, mantendo idempotência em qualquer repetição.',
    ),
    dict(
        title='APIs Públicas, Clima, Câmbio e Busca Web',
        tag='08 / INTEGRAÇÕES EXTERNAS E RESILIÊNCIA',
        big='Timeouts, circuit breakers e preservação de fonte, data e precisão',
        lines=[
            'Clima via Open-Meteo com geocodificação real da localidade consultada.',
            'Câmbio via AwesomeAPI com fallback para Frankfurter, preservando precisão e data.',
            'Busca web extrai e apresenta as URLs oficiais reais retornadas pelo provedor.',
            'Indisponibilidade externa gera aviso transparente, sem inventar cotações.',
        ],
        snippet=code('app/v2/adapters.py', 295, 315),
        source='app/v2/adapters.py:295–315 • ToolGateway public tools',
        voice='Para consultas externas de contexto, como meteorologia, cotação de moedas e busca web, o sistema utiliza clientes HTTP com timeout curto e disjuntores de falha. As respostas preservam a precisão numérica original, a data de referência e a distinção clara entre cotação comercial ao vivo e taxa de referência diária. Na busca web, os endereços oficiais encontrados são preservados e citados diretamente na resposta.',
    ),
    dict(
        title='Validação Nativa ao Vivo no Gemini Free Tier',
        tag='09 / EVIDÊNCIA DE EXECUÇÃO REAL',
        big='Recibos HTTP e de provedor versionados em docs/eval_results/v2_live_integration',
        lines=[
            'Fluxos validados ao vivo: Crediário, Chip Algar/POS, Recomendação Get Smart, HITL, Clima/Câmbio e Ação.',
            'Ciclo HITL completo (abertura, claim, reply e resume) comprovado fim a fim com LLM real.',
            'Proposta e confirmação de ação de terminal validadas com controle de escopo e hash.',
            'Ledger de consumo append-only com contagem auditada de tokens a custo zero.',
        ],
        snippet=code('app/v2/model.py', 92, 110),
        source='docs/eval_results/v2_live_integration/ • app/v2/model.py',
        voice='O comportamento real do núcleo dois com a API Gemini está documentado nos recibos dentro da pasta docs eval results. Os registros comprovam a execução nativa dos fluxos documentais, diagnóstico parcial de terminal, recomendação contextualizada para perfil de cliente, ciclo completo de escalonamento humano com retomada e confirmação de ação remota em duas etapas, todos auditados em ledger de consumo a custo zero no nível gratuito.',
    ),
    dict(
        title='Engenharia de Qualidade: 1.820 Testes Aprovados',
        tag='10 / EVALUATION & QUALITY ENGINEERING',
        big='Suíte determinística com rede bloqueada, zero falhas e zero print()',
        lines=[
            'scripts/run_offline_tests.py bloqueia sockets externos e isola banco e cache em diretório temporário.',
            '1.820 testes automatizados passando em menos de 30 segundos, com zero falhas.',
            'Suíte de holdout independente avalia roteamento, precisão de tools, Hit@3 e groundedness.',
            'scripts/check_no_print.py garante logs estruturados JSON sem print solto em produção.',
        ],
        snippet=code('scripts/run_offline_tests.py', 19, 36),
        source='scripts/run_offline_tests.py:19–36 • 1.820 passed in 27.8s',
        voice='A validação de engenharia de software conta com mil oitocentos e vinte testes automatizados que rodam em menos de trinta segundos com bloqueio total de rede e isolamento de banco de dados em diretório temporário multiplataforma. Além dos contratos do núcleo dois, segurança, fronteiras de autorização e idempotência, o pipeline verifica a ausência de comandos print no código de produção e executa a suíte de avaliação de holdout.',
    ),
    dict(
        title='Produção, Containers Não-Root e Reprodutibilidade',
        tag='11 / PRODUCTION ENGINEERING',
        big='Docker multi-stage UID 10001, filesystem read-only e quickstart direto',
        lines=[
            'Imagem Docker enxuta rodando como usuário não-root (UID 10001) sem arquivos de teste ou segredos.',
            'Docker Compose com rootfs somente leitura, /tmp efêmero e volume exclusivo para /app/state.',
            'Pipeline GitHub Actions executa linter, 1.820 testes, avaliador e smoke test do container.',
            'README enxuto e objetivo com instruções diretas de execução, arquitetura e verificação.',
        ],
        snippet=code('Dockerfile', 24, 42),
        source='Dockerfile:24–42 • docker-compose.yml • .github/workflows/ci.yml',
        voice='Para operação em produção, o serviço é empacotado em imagem Docker multiestágio que executa sob usuário sem privilégios de administrador, com sistema de arquivos somente leitura e persistência restrita ao volume de estado. O workflow de integração contínua no GitHub Actions reproduz toda a bateria de testes e o smoke test do container. Com isso, a entrega reúne rigor de segurança, arquitetura limpa e evidências reais verificáveis.',
    ),
]


def wrap(draw, text, x, y, width=1120, size=27, fill='#e5e9ee', bold=False):
    f = font(size, bold)
    words = text.split()
    line = ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if draw.textlength(candidate, font=f) > width and line:
            draw.text((x, y), line, font=f, fill=fill)
            y += size + 9
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=f, fill=fill)
        y += size + 9
    return y


if '--check-inputs' in sys.argv:
    print('OK: 11 cenas V2 carregadas diretamente do repositório')
    sys.exit(0)

for dname in ('frames', 'audio', 'segments'):
    shutil.rmtree(O / dname, ignore_errors=True)
    (O / dname).mkdir(exist_ok=True)

meta = []
srt = []
elapsed = 0.0

for i, s in enumerate(scenes, 1):
    im = Image.new('RGB', (W, H), '#10151d')
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 12, H), fill='#ed123c')
    d.text((48, 30), s['tag'], font=font(17, True), fill='#ff5b79')
    d.text((48, 70), s['title'], font=font(40, True), fill='white')
    y = wrap(d, s['big'], 48, 131, size=28, fill='#ff7790', bold=True)
    y = max(y + 14, 188)
    for l in s['lines']:
        y = wrap(d, '• ' + l, 54, y, width=1170, size=23) + 6
    if 'snippet' in s:
        y += 6
        lines_snip = s['snippet'].splitlines()[:10]
        box_h = min(615, y + len(lines_snip) * 18 + 20)
        d.rounded_rectangle((48, y, 1232, box_h), radius=10, fill='#192330')
        for line in lines_snip:
            for part in textwrap.wrap(line, 112, replace_whitespace=False, drop_whitespace=False) or ['']:
                if y + 22 < 615:
                    d.text((62, y + 8), part, font=font(15, m=True), fill='#c7eadc')
                    y += 18
    d.line((48, 632, 1232, 632), fill='#34404d', width=1)
    wrap(d, s['source'], 48, 644, width=1170, size=15, fill='#9aa9ba')
    d.text(
        (48, 688),
        f'WALKTHROUGH TÉCNICO V2  •  1.820 TESTES APROVADOS  •  REVISÃO {SHA}',
        font=font(14, True),
        fill='#aeb9c8',
    )
    d.text((1170, 688), f'{i:02}/11', font=font(14, True), fill='#aeb9c8')
    frame = O / 'frames' / f'{i:02}.png'
    im.save(frame)
    txt = O / 'audio' / f'{i:02}.txt'
    txt.write_text(s['voice'], encoding='utf-8')
    audio = O / 'audio' / f'{i:02}.aiff'
    run(['/usr/bin/say', '-v', 'Luciana', '-r', '180', '-f', str(txt), '-o', str(audio)])
    dur = (
        float(
            subprocess.check_output(
                [FP, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(audio)],
                text=True,
            )
        )
        / 1.45
        + 1.0
    )
    seg = O / 'segments' / f'{i:02}.mp4'
    run(
        [
            FF, '-y', '-loop', '1', '-framerate', '12',
            '-i', str(frame), '-i', str(audio),
            '-af', 'atempo=1.45,apad=pad_dur=1.0',
            '-t', str(dur),
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage',
            '-crf', '28', '-pix_fmt', 'yuv420p', '-r', '12',
            '-c:a', 'aac', '-b:a', '72k', '-movflags', '+faststart',
            str(seg),
        ]
    )
    meta.append(
        {
            'scene': i,
            'title': s['title'],
            'start': round(elapsed, 3),
            'duration': round(dur, 3),
            'source': s['source'],
            'narration': s['voice'],
        }
    )
    words = s['voice'].split()
    chunks = [' '.join(words[j : j + 14]) for j in range(0, len(words), 14)]
    step = (dur - 1.0) / len(chunks)

    def tc(n):
        ms = round(n * 1000)
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        ss, ms = divmod(ms, 1000)
        return f'{h:02}:{m:02}:{ss:02},{ms:03}'

    for j, c in enumerate(chunks):
        srt.append(f'{len(srt)+1}\n{tc(elapsed+j*step)} --> {tc(elapsed+(j+1)*step)}\n{c}\n')
    elapsed += dur

(O / 'concat.txt').write_text(''.join(f"file 'segments/{i:02}.mp4'\n" for i in range(1, len(scenes) + 1)), encoding='utf-8')
(O / 'subtitles.srt').write_text('\n'.join(srt), encoding='utf-8')
(O / 'narration.txt').write_text('\n\n'.join(f'{i+1}. {s["title"]}\n{s["voice"]}' for i, s in enumerate(scenes)), encoding='utf-8')
(O / 'sources.json').write_text(
    json.dumps(
        {
            'revision_sha': SHA,
            'scenes': meta,
            'offline_suite_passed': 1820,
            'offline_suite_failed': 0,
            'live_receipts_dir': 'docs/eval_results/v2_live_integration',
            'render_method': 'Pillow code excerpts and architecture cards + macOS say + FFmpeg',
            'voice': 'macOS say / Luciana / 180 base / atempo 1.45 final',
        },
        ensure_ascii=False,
        indent=2,
    )
    + '\n',
    encoding='utf-8',
)
run([FF, '-y', '-f', 'concat', '-safe', '0', '-i', str(O / 'concat.txt'), '-c', 'copy', '-movflags', '+faststart', str(O / 'Getnet_Walkthrough_Tecnico.mp4')])
print(json.dumps({'duration_seconds': round(elapsed, 3), 'bytes': (O / 'Getnet_Walkthrough_Tecnico.mp4').stat().st_size, 'scenes': len(scenes)}, indent=2))
