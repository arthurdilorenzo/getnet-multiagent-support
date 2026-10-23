"""Validação isolada do baseline; não lê .env e bloqueia sockets."""
import hashlib,io,json,os,pathlib,shutil,subprocess,sys,tarfile,tempfile,argparse
OUT=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Valida contratos em git archive do baseline, sem rede ou dotenv.')
parser.add_argument('--repo',type=pathlib.Path,help='Checkout Git que contém o baseline d7e2125')
parser.add_argument('--python',type=pathlib.Path,help='Python com dependências de requirements-lock.txt já instaladas')
args=parser.parse_args()
REPO=args.repo.resolve() if args.repo else next((p for p in OUT.parents if (p/'.git').exists() and (p/'app').is_dir()),None)
if REPO is None:parser.error('Não encontrei checkout do produto. Informe --repo /caminho/checkout')
PYTHON=args.python.resolve() if args.python else (REPO/'.venv/bin/python' if (REPO/'.venv/bin/python').exists() else pathlib.Path(sys.executable))
subprocess.run(['git','-C',str(REPO),'cat-file','-e','d7e2125129d5806ee38730f12868aeaaba469112^{commit}'],check=True)
SHA='d7e2125129d5806ee38730f12868aeaaba469112'
with tempfile.TemporaryDirectory(prefix='baseline-',dir=OUT) as temp:
    root=pathlib.Path(temp)
    raw=subprocess.check_output(['git','-C',str(REPO),'archive',SHA])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive: archive.extractall(root,filter='data')
    runner=root/'isolated_runner.py'
    runner.write_text('''import os,socket,sys\nfrom unittest.mock import patch\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parent\nos.environ.update(GEMINI_API_KEY='',SQLITE_DB_PATH=str(ROOT/'state.sqlite'),VECTOR_INDEX_PATH=str(ROOT/'vectors.json'),HITL_WEBHOOK_URL='',TELEGRAM_BOT_TOKEN='',TELEGRAM_CHAT_ID='')\nimport pytest\nwith patch('pydantic_settings.sources.providers.dotenv.DotEnvSettingsSource._read_env_files',return_value={}),patch('socket.socket.connect',side_effect=RuntimeError('walkthrough: network disabled')),patch('socket.create_connection',side_effect=RuntimeError('walkthrough: network disabled')):\n code=pytest.main(['-q','tests/test_action_integrity.py','tests/test_http_actions_v3.py','tests/test_p0_action_security.py','tests/test_security_boundaries.py','tests/test_gemini_function_calling.py','tests/test_state_hitl_supervisor_v3.py'])\nsys.exit(code)\n''')
    proc=subprocess.run([str(PYTHON),str(runner)],cwd=root,capture_output=True,text=True)
    # O log pytest contém apenas resumo: não publicar tracebacks de falha que possam conter payloads.
    lines=[line for line in proc.stdout.splitlines() if (' passed' in line or 'failed' in line or line.startswith('tests/'))]
    summary={'baseline_sha':SHA,'command':'python isolated_runner.py (archive baseline; 6 módulos de contratos)','returncode':proc.returncode,'summary_lines':lines,'network_disabled':True,'dotenv_read_suppressed':True,'hardware_effects':'simulated','test_files':['test_action_integrity.py','test_http_actions_v3.py','test_p0_action_security.py','test_security_boundaries.py','test_gemini_function_calling.py','test_state_hitl_supervisor_v3.py']}
    (OUT/'local-validation-rerun.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if proc.returncode:sys.exit(proc.returncode)
