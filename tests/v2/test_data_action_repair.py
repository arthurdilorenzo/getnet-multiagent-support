"""Regression probes run in a subprocess with network and dotenv disabled."""
import os
from pathlib import Path
import subprocess
import sys


def test_data_action_frontiers_are_offline_and_exact(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = r'''
import json, os, socket, sys
from pathlib import Path
scratch=Path(sys.argv[1])
def deny(*a,**k): raise AssertionError('Network prohibited')
socket.socket.connect=deny
socket.socket.connect_ex=deny
socket.create_connection=deny
from pydantic_settings import BaseSettings
original=BaseSettings.__init__
def isolated(self,*a,**kw):
    kw['_env_file']=None
    original(self,*a,**kw)
BaseSettings.__init__=isolated
os.environ.update(DATA_DIR=str(scratch),SQLITE_DB_PATH=str(scratch/'default.db'),MOCK_CUSTOMERS_PATH=str(scratch/'customers.json'),VECTOR_INDEX_PATH=str(scratch/'index.json'),GEMINI_API_KEY='',GETNET_LIVE_EMBEDDINGS='0',AUTH_API_KEY='test-action-key-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',JWT_SECRET='test-action-jwt-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',TELEGRAM_BOT_TOKEN='',TELEGRAM_CHAT_ID='',HITL_WEBHOOK_URL='')
from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.security.auth import SecurityContext
from app.v2.adapters import ToolGateway
from app.v2.contracts import Binding
from app.schemas import ToolExecutionResult
from app.security.action_policy import confirm_action, ActionPolicyError
customers=scratch/'customers.json'
customers.write_text(json.dumps({'customers':{'u':{'user_id':'u','ramo_declarado_sessao':'Padaria e Confeitaria','equipamento_atual_conta':'1x Smart','necessidade_registrada_em_perfil':'apps Android','terminals':[{'terminal_id':'POS-1111','chip_algar_status':'ativo'}]},'v':{'user_id':'v','segment':'Other tenant','terminals':[{'terminal_id':'POS-2222','status':'offline'}]}}}))
store=StateStore(scratch/'probe.db');store.ensure_session('s','u')
g=ToolGateway(customers,store=store,queue=HITLQueueManager(store))
read=SecurityContext('u',scopes=['chat:write','customer:read_own'])
write=SecurityContext('u',scopes=['chat:write','customer:read_own','terminal:action_own'])
def bind(tool,res_type,res,ptr,unit=None):return Binding(binding_id='r:b',tool_name=tool,resource_type=res_type,resource_id=res,field_pointer=ptr,unit=unit)
# Actual source observation cannot be relabeled as a different terminal.
result=g.execute('diagnose_terminal_issue',{'terminal_id':'POS-1111'},read,'s')
b=bind('diagnose_terminal_issue','terminal','POS-9999','/chip_algar_status')
assert not any(f.availability=='available' for f in g.extract_facts(result.tool_name,result,[b]))
b.resource_id='POS-1111'
assert g.extract_facts(result.tool_name,result,[b])[0].value=='ativo'
# Descriptors expose the real profile and no invented segment or status.
desc={d['tool_name']:d for d in g.get_tool_descriptors(user_id='u')}
profile_fields={f['field_pointer'] for f in desc['get_customer_profile']['fields']}
assert '/profile/ramo_declarado_sessao' in profile_fields
assert '/profile/equipamento_atual_conta' in profile_fields
assert '/profile/necessidade_registrada_em_perfil' in profile_fields
assert '/profile/segment' not in profile_fields
terminal_fields={f['field_pointer'] for f in desc['diagnose_terminal_issue']['fields']}
assert '/chip_algar_status' in terminal_fields and '/terminal/status' not in terminal_fields
assert 'POS-2222' not in json.dumps(desc)
# Only explicit server-owned symbolic references can resolve.
assert g.resolve_resource_id('customer','current_customer','u')=='u'
assert g.resolve_resource_id('terminal','current','u')=='POS-1111'
assert g.resolve_resource_id('terminal','POS-9999','u')=='POS-9999'
assert g.resolve_resource_id('customer','v','u')=='v'
# Scope is enforced before proposal creation, and registry knows this is a write.
args={'terminal_id':'POS-1111','action_type':'remote_reboot','reason':'offline probe','idempotency_key':'once'}
denied=g.execute('execute_terminal_remote_action',args,read,'s')
assert not denied.success and denied.error_code=='INSUFFICIENT_TOOL_SCOPE'
assert g.registry.get_tool('execute_terminal_remote_action').is_write_action
foreign=g.execute('execute_terminal_remote_action',{**args,'terminal_id':'POS-2222','idempotency_key':'foreign-denied'},write,'s')
assert not foreign.success and foreign.error_code=='CROSS_TENANT_ACCESS_DENIED'
first=g.execute('execute_terminal_remote_action',args,write,'s')
assert first.success and first.data['status']=='PENDING_CONFIRMATION'
again=g.execute('execute_terminal_remote_action',args,write,'s')
assert again.idempotent_replay and again.data['proposal_id']==first.data['proposal_id']
conflict=g.execute('execute_terminal_remote_action',{**args,'reason':'different'},write,'s')
assert conflict.error_code=='IDEMPOTENCY_CONFLICT'
with store._connect() as conn:
    assert conn.execute('SELECT count(*) FROM terminal_actions').fetchone()[0]==0
    try:
        conn.execute("UPDATE terminal_action_proposals SET payload_json='{}'")
    except Exception: pass
    else: raise AssertionError('Proposal payload was mutable')
try: confirm_action(first.data['proposal_id'],'u','0'*64,store=store,customers_path=customers)
except ActionPolicyError as exc: assert exc.code=='PROPOSAL_HASH_MISMATCH'
else: raise AssertionError('Wrong confirmation hash accepted')
receipt=confirm_action(first.data['proposal_id'],'u',first.data['payload_hash'],store=store,customers_path=customers)
replay=confirm_action(first.data['proposal_id'],'u',first.data['payload_hash'],store=store,customers_path=customers)
assert replay['idempotent_replay'] and replay['action_id']==receipt['action_id']
with store._connect() as conn: assert conn.execute('SELECT count(*) FROM terminal_actions').fetchone()[0]==1
# Public metadata retains actual pair/date/source/type, even without base_currency keys.
fx=ToolExecutionResult(tool_name='get_currency_exchange_rate',success=True,operation_id='fx',outcome_state='completed',data={'currency_pair':'USD/GBP','current_rate_brl':0.77,'updated_at':'2026-10-29T12:00:00','source':'AwesomeAPI','rate_type':'market_quote'})
facts=g.extract_facts(fx.tool_name,fx,[])
assert {f.resource_id for f in facts}=={'USD/GBP'}
rate=next(f for f in facts if f.field_pointer=='/rate')
assert rate.unit=='GBP' and rate.availability=='available'
assert any(f.field_pointer=='/rate_type' and f.value=='market_quote' for f in facts)
assert all('Referência diária oficial' not in f.scope_notice for f in facts)
assert 'AwesomeAPI' in rate.scope_notice and '2026-10-29' in rate.scope_notice
malformed=fx.model_copy(update={'data':{**fx.data,'current_rate_brl':'invented'}})
assert next(f for f in g.extract_facts(malformed.tool_name,malformed,[]) if f.field_pointer=='/rate').availability!='available'
wrong_pair=bind(fx.tool_name,'currency','EUR/BRL','/rate','BRL')
assert not any(f.availability=='available' for f in g.extract_facts(fx.tool_name,fx,[wrong_pair]))
missing=fx.model_copy(update={'data':{'rate':0.77,'source':'Unknown'}})
assert not any(f.availability=='available' for f in g.extract_facts(missing.tool_name,missing,[]))
weather=ToolExecutionResult(tool_name='get_weather_forecast',success=True,operation_id='w',outcome_state='completed',data={'city':'Curitiba, PR','target_date':'2026-10-30','forecast_horizon':'Amanhã','source':'Actual weather provider','max_temp_celsius':24.0,'rain_probability_percent':45.0})
wf=g.extract_facts(weather.tool_name,weather,[])
assert any(f.field_pointer=='/rain_probability_percent' and f.availability=='available' for f in wf)
assert all('Actual weather provider' in f.scope_notice and '2026-10-30' in f.scope_notice and 'Curitiba' in f.scope_notice for f in wf)
assert any(f.field_pointer=='/source' and f.value=='Actual weather provider' for f in wf)
assert g.resolve_resource_id('weather','Curitiba','u',{'city':'Curitiba'},weather)=='Curitiba, PR'
assert g.resolve_resource_id('weather','Other city','u',{'city':'Curitiba'},weather)=='Other city'
print('offline data/action probes passed')
'''
    completed = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=root,
                               env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'},
                               capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'offline data/action probes passed' in completed.stdout
