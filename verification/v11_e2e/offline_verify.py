"""Local verification wrapper: prohibit all outbound IP sockets before app import."""
import os
import socket
import sys

os.environ['GEMINI_API_KEY'] = ''
os.environ.pop('GETNET_LIVE_TESTS', None)
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex

def deny_ip_connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('IP network forbidden in offline verification')
    return original_connect(sock, address)

def deny_ip_connect_ex(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('IP network forbidden in offline verification')
    return original_connect_ex(sock, address)

socket.socket.connect = deny_ip_connect
socket.socket.connect_ex = deny_ip_connect_ex
sys.path.insert(0, os.getcwd())
if sys.argv[1:2] == ['eval']:
    import runpy
    sys.argv = ['app.evals.evaluator']
    runpy.run_module('app.evals.evaluator', run_name='__main__')
else:
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:] or ['-q']))
