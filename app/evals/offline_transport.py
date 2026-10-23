"""Synthetic raw HTTP contracts for the offline CLI; never a production fallback."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import httpx


_active = ContextVar('getnet_synthetic_external_eval', default=False)

def external_services_mode() -> str:
    return 'synthetic_external_api_fixtures' if _active.get() else 'live_external_services'

def external_fixture_response(request: httpx.Request) -> httpx.Response:
    """Only known evaluation provider routes; unexpected I/O cannot hit the net."""
    from app.tools.search_tools import _today_local
    today = _today_local()
    if request.url.host == 'api.open-meteo.com' and request.url.path == '/v1/forecast':
        body = {'daily': {
            'time': [(today + timedelta(days=i)).isoformat() for i in range(7)],
            'temperature_2m_max': [24.0] * 7, 'temperature_2m_min': [16.0] * 7,
            'precipitation_probability_max': [35] * 7, 'weathercode': [3] * 7,
        }}
        return httpx.Response(200, json=body, request=request)
    if request.url.host == 'economia.awesomeapi.com.br' and request.url.path == '/last/EUR-BRL':
        body = {'EURBRL': {'code': 'EUR', 'codein': 'BRL', 'bid': '6.1234',
            'high': '6.2', 'low': '6.0', 'create_date': today.isoformat() + ' 12:00:00'}}
        return httpx.Response(200, json=body, request=request)
    raise RuntimeError('Unexpected HTTP request in synthetic offline evaluation')


@contextmanager
def synthetic_external_transport():
    """Replace only search-tool HTTP transport, restoring it even after failure."""
    from app.tools import search_tools
    def client(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(external_fixture_response), **kwargs)
    token = _active.set(True)
    try:
        with patch.object(search_tools, 'httpx', SimpleNamespace(Client=client)):
            yield
    finally:
        _active.reset(token)
