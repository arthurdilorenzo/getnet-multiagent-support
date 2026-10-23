"""The offline evaluation must validate raw provider contracts without internet."""
import importlib
import socket

import httpx
import pytest

from app.config import settings
from app.observability.resilience import breaker_registry
from app.tools import search_tools


def test_synthetic_context_runs_real_provider_parsers_without_network(monkeypatch):
    from app.evals.offline_transport import synthetic_external_transport
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('offline attempted network'))
    original=search_tools.httpx
    breaker_registry.reset_all()
    with synthetic_external_transport():
        weather=search_tools.search_weather('Porto Alegre',1)
        rate=search_tools.search_currency_rate('EUR','BRL')
    assert search_tools.httpx is original
    assert weather['max_temp_celsius']==24.0 and weather['min_temp_celsius']==16.0
    assert rate['current_rate_brl']==6.1234 and rate['currency_pair']=='EUR/BRL'
    assert weather['source_url'].startswith('https://api.open-meteo.com/')
    breaker_registry.reset_all()


def test_unknown_fixture_request_fails_closed():
    from app.evals.offline_transport import external_fixture_response
    with pytest.raises(RuntimeError):
        external_fixture_response(httpx.Request('GET','https://unrecognized.example/data'))


def test_context_restores_transport_after_exception():
    from app.evals.offline_transport import synthetic_external_transport
    original=search_tools.httpx
    with pytest.raises(ValueError):
        with synthetic_external_transport():
            raise ValueError('synthetic interruption')
    assert search_tools.httpx is original


def test_offline_cli_marks_fixture_mode_and_keeps_all_acceptance_gates(monkeypatch,capsys):
    module=importlib.import_module('app.evals.evaluator')
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    def evaluate():
        assert search_tools.httpx is not httpx
        return {'total_cases':13,'metrics':{'routing_accuracy':1.,'mean_tool_recall':1.,
            'mean_groundedness_score':1.,'mean_tool_precision':1.,'safe_tool_selection_rate':1.,
            'overall_pass_rate':.8462}}
    monkeypatch.setattr(module.evaluator,'run_suite',evaluate)
    assert module.main()==1
    captured=capsys.readouterr()
    assert 'overall_pass_rate' in captured.err
    assert 'synthetic_external_api_fixtures' in captured.out


def test_configured_native_cli_does_not_use_synthetic_transport(monkeypatch,capsys):
    module=importlib.import_module('app.evals.evaluator')
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    def evaluate():
        assert search_tools.httpx is httpx
        return {'total_cases':1,'metrics':{'routing_accuracy':1.,'mean_tool_recall':1.,
            'mean_groundedness_score':1.,'overall_pass_rate':1.}}
    monkeypatch.setattr(module.evaluator,'run_suite',evaluate)
    assert module.main()==0
    assert 'live_external_services' in capsys.readouterr().out


@pytest.mark.parametrize('bad_field',['nonfinite_temperature','wrong_date'])
def test_fixture_payload_still_passes_through_real_rejection_gates(monkeypatch,bad_field):
    from app.evals import offline_transport
    from app.observability.resilience import ExternalServiceUnavailableError
    import app.observability.resilience as resilience
    original=offline_transport.external_fixture_response
    def malformed(request):
        response=original(request);body=response.json()
        if bad_field=='nonfinite_temperature':body['daily']['temperature_2m_max'][1]='NaN'
        else:body['daily']['time'][1]='2001-01-01'
        return httpx.Response(200,json=body,request=request)
    monkeypatch.setattr(offline_transport,'external_fixture_response',malformed)
    monkeypatch.setattr(resilience.time,'sleep',lambda *args:None)
    breaker_registry.reset_all()
    try:
        with offline_transport.synthetic_external_transport(),pytest.raises(ExternalServiceUnavailableError):
            search_tools.search_weather('Porto Alegre',1)
    finally:
        breaker_registry.reset_all()
