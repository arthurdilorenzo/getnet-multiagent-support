import concurrent.futures
import copy
import json
import math
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from app.config import settings
from app.rag.embeddings import SemanticEmbedder as E
from app.rag.retriever import HybridKnowledgeRetriever, IndexedChunk, expanded_query
from app.tools.search_tools import search_getnet_knowledge
from google import genai


@pytest.fixture(autouse=True)
def restore_embedder_state():
    E._ensure_cache_loaded()
    saved = (
        copy.deepcopy(E._neural_vectors),
        list(E._mean_unit),
        E._alpha,
        E._cache_loaded,
        getattr(E, "_transformation_fingerprint", ""),
    )
    yield
    with E._lock:
        E._neural_vectors = copy.deepcopy(saved[0])
        E._mean_unit = list(saved[1])
        E._alpha = saved[2]
        E._cache_loaded = saved[3]
        E._transformation_fingerprint = saved[4]


# ==============================================================================
# DEFEITO 1: ORÇAMENTO DE EMBEDDINGS (A1 / R1)
# ==============================================================================

def test_budget_zero_or_negative_makes_zero_sdk_calls(monkeypatch):
    """
    1.a) Saldo zero/negativo: zero chamadas ao SDK.
    No código atual, falha porque actual_timeout = max(0.1, timeout_sec) eleva
    0.0 ou -1.0 para 100 ms e dispara a chamada remota.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    retriever = HybridKnowledgeRetriever()
    calls = []

    def mock_client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(embed_content=lambda **kw: None)
        )

    with patch.object(genai, "Client", side_effect=mock_client):
        # 1. Orçamento zero
        calls.clear()
        ctx_zero = retriever.resolve_query_vector("Consulta teste orçamento zero", timeout_sec=0.0)
        assert len(calls) == 0, f"Esperado 0 chamadas para budget 0.0, mas foram feitas {len(calls)}"
        assert ctx_zero["retrieval_mode"] == "sparse_only"

        # 2. Orçamento negativo
        calls.clear()
        ctx_neg = retriever.resolve_query_vector("Consulta teste orçamento negativo", timeout_sec=-1.0)
        assert len(calls) == 0, f"Esperado 0 chamadas para budget negativo, mas foram feitas {len(calls)}"
        assert ctx_neg["retrieval_mode"] == "sparse_only"


def test_budget_short_20ms_never_exceeds_20ms(monkeypatch):
    """
    1.b) Saldo de 20 ms: timeout configurado nunca pode superar 20 ms.
    No código atual, falha porque actual_timeout = max(0.1, 0.02) eleva para 100 ms.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    retriever = HybridKnowledgeRetriever()
    calls = []

    def mock_client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(embed_content=lambda **kw: None)
        )

    with patch.object(genai, "Client", side_effect=mock_client):
        ctx = retriever.resolve_query_vector("Consulta teste orçamento 20ms", timeout_sec=0.02)

    assert len(calls) == 1
    observed_timeout = calls[0].get("http_options", {}).get("timeout")
    assert observed_timeout is not None
    assert observed_timeout <= 20, f"Timeout configurado ({observed_timeout} ms) superou o saldo de 20 ms!"


def test_local_time_consumes_entire_balance_makes_zero_sdk_calls(monkeypatch):
    """
    1.c) Tempo local consome todo o saldo: zero chamadas remotas.
    A tool pública search_getnet_knowledge gerencia deadline monotônico;
    se o tempo consumido antes da chamada remota exceder o orçamento, zero chamadas ocorrem.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    calls = []

    def mock_client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(embed_content=lambda **kw: None)
        )

    # Simula que o relógio monotônico avançou 10 segundos durante o processamento local antes da rede
    orig_monotonic = time.monotonic
    time_offset = [0.0]

    def fake_monotonic():
        return orig_monotonic() + time_offset[0]

    with patch.object(time, "monotonic", side_effect=fake_monotonic), patch.object(genai, "Client", side_effect=mock_client):
        # Avança o relógio para simular que o deadline estourou durante o pipeline local
        # Em search_getnet_knowledge, total_budget_sec é lido de settings (5.0s)
        monkeypatch.setattr(settings, "TOOL_TIMEOUT_SECONDS", 1.0)
        monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 1.0)

        # Hook para avançar o relógio antes de tentar a chamada remota
        real_get = E.get_cached_vector
        def slow_lookup(text):
            time_offset[0] += 2.0  # consome 2 segundos (estoura o orçamento de 1s)
            return real_get(text)

        with patch.object(E, "get_cached_vector", side_effect=slow_lookup):
            outcome = search_getnet_knowledge("Consulta com deadline local esgotado antes da rede")

    assert len(calls) == 0, f"Esperado 0 chamadas quando o saldo local se esgota, mas foram feitas {len(calls)}"
    assert outcome["retrieval_mode"] == "sparse_only"


def test_timeout_or_429_single_attempt_no_cascade(monkeypatch):
    """
    1.d) Timeout/429: uma única tentativa remota, sem cascata de variantes.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    calls = []

    def timeout_client(**kwargs):
        def embed(**kw):
            calls.append(kwargs)
            raise TimeoutError("Synthetic timeout")
        return SimpleNamespace(models=SimpleNamespace(embed_content=embed))

    with patch.object(genai, "Client", side_effect=timeout_client):
        outcome = search_getnet_knowledge("Consulta única tentativa timeout 429")

    assert len(calls) == 1
    assert calls[0].get("http_options", {}).get("retry_options") == {"attempts": 1}
    assert outcome["retrieval_mode"] == "sparse_only"


def test_usable_cache_makes_zero_remote_calls(monkeypatch):
    """
    1.e) Cache utilizável: zero chamadas remotas.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    calls = []

    def mock_client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(embed_content=lambda **kw: None)
        )

    # Consulta presente no cache neural
    known_query = "What's the difference between the Get Clássica and the Get Smart?"
    with patch.object(genai, "Client", side_effect=mock_client):
        outcome = search_getnet_knowledge(known_query)

    assert len(calls) == 0


# ==============================================================================
# DEFEITO 2: VETOR E FINGERPRINT DO MESMO SNAPSHOT (A2 / R3)
# ==============================================================================

def test_concurrent_reload_before_cache_read_degrades_to_sparse_only():
    """
    2.a–e) Teste concorrente obrigatório com barreira:
    a) documentos indexados sob transformação A (alpha=0.68);
    b) busca pausa antes do lookup real do cache;
    c) outra thread publica transformação B (alpha=0.0) com query cacheada;
    d) busca continua;
    e) resultado deve ser sparse_only, com zero cossenos incompatíveis.

    No código atual, falha porque o lookup do cache não captura o fingerprint
    junto com o vetor sob o mesmo lock, nem revalida a identidade do vetor
    retornado contra os documentos antes do cosseno. O resultado é neural_hybrid
    com 13 cossenos incompatíveis.
    """
    retriever = HybridKnowledgeRetriever()
    q = "Consulta teste concorrência reload entre verificação e leitura"
    canonical = f"{expanded_query(q)} {' '.join(retriever._expand_query_tokens(expanded_query(q)))}"

    entered = threading.Event()
    resume = threading.Event()

    real_get = E.get_cached_vector

    def paused_get(text):
        if text == canonical and not entered.is_set():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError("Barrier timed out")
        return real_get(text)

    cosines = []
    real_cos = E.cosine_similarity

    def spy_cos(a, b):
        cosines.append((a, b))
        return real_cos(a, b)

    new_payload = {
        "model": E.EMBEDDING_MODEL,
        "dimension": 384,
        "vectors": {E.cache_key(canonical): [1.0 / math.sqrt(384)] * 384},
        "mean_unit": list(E._mean_unit),
        "alpha": 0.0,
    }

    with patch.object(E, "get_cached_vector", side_effect=paused_get), patch.object(E, "cosine_similarity", side_effect=spy_cos):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(retriever.retrieve_candidates, q)
            if not entered.wait(5):
                raise RuntimeError("Resolver never reached the barrier")
            # Publica nova transformação B
            loaded = E.validate_and_load_cache_payload(json.dumps(new_payload))
            assert loaded is True
            resume.set()
            result = future.result(timeout=5)

    assert result["retrieval_mode"] == "sparse_only", f"Modo esperado 'sparse_only', obtido '{result['retrieval_mode']}'"
    assert len(cosines) == 0, f"Esperado 0 chamadas de cosseno incompatíveis, mas ocorreram {len(cosines)}"


def test_compatible_query_and_documents_use_neural_hybrid():
    """
    Documentos e consultas com mesmo snapshot/fingerprint usam neural_hybrid.
    """
    retriever = HybridKnowledgeRetriever()
    known_query = "What's the difference between the Get Clássica and the Get Smart?"
    res = retriever.retrieve_candidates(known_query)
    assert res["retrieval_mode"] == "neural_hybrid"
    assert res["retrieval_method"] == "bm25_neural_rrf_candidates"


def test_sdk_inflight_reload_discards_old_response(monkeypatch):
    """
    Reload durante chamada SDK não contamina o novo cache.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    provider_entered = threading.Event()
    provider_resume = threading.Event()
    raw_q = "Consulta inédita teste em voo"

    def paused_client(**kwargs):
        def embed(**kw):
            provider_entered.set()
            if not provider_resume.wait(5):
                raise RuntimeError("Provider barrier timed out")
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 384)])
        return SimpleNamespace(models=SimpleNamespace(embed_content=embed))

    with patch.object(genai, "Client", side_effect=paused_client):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(E.embed_query_live, raw_q)
            if not provider_entered.wait(5):
                raise RuntimeError("SDK barrier was not reached")
            # Recarrega cache com nova transformação
            p = {
                "model": E.EMBEDDING_MODEL,
                "dimension": 384,
                "vectors": {},
                "mean_unit": list(E._mean_unit),
                "alpha": 0.0,
            }
            assert E.validate_and_load_cache_payload(json.dumps(p))
            provider_resume.set()
            vec, mode = future.result(timeout=5)

    assert vec is None
    assert mode == "sparse_only"
    assert E.cache_key(raw_q) not in E._neural_vectors
