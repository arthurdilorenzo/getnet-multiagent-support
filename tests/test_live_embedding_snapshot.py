"""Exercise snapshot identity through the public tool; only the SDK is fake."""
import concurrent.futures
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google import genai

from app.config import settings
from app.rag.embeddings import SemanticEmbedder as E
from app.rag.retriever import HybridKnowledgeRetriever, expanded_query
from app.tools import search_tools


@pytest.fixture
def neural_tool(monkeypatch, tmp_path):
    E._ensure_cache_loaded()
    with E._lock:
        saved = (copy.deepcopy(E._neural_vectors), list(E._mean_unit), E._alpha,
                 E._cache_loaded, E._transformation_fingerprint)
    payload_a = dict(model=E.EMBEDDING_MODEL, dimension=384,
                     vectors=copy.deepcopy(saved[0]), mean_unit=saved[1], alpha=saved[2])
    retriever = HybridKnowledgeRetriever(vector_index_path=tmp_path / "index.json")
    monkeypatch.setattr(search_tools, "retriever", retriever)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-snapshot-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)
    calls = []

    def fake_client(**kwargs):
        def embed(**request):
            calls.append(request)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 384)])
        return SimpleNamespace(models=SimpleNamespace(embed_content=embed))

    monkeypatch.setattr(genai, "Client", fake_client)
    try:
        yield retriever, payload_a, calls
    finally:
        with E._lock:
            (E._neural_vectors, E._mean_unit, E._alpha,
             E._cache_loaded, E._transformation_fingerprint) = saved


def test_public_tool_uses_dense_scores_for_stable_live_snapshot(neural_tool):
    """A blanket sparse fallback must not disable compatible live retrieval."""
    retriever, _, calls = neural_tool
    with patch.object(E, "cosine_similarity", wraps=E.cosine_similarity) as cosines:
        result = search_tools.search_getnet_knowledge("Controle live estável Calisto snapshot")

    assert len(calls) == 1
    assert result["retrieval_mode"] == "neural_hybrid"
    assert result["retrieval_method"] == "bm25_neural_rrf_candidates"
    assert cosines.call_count == len(retriever.chunks) > 0


@pytest.mark.parametrize("cache_at_live_entry", [False, True], ids=["generated", "cached"])
def test_public_tool_rejects_aba_live_snapshot(neural_tool, cache_at_live_entry):
    """Relabelling a B vector from global A after return must never authorize cosine."""
    retriever, payload_a, calls = neural_tool
    query = "Consulta inédita Calisto recarga concorrente ABA snapshot"
    search_query = expanded_query(query)
    canonical = f"{search_query} {' '.join(retriever._expand_query_tokens(search_query))}"
    payload_b = dict(model=E.EMBEDDING_MODEL, dimension=384,
                     vectors={E.cache_key(canonical): [0.0] * 383 + [1.0]}
                     if cache_at_live_entry else {},
                     mean_unit=list(payload_a["mean_unit"]), alpha=0.0)
    before_live, start_live = threading.Event(), threading.Event()
    after_live, return_live = threading.Event(), threading.Event()
    real_live = E.embed_query_live_snapshot
    captured = {}

    def paused_live(*args, **kwargs):
        before_live.set()
        assert start_live.wait(5), "Before-live barrier timed out"
        result = real_live(*args, **kwargs)
        captured.update(mode=result[1], fingerprint=E.current_transformation_fingerprint())
        after_live.set()
        assert return_live.wait(5), "After-live barrier timed out"
        return result

    with patch.object(E, "embed_query_live_snapshot", side_effect=paused_live), \
         patch.object(E, "cosine_similarity", wraps=E.cosine_similarity) as cosines:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(search_tools.search_getnet_knowledge, query)
            try:
                assert before_live.wait(5), "Resolver did not reach the live boundary"
                assert E.validate_and_load_cache_payload(json.dumps(payload_b))
                fp_b = E.current_transformation_fingerprint()
                assert fp_b not in {c.transformation_fingerprint for c in retriever.chunks}
                start_live.set()
                assert after_live.wait(5), "Real embedder did not finish"
                assert captured == {"mode": "neural_cached" if cache_at_live_entry else "neural_live",
                                    "fingerprint": fp_b}
                assert E.validate_and_load_cache_payload(json.dumps(payload_a))
                return_live.set()
                result = future.result(timeout=5)
            finally:
                start_live.set()
                return_live.set()

    assert len(calls) == (0 if cache_at_live_entry else 1)
    assert result["retrieval_mode"] == "sparse_only"
    assert result["retrieval_method"] == "bm25_sparse_only_candidates"
    assert cosines.call_count == 0


def test_public_tool_uses_compatible_cache_published_at_live_entry(neural_tool):
    """A cache hit inside the live helper must carry its own compatible identity."""
    retriever, payload_a, calls = neural_tool
    query = "Consulta inédita Calisto cache publicado antes do helper"
    search_query = expanded_query(query)
    canonical = f"{search_query} {' '.join(retriever._expand_query_tokens(search_query))}"
    payload_a["vectors"][E.cache_key(canonical)] = [0.0] * 383 + [1.0]
    real_live = E.embed_query_live_snapshot

    def publish_cache_before_live(*args, **kwargs):
        assert E.validate_and_load_cache_payload(json.dumps(payload_a))
        return real_live(*args, **kwargs)

    with patch.object(E, "embed_query_live_snapshot", side_effect=publish_cache_before_live), \
         patch.object(E, "cosine_similarity", wraps=E.cosine_similarity) as cosines:
        result = search_tools.search_getnet_knowledge(query)

    assert not calls
    assert result["retrieval_mode"] == "neural_hybrid"
    assert cosines.call_count == len(retriever.chunks) > 0
