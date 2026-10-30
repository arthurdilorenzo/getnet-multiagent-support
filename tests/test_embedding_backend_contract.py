import copy
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from app.config import settings
from app.rag.embeddings import SemanticEmbedder
from app.rag.retriever import HybridKnowledgeRetriever, IndexedChunk, expanded_query
from app.tools.search_tools import search_getnet_knowledge
from google import genai


@pytest.fixture(autouse=True)
def restore_embedder_cache():
    SemanticEmbedder._ensure_cache_loaded()
    saved_vectors = copy.deepcopy(SemanticEmbedder._neural_vectors)
    saved_mean = list(SemanticEmbedder._mean_unit)
    saved_alpha = SemanticEmbedder._alpha
    saved_loaded = SemanticEmbedder._cache_loaded
    yield
    SemanticEmbedder._neural_vectors = saved_vectors
    SemanticEmbedder._mean_unit = saved_mean
    SemanticEmbedder._alpha = saved_alpha
    SemanticEmbedder._cache_loaded = saved_loaded


def test_unseen_query_without_api_uses_sparse_only_and_never_hashes_against_neural_docs(monkeypatch):
    """
    Query inédita + corpus neural com API desligada:
    Garante que tanto retrieve_candidates quanto retrieve_hybrid operam em sparse-only,
    sem nenhum hashing implícito e sem calcular cosseno incompatível.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", False)

    unseen_query = "Uma pergunta absolutamente inédita sobre protocolo Orion Getnet 987654321"
    retriever = HybridKnowledgeRetriever()

    # Espião em cosine_similarity para assegurar que nenhum cosseno é calculado em sparse-only
    cosine_calls = []
    orig_cosine = SemanticEmbedder.cosine_similarity
    def spy_cosine(a, b):
        cosine_calls.append((a, b))
        return orig_cosine(a, b)

    with patch.object(SemanticEmbedder, "cosine_similarity", side_effect=spy_cosine):
        candidates = retriever.retrieve_candidates(unseen_query)
        hybrid = retriever.retrieve_hybrid(unseen_query)

    assert candidates["retrieval_mode"] == "sparse_only"
    assert hybrid["retrieval_mode"] == "sparse_only"
    assert hybrid["abstained"] is True
    # Zero chamadas a cosine_similarity
    assert len(cosine_calls) == 0


def test_public_tool_offline_propagates_backend_and_does_not_mix_backends(monkeypatch):
    """
    Mesmo caso pela tool search_getnet_knowledge real:
    A tool pública propaga retrieval_mode, abstém honestamente e não compara hashing com neural.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", False)

    novel = "Consulta pública inédita de terminal sem presença de comprador, protocolo Vega"
    pairs = []
    orig_cosine = SemanticEmbedder.cosine_similarity
    def capture(a, b):
        pairs.append((a, b))
        return orig_cosine(a, b)

    with patch.object(SemanticEmbedder, "cosine_similarity", side_effect=capture):
        outcome = search_getnet_knowledge(novel)

    assert outcome.get("retrieval_mode") == "sparse_only"
    assert outcome.get("abstained") is True
    assert len(pairs) == 0


def test_public_tool_native_mode_uses_compatible_candidates_and_propagates_mode(monkeypatch):
    """
    Mesmo caso pela tool nativa (GEMINI_API_KEY ativa em settings, fronteira mockada offline):
    Ramo nativo usa candidatos compatíveis, propaga retrieval_mode e não oculta o backend.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", False)

    novel = "Consulta nativa inédita Orion para autorização semântica pendente"
    pairs = []
    orig_cosine = SemanticEmbedder.cosine_similarity
    def capture(a, b):
        pairs.append((a, b))
        return orig_cosine(a, b)

    with patch.object(SemanticEmbedder, "cosine_similarity", side_effect=capture):
        outcome = search_getnet_knowledge(novel)

    assert outcome.get("retrieval_mode") == "sparse_only"
    assert outcome.get("answerability", {}).get("method") == "pending_native_semantics"
    assert len(pairs) == 0


def test_partial_corpus_update_conservatively_degrades_to_sparse_only():
    """
    Corpus parcialmente atualizado: indexação de um chunk cacheado e outro ausente do cache.
    Política conservadora sparse-only: documento atualizado não some e nenhum par neural/hashing é comparado.
    """
    retriever = HybridKnowledgeRetriever()
    original = retriever.chunks[0]
    updated = IndexedChunk(
        chunk_id="updated_doc_editorial",
        source_file=original.source_file,
        source_url=original.source_url,
        title=original.title,
        content=original.content + " Revisão editorial Orion vinte e sete de outubro.",
        provenance=original.provenance,
    )
    retriever.chunks = [original, updated]
    retriever._build_bm25_statistics()

    known_query = "What's the difference between the Get Clássica and the Get Smart?"
    mixed_comparisons = []
    orig_cosine = SemanticEmbedder.cosine_similarity
    def capture_mixed(a, b):
        mixed_comparisons.append((a, b))
        return orig_cosine(a, b)

    with patch.object(SemanticEmbedder, "cosine_similarity", side_effect=capture_mixed):
        result = retriever.retrieve_hybrid(known_query)

    # Degradação conservadora para sparse-only para o conjunto inteiro
    assert result.get("retrieval_mode") == "sparse_only"
    assert len(mixed_comparisons) == 0


def test_entirely_uncached_corpus_uses_sparse_only_in_production():
    """
    Corpus inteiro fora do cache sem opt-in explícito de teste:
    Opera em sparse-only, nunca infere synthetic_hashing automaticamente.
    """
    retriever = HybridKnowledgeRetriever()
    uncached_chunk = IndexedChunk(
        chunk_id="uncached_1",
        source_file="new_file.md",
        source_url="https://site.getnet.com.br/novo",
        title="Novo Produto",
        content="Informações sobre um produto totalmente inédito fora do cache.",
    )
    retriever.chunks = [uncached_chunk]
    retriever._build_bm25_statistics()

    result = retriever.retrieve_hybrid("Como configurar o novo produto?")
    assert result.get("retrieval_mode") == "sparse_only"


def test_explicit_synthetic_hashing_opt_in_for_test_fixtures():
    """
    Fixture explicitamente hashing:
    Quando allow_synthetic_hashing=True e todos os chunks são hashing,
    retrieval_mode é synthetic_hashing e a escolha não afeta o default de produção.
    """
    retriever = HybridKnowledgeRetriever(allow_synthetic_hashing=True)
    c1 = IndexedChunk(
        chunk_id="synth_1",
        source_file="synth.md",
        source_url="https://site.getnet.com.br/synth",
        title="Synth Chunk",
        content="Conteúdo sintético para teste homogêneo de hashing.",
        embedding_backend="synthetic_hashing",
    )
    retriever.chunks = [c1]
    retriever._build_bm25_statistics()

    result = retriever.retrieve_hybrid("Conteúdo sintético")
    assert result.get("retrieval_mode") == "synthetic_hashing"

    # Default de outro retriever sem opt-in permanece isolado
    default_r = HybridKnowledgeRetriever()
    assert default_r.allow_synthetic_hashing is False


def test_cache_mutation_after_indexing_does_not_change_stored_provenance():
    """
    Mutação ou reload de cache após indexação:
    A proveniência do vetor já armazenado no IndexedChunk não muda só porque uma chave entrou/saiu do cache.
    """
    retriever = HybridKnowledgeRetriever()
    initial_backend = retriever.chunks[0].embedding_backend
    assert initial_backend == "neural"

    # Esvazia o mapa de cache global
    SemanticEmbedder._neural_vectors = {}

    # O objeto já indexado preserva sua proveniência imutável
    assert retriever.chunks[0].embedding_backend == "neural"


def test_cache_validation_edge_cases_and_atomic_preservation():
    """
    F3: Validação de cache estrita e publicação atômica.
    Rejeita NaN, Infinity, booleanos, norma zero, alpha inválido e não altera mapa global no erro.
    """
    def minimal_payload():
        return {
            "model": SemanticEmbedder.EMBEDDING_MODEL,
            "dimension": SemanticEmbedder.DIM,
            "vectors": {"test_vec": [1.0 / math.sqrt(SemanticEmbedder.DIM)] * SemanticEmbedder.DIM},
            "mean_unit": [1.0 / math.sqrt(SemanticEmbedder.DIM)] * SemanticEmbedder.DIM,
            "alpha": 0.68,
        }

    # 1. Payload válido aceito
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(minimal_payload())) is True

    # 2. Snapshot antes da rejeição
    before_vectors = copy.deepcopy(SemanticEmbedder._neural_vectors)
    before_mean = list(SemanticEmbedder._mean_unit)
    before_alpha = SemanticEmbedder._alpha

    # 3. Rejeita NaN em vetor
    p_nan_vec = minimal_payload()
    p_nan_vec["vectors"]["test_vec"][0] = float("nan")
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_nan_vec)) is False
    assert SemanticEmbedder._neural_vectors == before_vectors

    # 4. Rejeita booleano em vetor (bool subclass de int)
    p_bool_vec = minimal_payload()
    p_bool_vec["vectors"]["test_vec"][0] = True
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_bool_vec)) is False
    assert SemanticEmbedder._neural_vectors == before_vectors

    # 5. Rejeita vetor de norma zero
    p_zero_vec = minimal_payload()
    p_zero_vec["vectors"]["test_vec"] = [0.0] * SemanticEmbedder.DIM
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_zero_vec)) is False

    # 6. Rejeita NaN em mean_unit
    p_nan_mean = minimal_payload()
    p_nan_mean["mean_unit"][0] = float("nan")
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_nan_mean)) is False

    # 7. Rejeita Infinity em alpha
    p_inf_alpha = minimal_payload()
    p_inf_alpha["alpha"] = float("inf")
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_inf_alpha)) is False

    # 8. Rejeita alpha fora do intervalo [0, 1]
    p_bad_alpha = minimal_payload()
    p_bad_alpha["alpha"] = 1.5
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_bad_alpha)) is False

    # 9. Rejeita string para alpha e preserva mapa anterior sem mutação parcial
    p_str_alpha = minimal_payload()
    p_str_alpha["alpha"] = "not-a-number"
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_str_alpha)) is False
    assert SemanticEmbedder._neural_vectors == before_vectors
    assert SemanticEmbedder._mean_unit == before_mean
    assert SemanticEmbedder._alpha == before_alpha

    # 10. Rejeita modelo errado
    p_bad_model = minimal_payload()
    p_bad_model["model"] = "models/wrong-embedding-model"
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_bad_model)) is False

    # 11. Rejeita dimensão errada
    p_bad_dim = minimal_payload()
    p_bad_dim["dimension"] = 128
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_bad_dim)) is False

    # 12. Rejeita vetor curto
    p_short_vec = minimal_payload()
    p_short_vec["vectors"]["test_vec"] = [0.1] * 10
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(p_short_vec)) is False

    # 13. Rejeita JSON truncado ou malformado
    assert SemanticEmbedder.validate_and_load_cache_payload("{\"model\": \"incomplete") is False
    assert SemanticEmbedder.validate_and_load_cache_payload("") is False


def test_live_embedding_boundary_timeout_and_output_validation(monkeypatch):
    """
    F6: Fronteira live com timeout configurado, validação de dimensão/finitude e degradação segura.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-key-live")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    provider_calls = []
    def fake_valid_client(**kwargs):
        provider_calls.append(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(
                embed_content=lambda **kw: SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[0.5] * SemanticEmbedder.DIM)]
                )
            )
        )

    # 1. Sucesso: chave de settings usada e timeout explícito em http_options
    with patch.object(genai, "Client", side_effect=fake_valid_client):
        vec, mode = SemanticEmbedder.embed_query("Consulta inédita para teste live")
    assert mode == "neural_live"
    assert vec is not None
    assert len(vec) == 384
    assert provider_calls[0].get("api_key") == "synthetic-key-live"
    assert "http_options" in provider_calls[0]
    assert "timeout" in provider_calls[0]["http_options"]

    # 2. Resposta com dimensão inválida (2d) é rejeitada e degrada para sparse_only
    def fake_wrong_dim(**kwargs):
        return SimpleNamespace(
            models=SimpleNamespace(
                embed_content=lambda **kw: SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[0.5, 0.5])]
                )
            )
        )
    with patch.object(genai, "Client", side_effect=fake_wrong_dim):
        bad_v, bad_m = SemanticEmbedder.embed_query("Consulta dimensão incorreta")
    assert bad_v is None
    assert bad_m == "sparse_only"

    # 3. Falha do provedor (429/timeout) degrada para sparse_only sem quebrar
    def fake_fail(**kwargs):
        raise RuntimeError("Synthetic 429 quota exceeded")
    with patch.object(genai, "Client", side_effect=fake_fail):
        fail_v, fail_m = SemanticEmbedder.embed_query("Consulta com erro de rede")
    assert fail_v is None
    assert fail_m == "sparse_only"

    # 4. Resposta com lista vazia de embeddings
    def fake_empty_list(**kwargs):
        return SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kw: SimpleNamespace(embeddings=[])))
    with patch.object(genai, "Client", side_effect=fake_empty_list):
        empty_v, empty_m = SemanticEmbedder.embed_query("Consulta lista vazia")
    assert empty_v is None
    assert empty_m == "sparse_only"

    # 5. Resposta com NaN ou valor não-finito
    def fake_nan_resp(**kwargs):
        return SimpleNamespace(
            models=SimpleNamespace(
                embed_content=lambda **kw: SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[float("nan")] * SemanticEmbedder.DIM)]
                )
            )
        )
    with patch.object(genai, "Client", side_effect=fake_nan_resp):
        nan_v, nan_m = SemanticEmbedder.embed_query("Consulta com NaN")
    assert nan_v is None
    assert nan_m == "sparse_only"

    # 6. Resposta com norma zero
    def fake_zero_norm(**kwargs):
        return SimpleNamespace(
            models=SimpleNamespace(
                embed_content=lambda **kw: SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[0.0] * SemanticEmbedder.DIM)]
                )
            )
        )
    with patch.object(genai, "Client", side_effect=fake_zero_norm):
        zero_v, zero_m = SemanticEmbedder.embed_query("Consulta norma zero")
    assert zero_v is None
    assert zero_m == "sparse_only"


def test_cached_neural_query_with_uncached_docs_never_compares_neural_with_hashing():
    """
    Query neural cacheada + documentos sem vetor:
    Garante que nunca calcula cosseno entre query neural e documentos sem vetor (degrada para sparse-only).
    """
    retriever = HybridKnowledgeRetriever()
    uncached_chunk_1 = IndexedChunk(
        chunk_id="uncached_doc_a",
        source_file="new_a.md",
        source_url="https://site.getnet.com.br/novo-a",
        title="Documento Novo A",
        content="Conteúdo novo sobre taxas de crediário sem vetor neural.",
    )
    uncached_chunk_2 = IndexedChunk(
        chunk_id="uncached_doc_b",
        source_file="new_b.md",
        source_url="https://site.getnet.com.br/novo-b",
        title="Documento Novo B",
        content="Outro conteúdo novo sobre prazos de crediário sem vetor neural.",
    )
    retriever.chunks = [uncached_chunk_1, uncached_chunk_2]
    retriever._build_bm25_statistics()

    # Query que existe no cache neural pré-calculado
    known_neural_query = "What's the difference between the Get Clássica and the Get Smart?"
    assert SemanticEmbedder.embed_query(known_neural_query)[0] is not None

    cosine_calls = []
    orig_cosine = SemanticEmbedder.cosine_similarity
    def capture_cosine(a, b):
        cosine_calls.append((a, b))
        return orig_cosine(a, b)

    with patch.object(SemanticEmbedder, "cosine_similarity", side_effect=capture_cosine):
        hybrid_res = retriever.retrieve_hybrid(known_neural_query)
        cand_res = retriever.retrieve_candidates(known_neural_query)

    assert hybrid_res["retrieval_mode"] == "sparse_only"
    assert cand_res["retrieval_mode"] == "sparse_only"
    assert len(cosine_calls) == 0


def test_router_and_knowledge_consumers_use_unified_retrieval_contract():
    """
    Uso nos consumidores reais de Router e KnowledgeAgent:
    Ambos usam retrieve_candidates sob o contrato unificado, sem chamada oculta a fallback de hashing.
    """
    from app.agents.router_agent import RouterAgent
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.schemas import RoutingPlan, PlanStep

    # 1. RouterAgent: _prefer_local_documentary_evidence
    plan = RoutingPlan(
        primary_agent="WEB_SEARCH_AGENT",
        confidence=0.9,
        is_multi_step=False,
        reasoning="Consulta teste",
        plan_steps=[PlanStep(step_order=1, target_agent="WEB_SEARCH_AGENT", sub_query="Como funciona Pix Getnet?", rationale="busca web", request_unit_ids=["u1"])],
    )
    revised = RouterAgent._prefer_local_documentary_evidence(plan, "Como funciona Pix Getnet?")
    assert revised is not None
    # Deve preferir a evidência documental local de Getnet
    assert revised.plan_steps[0].target_agent == "KNOWLEDGE_AGENT"

    # 2. KnowledgeAgent: fallback de evidência documental local
    from app.schemas import ToolCallSpec
    mock_call = ToolCallSpec(
        tool_name="web_search_general",
        arguments={"query": "Pix Getnet conta elegibilidade"},
        call_id="call_mock_1",
    )
    resolved_calls = KnowledgeAgent._prefer_local_pix_product_evidence(
        calls=[mock_call],
        original_request="Pix Getnet conta elegibilidade",
        scope={"assigned_unit_ids": ["u1"]},
    )
    assert len(resolved_calls) == 1
    assert resolved_calls[0].tool_name == "search_getnet_knowledge"


def test_offline_suite_does_not_call_genai_client_accidentally_even_with_key_in_settings(monkeypatch):
    """
    Suíte offline e import/startup não chamam o SDK Gemini por acidente,
    mesmo que settings.GEMINI_API_KEY contenha um valor, desde que GETNET_LIVE_EMBEDDINGS=False.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "accidental-key-in-env")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", False)

    genai_calls = []
    def spy_client(**kwargs):
        genai_calls.append(kwargs)
        raise AssertionError("genai.Client was called in offline mode!")

    with patch.object(genai, "Client", side_effect=spy_client):
        # Query inédita sem live habilitado deve degradar para sparse_only sem instanciar o cliente do SDK
        vec, mode = SemanticEmbedder.embed_query("Query que jamais deve chamar SDK offline")

    assert vec is None
    assert mode == "sparse_only"
    assert len(genai_calls) == 0


def test_r1_single_remote_embedding_attempt_on_timeout(monkeypatch):
    """
    R1: Orçamento único e bounded por operação.
    Em caso de timeout ou falha remota, realiza no máximo 1 tentativa no provedor,
    com retry desabilitado (attempts=1) e sem encadear tentativas com outras variantes.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    sdk_calls = []
    def timeout_client(**kwargs):
        def embed(**kw):
            sdk_calls.append({"client_options": kwargs.get("http_options"), "contents": kw.get("contents")})
            raise TimeoutError("Synthetic provider timeout")
        return SimpleNamespace(models=SimpleNamespace(embed_content=embed))

    with patch.object(genai, "Client", side_effect=timeout_client):
        res = search_getnet_knowledge("Pergunta completamente inédita R1 para teste de timeout único 2026")

    assert len(sdk_calls) == 1
    assert sdk_calls[0]["client_options"]["retry_options"] == {"attempts": 1}
    assert res["retrieval_mode"] == "sparse_only"
    assert res["abstained"] is True


def test_r1_zero_remote_calls_when_corpus_has_non_neural_chunks(monkeypatch):
    """
    R1: Proteção contra chamadas inúteis.
    Se o corpus contém chunks não-neurais ou mistos, zero chamadas remotas são disparadas.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    retriever = HybridKnowledgeRetriever()
    uncached = IndexedChunk("cid_x", "file.md", "https://site.getnet.com.br", "Novo", "Conteúdo inédito sem vetor")
    retriever.chunks = [uncached]
    retriever._build_bm25_statistics()

    calls = []
    def client_spy(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kw: None))

    with patch.object(genai, "Client", side_effect=client_spy):
        res = retriever.retrieve_candidates("Query qualquer sobre documento não neural")

    assert len(calls) == 0
    assert res["retrieval_mode"] == "sparse_only"


def test_r2_strict_unit_norm_validation_vectors_and_mean():
    """
    R2: Validação estrita de norma L2 em todos os pontos de entrada do cache.
    Rejeita vetores e mean_unit com norma fora de abs(norm - 1.0) <= 1e-4, overflow ou NaN.
    """
    u = 1.0 / math.sqrt(384)
    valid_payload = {
        "model": SemanticEmbedder.EMBEDDING_MODEL,
        "dimension": 384,
        "vectors": {"doc_1": [u] * 384},
        "mean_unit": [u] * 384,
        "alpha": 0.68,
    }

    # 1. Payload válido aceito
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(valid_payload)) is True

    # 2. Vetor com norma != 1.0 rejeitado
    bad_vec_payload = copy.deepcopy(valid_payload)
    bad_vec_payload["vectors"]["doc_1"] = [2.0] * 384
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(bad_vec_payload)) is False

    # 3. mean_unit com norma != 1.0 rejeitado
    bad_mean_payload = copy.deepcopy(valid_payload)
    bad_mean_payload["mean_unit"] = [0.5] * 384
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(bad_mean_payload)) is False

    # 4. Overflow na soma de quadrados tratado com segurança
    overflow_payload = copy.deepcopy(valid_payload)
    overflow_payload["vectors"]["doc_1"] = [1e308] * 384
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(overflow_payload)) is False


def test_r2_live_embedding_accepts_nonunit_raw_then_validates_unit_norm_transformed(monkeypatch):
    """
    R2: A resposta crua da SDK do Gemini pode não ter norma unitária antes do pós-processamento,
    mas o vetor final transformado deve satisfazer abs(norm - 1.0) <= 1e-4.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(settings, "GETNET_LIVE_EMBEDDINGS", True)

    raw_values = [0.5] * 384  # Norma crua = sqrt(384 * 0.25) ~ 9.798 (não unitária)
    mock_resp = SimpleNamespace(embeddings=[SimpleNamespace(values=raw_values)])
    mock_client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kw: mock_resp))

    with patch.object(genai, "Client", return_value=mock_client):
        vec, mode = SemanticEmbedder.embed_query_live("Query R2 para teste de resposta crua não unitária")

    assert mode == "neural_live"
    assert vec is not None
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) <= 1e-4


def test_r3_reload_transformation_safely_degrades_without_invalid_cosine(monkeypatch):
    """
    R3: Rastreabilidade e compatibilidade de transformações no cache.
    Se o cache for recarregado com novo alpha ou mean_unit, chunks com fingerprint antigo
    não calculam cosseno contra query com novo fingerprint e degradam para sparse_only.
    """
    retriever = HybridKnowledgeRetriever()
    old_fp = retriever.chunks[0].transformation_fingerprint

    # Recarrega payload alterando alpha para 0.0 (fingerprint diferente)
    reload_payload = {
        "model": SemanticEmbedder.EMBEDDING_MODEL,
        "dimension": 384,
        "vectors": {},
        "mean_unit": list(SemanticEmbedder._mean_unit),
        "alpha": 0.0,
    }
    assert SemanticEmbedder.validate_and_load_cache_payload(json.dumps(reload_payload)) is True
    new_fp = SemanticEmbedder.current_transformation_fingerprint()
    assert old_fp != new_fp

    cosine_calls = []
    def spy_cosine(a, b):
        cosine_calls.append((a, b))
        return SemanticEmbedder.cosine_similarity(a, b)

    with patch.object(SemanticEmbedder, "cosine_similarity", side_effect=spy_cosine):
        res = retriever.retrieve_candidates("Consulta R3 pós recarga de transformação")

    assert res["retrieval_mode"] == "sparse_only"
    assert len(cosine_calls) == 0


def test_dynamic_retrieval_method_in_candidates_and_search_tool():
    """
    Ajuste menor: retrieval_method dinâmico alinhado ao modo real de recuperação.
    """
    retriever = HybridKnowledgeRetriever()

    # 1. Modo neural ativo
    res_neural = retriever.retrieve_candidates("What's the difference between the Get Clássica and the Get Smart?")
    assert res_neural["retrieval_method"] == "bm25_neural_rrf_candidates"

    # 2. Modo sparse_only ativo
    res_sparse = retriever.retrieve_candidates("Pergunta inédita sem vetor")
    assert res_sparse["retrieval_method"] == "bm25_sparse_only_candidates"


def test_deprecations_propagated_in_metrics_and_cli():
    """
    Ajuste menor: metadados de depreciação propagados em metrics do report.
    """
    from app.evals.evaluator import evaluator
    report = evaluator.run_suite()
    metrics = report["metrics"]
    assert "deprecations" in metrics
    deprecations = metrics["deprecations"]
    assert "retrieval_recall_at_1" in deprecations
    assert "retrieval_recall_at_3" in deprecations
    assert "retrieval_recall_at_5" in deprecations
    assert "retrieval_mrr" in deprecations
    assert deprecations["retrieval_recall_at_3"]["replacement"] == "hit_at_3"
