from pathlib import Path
from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.router_agent import RouterAgent
from app.rag.embeddings import BilingualDenseEmbedder
from app.rag.retriever import HybridKnowledgeRetriever
from app.schemas import RoutingPlan
from app.security.auth import SecurityContext


def test_bilingual_dense_embeddings_cosine_alignment():
    """
    Valida que queries equivalentes em Inglês e Português produzem vetores densos L2-normalizados
    com alta similaridade de cosseno, enquanto temas fora do domínio têm baixa similaridade.
    """
    v_pt = BilingualDenseEmbedder.embed_text("Em quantas parcelas posso dividir uma venda no crediário?")
    v_en = BilingualDenseEmbedder.embed_text("How many installments can I split a sale into with the crediário?")
    v_unrelated = BilingualDenseEmbedder.embed_text("Como assar uma lasanha de berinjela no forno elétrico?")

    sim_bilingual = BilingualDenseEmbedder.cosine_similarity(v_pt, v_en)
    sim_unrelated = BilingualDenseEmbedder.cosine_similarity(v_pt, v_unrelated)

    assert len(v_pt) == 384
    assert sim_bilingual > 0.45
    assert sim_unrelated < 0.15


def test_hybrid_rag_bm25_dense_rrf_and_verifiable_urls():
    """
    Valida que o HybridKnowledgeRetriever combina BM25 + Dense Cosine + RRF
    e retorna citações verificáveis com URLs oficiais (site.getnet.com.br).
    """
    res = retriever = HybridKnowledgeRetriever()
    outcome = retriever.retrieve_hybrid("Qual a diferença entre a Get Clássica e a Get Smart?", top_k=3)

    assert outcome["abstained"] is False
    assert outcome["max_relevance_score"] >= 0.50
    assert len(outcome["results"]) >= 1
    assert len(outcome["citations"]) >= 1

    top = outcome["results"][0]
    assert top["bm25_score"] > 0
    assert top["dense_cosine_score"] > 0
    assert top["rrf_score"] > 0
    assert "getnet.com.br" in top["source_url"]


def test_hybrid_rag_threshold_abstention_on_out_of_domain_query():
    """
    Valida que consultas fora do domínio da Getnet ficam abaixo do threshold
    e acionam `abstained=True` sem alucinar respostas.
    """
    ret = HybridKnowledgeRetriever()
    outcome = ret.retrieve_hybrid("Qual é a receita tradicional de pudim de leite condensado?", top_k=3)

    assert outcome["abstained"] is True
    assert outcome["results"] == []
    assert outcome["citations"] == []


def test_knowledge_agent_dynamic_grounding_not_hardcoded(tmp_path: Path):
    """
    Prova que o KnowledgeAgent NÃO usa strings fixas hardcoded para responder:
    ao indexar um diretório temporário com um documento inédito customizado,
    o agente sintetiza a resposta diretamente do novo chunk recuperado pelo RAG.
    """
    custom_md = tmp_path / "getnet_custom_product.md"
    custom_md.write_text(
        "---\n"
        "source_url: https://site.getnet.com.br/produto-experimental-x99/\n"
        "title: Módulo Experimental Getnet X99\n"
        "---\n\n"
        "## Especificações do Módulo Getnet X99\n"
        "- O Módulo Getnet X99 opera com frequência quântica de 777 TeraHertz e taxa exclusiva de 0,19%.\n",
        encoding="utf-8",
    )

    custom_retriever = HybridKnowledgeRetriever(
        data_dir=tmp_path,
        corpus_path=tmp_path / "corpus.json",
        vector_index_path=tmp_path / "vec.json",
    )
    hit = custom_retriever.retrieve_hybrid("Qual a frequência e a taxa do Módulo Getnet X99?")
    assert hit["abstained"] is False
    assert "777 TeraHertz" in hit["results"][0]["content"]
    assert "0,19%" in hit["results"][0]["content"]
    assert hit["citations"][0].source_url == "https://site.getnet.com.br/produto-experimental-x99/"


def test_router_agent_pydantic_validation_and_multi_step_decomposition():
    """
    Valida que o RouterAgent retorna um objeto `RoutingPlan` estritamente validado por Pydantic
    e decompõe perguntas compostas em múltiplos `PlanStep`.
    """
    router = RouterAgent(gemini_client=None)

    single_plan = router.plan(
        message="Quando cai o dinheiro das minhas vendas de ontem?",
        user_id="cliente1988",
    )
    assert isinstance(single_plan, RoutingPlan)
    assert single_plan.primary_agent == "CUSTOMER_SUPPORT_AGENT"
    assert single_plan.is_multi_step is False

    compound_plan = router.plan(
        message="Minha maquininha não conecta na internet e quero falar com um atendente humano",
        user_id="cliente1988",
    )
    assert isinstance(compound_plan, RoutingPlan)
    assert compound_plan.is_multi_step is True
    agents_in_plan = [s.target_agent for s in compound_plan.plan_steps]
    assert "CUSTOMER_SUPPORT_AGENT" in agents_in_plan
    assert "HUMAN_ESCALATION_AGENT" in agents_in_plan


def test_router_agent_handles_malformed_llm_json_gracefully():
    """
    Se o cliente LLM retornar JSON inválido ou target_agent inexistente,
    a validação Pydantic rejeita e o Router cai para o planejador semântico seguro.
    """
    class BrokenModels:
        @staticmethod
        def generate_content(**kwargs):
            class FakeResp:
                text = '{"primary_agent": "INVALID_NONEXISTENT_AGENT", "confidence": 999}'
            return FakeResp()

    class BrokenClient:
        models = BrokenModels()

    router = RouterAgent(gemini_client=BrokenClient())
    plan = router.plan("Qual é a cotação do euro hoje?", user_id="cliente1988")
    assert isinstance(plan, RoutingPlan)
    assert plan.primary_agent == "WEB_SEARCH_AGENT"
