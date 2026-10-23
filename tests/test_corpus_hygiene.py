import hashlib
import json
from pathlib import Path
import re

import httpx

from app.rag.embeddings import BilingualDenseEmbedder
from app.rag.ingestor import GetnetCorpusIngestor
from app.rag.retriever import HybridKnowledgeRetriever


ROOT = Path(__file__).resolve().parents[1]
NOISE = re.compile(
    r"nos informe cpf|\bSAC\b|\b0800[\s-]*\d|\b400[23][\s-]*4000\b|"
    r"black friday|cookies?|como voc[eê] prefere continuar|como podemos te ajudar|"
    r"selecione para ver as taxas|pol[ií]tica de privacidade",
    re.IGNORECASE,
)


def test_clean_content_removes_boilerplate_and_keeps_product_facts():
    source = "\n".join(
        [
            "## Crediário Getnet",
            "- Nos informe CPF ou CNPJ",
            "- Capitais e Regiões Metropolitanas 4002-4000 ou 4003-4000",
            "- O Crediário parcela em até 48x e paga ao lojista em até 2 dias úteis.",
            "- Consulte a habilitação no app ou na Central (`4002-4000` / `0800-648-8000`).",
            "## Acelere suas vendas na Black Friday",
            "- Selecione para ver as taxas",
            "- Aceitar todos os cookies",
            "## Como você prefere continuar?",
            "- Ao clicar em enviar, você concorda com a Política de Privacidade.",
            "- Pix aceita chave CPF, CNPJ, e-mail ou chave aleatória.",
        ]
    )

    cleaned = GetnetCorpusIngestor.clean_content(source)

    assert not NOISE.search(cleaned)
    assert "até 48x" in cleaned
    assert "até 2 dias úteis" in cleaned
    assert "Consulte a habilitação no app" in cleaned
    assert "chave CPF, CNPJ" in cleaned
    assert GetnetCorpusIngestor.clean_content(cleaned) == cleaned


def test_inline_contact_labels_do_not_discard_support_instructions():
    source = (
        "5. **Acionamento do Suporte Técnico:**\n"
        "- Caso o erro persista após validar chip e Wi-Fi, acione o suporte técnico "
        "(Capitais: `4002-4000` ou `4003-4000` / Demais localidades: `0800-648-8000`) "
        "ou solicite troca remota de perfil de operadora / substituição de terminal."
    )

    cleaned = GetnetCorpusIngestor.clean_content(source)

    assert "Caso o erro persista" in cleaned
    assert "substituição de terminal" in cleaned
    assert not NOISE.search(cleaned)


def test_live_ingestion_excludes_structural_navigation_and_flushes_short_tail(tmp_path, monkeypatch):
    html = """
    <html><header><p>Menu institucional para pessoas e empresas</p></header>
    <main><nav><div><p>Menu de contratação de produtos</p></div></nav>
    <form><p>Nos informe CPF ou CNPJ</p><button>Enviar formulário</button></form>
    <section class="cookie-banner"><p>Gerencie os seus cookies e sua privacidade</p></section>
    <h1>Crediário Getnet</h1><p>Parcele em até 48x, conforme o emissor do cartão.</p>
    <p>Receba em até 2 dias úteis com taxa única.</p></main>
    <footer><p>SAC: 0800-648-8000</p></footer></html>
    """
    url = "https://site.getnet.com.br/crediario/"
    monkeypatch.setattr(GetnetCorpusIngestor, "OFFICIAL_GETNET_URLS", [{"url": url, "topic": "Crediário"}])
    monkeypatch.setattr(
        httpx.Client, "get", lambda self, requested: httpx.Response(200, text=html, request=httpx.Request("GET", requested))
    )
    ingestor = GetnetCorpusIngestor(data_dir=tmp_path, corpus_path=tmp_path / "corpus.json")

    chunks = ingestor.scrape_live_getnet_pages()

    assert chunks, "A useful final window shorter than 650 characters must be indexed."
    content = "\n".join(chunk["content"] for chunk in chunks)
    assert not NOISE.search(content)
    assert "Menu institucional" not in content
    assert "Menu de contratação" not in content
    assert "até 48x" in content and "até 2 dias úteis" in content
    assert all(chunk["source_url"] == url for chunk in chunks)


def test_markdown_ingestion_cleans_contact_noise_and_records_exact_sha256(tmp_path):
    url = "https://site.getnet.com.br/crediario/"
    (tmp_path / "product.md").write_text(
        f"---\nsource_url: {url}\ntitle: Crediário\n---\n"
        "## Como habilitar\nConsulte a habilitação no aplicativo. Central: 4002-4000.\n",
        encoding="utf-8",
    )
    ingestor = GetnetCorpusIngestor(data_dir=tmp_path, corpus_path=tmp_path / "corpus.json")

    chunks = ingestor.build_and_persist_corpus(run_live_scrape=False)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "habilitação no aplicativo" in chunk["content"]
    assert not NOISE.search(chunk["content"])
    assert chunk["source_url"] == url
    assert chunk["content_hash"] == hashlib.sha256(chunk["content"].encode()).hexdigest()


def test_versioned_corpus_and_vector_index_are_clean_consistent_and_reproducible(tmp_path):
    payload = json.loads((ROOT / "data/getnet_corpus.json").read_text(encoding="utf-8"))
    vectors = json.loads((ROOT / "data/vector_index.json").read_text(encoding="utf-8"))
    chunks = payload["chunks"]

    assert payload["total_chunks"] == len(chunks)
    assert payload["markdown_chunks_count"] == sum(c["origin_type"] == "verified_markdown_mirror" for c in chunks)
    assert payload["live_web_chunks_count"] == sum(c["origin_type"] == "live_web_scrape" for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert set(vectors) == {c["chunk_id"] for c in chunks}
    for chunk in chunks:
        assert not NOISE.search(chunk["content"]), chunk["chunk_id"]
        assert chunk["content_hash"] == hashlib.sha256(chunk["content"].encode()).hexdigest()
        assert chunk["source_url"].startswith(("https://site.getnet.com.br/", "https://www.getnet.net/"))
        expected = BilingualDenseEmbedder.embed_text(f'{chunk["title"]} {chunk["title"]} {chunk["content"]}')
        assert vectors[chunk["chunk_id"]] == expected

    target = tmp_path / "corpus.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    ingestor = GetnetCorpusIngestor(data_dir=ROOT / "data", corpus_path=target)
    ingestor.clean_existing_corpus()
    first = json.loads(target.read_text(encoding="utf-8"))
    ingestor.clean_existing_corpus()
    second = json.loads(target.read_text(encoding="utf-8"))
    assert first == second
    assert first["chunks"] == chunks


def test_retriever_rejects_stale_cached_vector_for_changed_content(tmp_path):
    (tmp_path / "product.md").write_text(
        "---\nsource_url: https://site.getnet.com.br/crediario/\ntitle: Crediário\n---\n"
        "## Crediário Getnet\nParcele em até 48x e consulte a taxa no aplicativo.\n",
        encoding="utf-8",
    )
    corpus_path = tmp_path / "corpus.json"
    chunks = GetnetCorpusIngestor(data_dir=tmp_path, corpus_path=corpus_path).build_and_persist_corpus(False)
    index_path = tmp_path / "vectors.json"
    index_path.write_text(json.dumps({chunks[0]["chunk_id"]: [0.0] * 384}), encoding="utf-8")

    retriever = HybridKnowledgeRetriever(data_dir=tmp_path, corpus_path=corpus_path, vector_index_path=index_path)

    expected = BilingualDenseEmbedder.embed_text(f'{chunks[0]["title"]} {chunks[0]["title"]} {chunks[0]["content"]}')
    assert retriever.chunks[0].embedding == expected
    assert json.loads(index_path.read_text())[chunks[0]["chunk_id"]] == expected
