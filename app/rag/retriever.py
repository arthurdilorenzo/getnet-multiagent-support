from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import replace
from urllib.parse import urlsplit
from app.config import settings
from app.observability.logger import logger
from app.rag.embeddings import BilingualDenseEmbedder
from app.rag.ingestor import GetnetCorpusIngestor
from app.rag.answerability import assess, expanded_query, parse_requirements
from app.rag.facts import FactCatalog, extract_facts
from app.schemas import Citation


class IndexedChunk:
    def __init__(
        self,
        chunk_id: str,
        source_file: str,
        source_url: str,
        title: str,
        content: str,
        origin_type: str = "verified_markdown_mirror",
        embedding: Optional[List[float]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.source_file = source_file
        self.source_url = source_url
        self.title = title
        self.content = content
        self.origin_type = origin_type
        self.provenance = provenance or {}
        # Texto indexado dá peso 2x ao título
        full_index_text = f"{title} {title} {content}"
        self.tokens: List[str] = BilingualDenseEmbedder.tokenize(full_index_text)
        self.term_freqs: Counter = Counter(self.tokens)
        self.doc_len: int = max(1, len(self.tokens))
        self.embedding: List[float] = embedding or BilingualDenseEmbedder.embed_text(full_index_text)


class HybridKnowledgeRetriever:
    """
    Retrieval híbrido local da Getnet (sem encoder neural):
    1. Sparse Retrieval via algoritmo Okapi BM25 (k1=1.5, b=0.75) com expansão léxica bilíngue PT/EN.
    2. Projeção por hashing de n-gramas e âncoras conceituais manuais em 384 coordenadas, com cosseno.
    3. Score heurístico lexical/cosseno [0, 1], usando Reciprocal Rank Fusion (RRF) para desempate.
    4. Threshold de Relevância e Abstention explícita para consultas fora de domínio.
    5. Citações verificáveis contendo `chunk_id`, `source_file` e `source_url` oficial da Getnet.
    """

    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 20

    # Dicionário de tradução de termos técnicos EN -> PT para paridade de BM25 esparso
    BILINGUAL_EXPANSION: Dict[str, List[str]] = {
        "difference": ["diferenca", "comparativo"],
        "between": ["entre"],
        "sales": ["vendas"],
        "yesterday": ["ontem"],
        "deposited": ["depositado", "liquidacao", "receber"],
        "bank": ["banco", "bancaria", "domicilio"],
        "account": ["conta", "get", "corrente"],
        "receive": ["receber", "recebimento"],
        "machine": ["maquininha", "maquina", "terminal"],
        # SIM is the English acronym used on some terminal manuals; the
        # indexed Getnet article uses the Portuguese term "chip".
        "sim": ["chip"],
        "connect": ["conecta", "conexao", "sinal", "rede"],
        "internet": ["internet", "wifi", "chip", "sinal"],
        "receivables": ["recebiveis", "vendas"],
        "advance": ["antecipacao", "adiantar", "receba"],
        "decline": ["recusada", "recusa", "erro", "codigo"],
        "transaction": ["transacao", "venda"],
        "error": ["erro", "codigo", "recusada"],
        "installments": ["parcelas", "parcelar", "parcelamento", "vezes"],
        "split": ["parcelar", "dividir", "parcelas"],
        "payment": ["pagamento"],
        "link": ["link", "whatsapp"],
        "sell": ["vender", "vendas"],
    }

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
        vector_index_path: Optional[Path] = None,
        min_relevance_threshold: float = settings.RAG_MIN_RELEVANCE_THRESHOLD,
        fact_catalog_path: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.corpus_path = Path(corpus_path or settings.CORPUS_JSON_PATH)
        self.vector_index_path = Path(vector_index_path or settings.VECTOR_INDEX_PATH)
        self.min_relevance_threshold = min_relevance_threshold
        self.fact_catalog_path = Path(fact_catalog_path or self.data_dir / 'knowledge_facts.json')
        self.fact_catalog = FactCatalog.load(self.fact_catalog_path)
        self.catalog_required = self.fact_catalog_path.exists()
        self.chunks: List[IndexedChunk] = []
        self.doc_freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 1.0
        self._load_and_index()

    def _load_and_index(self) -> None:
        ingestor = GetnetCorpusIngestor(data_dir=self.data_dir, corpus_path=self.corpus_path)
        raw_chunks: List[Dict[str, Any]] = []

        if self.corpus_path.exists():
            try:
                data = json.loads(self.corpus_path.read_text(encoding="utf-8"))
                raw_chunks = data.get("chunks", [])
            except Exception as exc:
                logger.warning("rag_corpus_json_read_failed", error=str(exc))

        # Garante que os arquivos Markdown mais recentes estejam sincronizados
        md_chunks = ingestor.load_markdown_chunks()
        if not raw_chunks:
            raw_chunks = md_chunks
        else:
            web_only = [c for c in raw_chunks if c.get("origin_type") == "live_web_scrape"]
            raw_chunks = md_chunks + web_only
        raw_chunks = [cleaned for item in raw_chunks if (cleaned := ingestor.clean_chunk(item))]

        self.chunks = []
        updated_vectors: Dict[str, List[float]] = {}
        for item in raw_chunks:
            cid = item["chunk_id"]
            idx_chunk = IndexedChunk(
                chunk_id=cid,
                source_file=item.get("source_file", "getnet_docs.md"),
                source_url=item.get("source_url", "https://site.getnet.com.br/"),
                title=item.get("title", ""),
                content=item.get("content", ""),
                origin_type=item.get("origin_type", "verified_markdown_mirror"),
                provenance={key:item.get(key) for key in ('verification_status','verified_at','review_method','source_location','source_access_note','secondary_url','additional_url')},
            )
            self.chunks.append(idx_chunk)
            updated_vectors[cid] = idx_chunk.embedding

        # A projeção local é barata: regenerar evita aceitar cache obsoleto ou adulterado.
        # O arquivo persistido é um artefato inspecionável, não uma fonte de verdade.
        try:
            self.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_index_path.write_text(
                json.dumps(updated_vectors, separators=(",", ":")), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("rag_vector_index_write_failed", error=str(exc))

        self._build_bm25_statistics()

    def _build_bm25_statistics(self) -> None:
        self.doc_freqs.clear()
        self.idf.clear()
        n_docs = len(self.chunks)
        if n_docs == 0:
            self.avgdl = 1.0
            return

        total_len = 0
        for chunk in self.chunks:
            total_len += chunk.doc_len
            for unique_token in set(chunk.tokens):
                self.doc_freqs[unique_token] += 1

        self.avgdl = total_len / n_docs
        for term, df in self.doc_freqs.items():
            # Fórmula padrão Okapi BM25 IDF (sempre positiva com +1)
            self.idf[term] = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

    def _expand_query_tokens(self, query: str) -> List[str]:
        base_tokens = BilingualDenseEmbedder.tokenize(query)
        expanded = list(base_tokens)
        for tok in base_tokens:
            if tok in self.BILINGUAL_EXPANSION:
                expanded.extend(self.BILINGUAL_EXPANSION[tok])
        return expanded

    def _compute_bm25_score(self, query_tokens: List[str], chunk: IndexedChunk) -> float:
        score = 0.0
        for term in query_tokens:
            if term not in chunk.term_freqs:
                continue
            tf = chunk.term_freqs[term]
            idf_val = self.idf.get(term, 0.0)
            numerator = tf * (self.BM25_K1 + 1.0)
            denominator = tf + self.BM25_K1 * (
                1.0 - self.BM25_B + self.BM25_B * (chunk.doc_len / self.avgdl)
            )
            score += idf_val * (numerator / denominator)
        return score

    def retrieve_candidates(self, query: str, top_k: int = 48) -> Dict[str, Any]:
        """Rank raw bodies without the local answerability grammar or threshold.

        Candidate metadata never authorizes publication. A small bounded corpus
        permits broad context, including low-score candidates for novel wording.
        """
        if not self.chunks:
            self._load_and_index()
        tokens=self._expand_query_tokens(expanded_query(query))
        query_vec=BilingualDenseEmbedder.embed_text(' '.join(tokens))
        bm25=[self._compute_bm25_score(tokens,c) for c in self.chunks]
        dense=[BilingualDenseEmbedder.cosine_similarity(query_vec,c.embedding) for c in self.chunks]
        br={idx:rank for rank,idx in enumerate(sorted(range(len(self.chunks)),key=lambda i:bm25[i],reverse=True),1)}
        dr={idx:rank for rank,idx in enumerate(sorted(range(len(self.chunks)),key=lambda i:dense[i],reverse=True),1)}
        results=[]
        for i,chunk in enumerate(self.chunks):
            coverage=min(1.0,len(set(tokens)&set(chunk.tokens))/max(1.5,len(set(tokens))*.45))
            b_norm=bm25[i]/(bm25[i]+4)*coverage if bm25[i]>0 else 0
            score=round(min(1.0,.5*b_norm+.5*dense[i]),4)
            results.append({'chunk_id':chunk.chunk_id,'doc_id':chunk.chunk_id,'content':chunk.content,
                            'content_hash':hashlib.sha256(chunk.content.encode()).hexdigest(),
                            'source':chunk.source_file,'source_url':chunk.source_url,'title':chunk.title,
                            'origin_type':chunk.origin_type,'provenance':chunk.provenance,
                            'bm25_score':round(bm25[i],4),'dense_cosine_score':round(dense[i],4),
                            'rrf_score':round(1/(self.RRF_K+br[i])+1/(self.RRF_K+dr[i]),5),
                            'score':score,'relevance_score':score})
        results.sort(key=lambda item:(item['score'],item['rrf_score']),reverse=True)
        selected=[];chars=0
        for item in results[:max(1,min(top_k,48))]:
            if chars+len(item['content'])>48000: break
            chars+=len(item['content']);selected.append(item)
        return {'candidate_chunks':selected,'candidate_count':len(selected),'candidate_chars':chars,
                'retrieval_method':'bm25_hash_anchors_rrf_candidates_only','answerability_evaluated':False}

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = settings.RAG_TOP_K,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executa BM25 + cosseno de projeções por hashing; RRF desempata o score heurístico.
        Retorna:
        {
          "abstained": bool,
          "abstention_reason": Optional[str],
          "max_relevance_score": float,
          "results": List[Dict],
          "citations": List[Citation]
        }
        """
        if not self.chunks:
            self._load_and_index()

        min_thresh = threshold if threshold is not None else self.min_relevance_threshold
        search_query = expanded_query(query)
        query_tokens = self._expand_query_tokens(search_query)
        if not query_tokens or not self.chunks:
            return {
                "abstained": True,
                "abstention_reason": "Query vazia ou base de conhecimento não carregada.",
                "max_relevance_score": 0.0,
                "results": [],
                "citations": [],
            }

        # Projeta a query e sua expansão léxica PT/EN no mesmo espaço determinístico.
        expanded_query_text = f"{search_query} {' '.join(query_tokens)}"
        query_vec = BilingualDenseEmbedder.embed_text(expanded_query_text)

        bm25_scores: List[Tuple[int, float]] = []
        dense_scores: List[Tuple[int, float]] = []

        for idx, chunk in enumerate(self.chunks):
            b_score = self._compute_bm25_score(query_tokens, chunk)
            d_score = BilingualDenseEmbedder.cosine_similarity(query_vec, chunk.embedding)
            bm25_scores.append((idx, b_score))
            dense_scores.append((idx, d_score))

        # Ordena rankings individuais para RRF
        bm25_sorted = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
        dense_sorted = sorted(dense_scores, key=lambda x: x[1], reverse=True)

        bm25_rank_map = {doc_idx: rank for rank, (doc_idx, _) in enumerate(bm25_sorted, start=1)}
        dense_rank_map = {doc_idx: rank for rank, (doc_idx, _) in enumerate(dense_sorted, start=1)}

        max_bm25 = max((s for _, s in bm25_scores), default=0.0)

        fused_items: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(self.chunks):
            b_raw = bm25_scores[idx][1]
            d_raw = dense_scores[idx][1]

            # Se nem BM25 nem a projeção possuem sinal relevante, descarta.
            if b_raw <= 0.0 and d_raw < 0.12:
                continue

            r_bm25 = bm25_rank_map[idx]
            r_dense = dense_rank_map[idx]
            rrf_score = (1.0 / (self.RRF_K + r_bm25)) + (1.0 / (self.RRF_K + r_dense))

            # Normalização de BM25 saturada ponderada pela cobertura de termos da consulta
            unique_q = set(query_tokens)
            matched_q = sum(1 for t in unique_q if t in chunk.term_freqs)
            coverage_factor = min(1.0, matched_q / max(1.5, len(unique_q) * 0.45))
            b_norm = ((b_raw / (b_raw + 4.0)) * coverage_factor) if b_raw > 0 else 0.0
            # Score heurístico de BM25 e cosseno; não é probabilidade de correção.
            calibrated_score = round(min(1.0, (0.50 * b_norm) + (0.50 * d_raw)), 4)
            facts = self.fact_catalog.for_chunk(chunk.chunk_id, chunk.content) if self.catalog_required else extract_facts(chunk.chunk_id, chunk.content)
            source_url = urlsplit(chunk.source_url)
            hostname = (source_url.hostname or '').lower()
            official_getnet_host = hostname == 'getnet.com.br' or hostname.endswith('.getnet.com.br')
            if (source_url.scheme == 'https' and official_getnet_host
                    and chunk.origin_type == 'verified_markdown_mirror'
                    and chunk.provenance.get('verification_status') == 'reviewed_against_official_page'):
                # This verified official host is the Brazil Getnet property.
                # Its region can scope otherwise generic facts, but never to
                # another country; an explicit foreign query therefore fails.
                facts = [replace(fact, geographic_scope='brasil')
                         if not fact.geographic_scope else fact for fact in facts]
            answerability = assess(query, chunk.content, chunk.title, facts=facts, chunk_id=chunk.chunk_id)

            fused_items.append(
                {
                    "doc_id": chunk.chunk_id,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source_file,
                    "source_url": chunk.source_url,
                    "title": chunk.title,
                    "content": chunk.content,
                    "origin_type": chunk.origin_type,
                    "provenance": chunk.provenance,
                    "answerability": answerability,
                    "answer_snippets": answerability['answer_snippets'],
                    "approved_facts": [f for check in answerability["coverage"] for f in check["facts"]],
                    "bm25_score": round(b_raw, 4),
                    "dense_cosine_score": round(d_raw, 4),
                    "rrf_score": round(rrf_score, 5),
                    "score": calibrated_score,
                    "relevance_score": calibrated_score,
                }
            )

        # Ordena primariamente pelo score calibrado + desempate por RRF
        fused_items.sort(key=lambda x: (x["relevance_score"], x["rrf_score"]), reverse=True)
        # Exact relation coverage against verified body spans is stronger than
        # the coarse retrieval score gate. Keep such a source even when noisy
        # surrounding wording lowers the heuristic score slightly.
        eligible = [item for item in fused_items if item['approved_facts'] and (
            item["relevance_score"] >= min_thresh or
            (item['answerability']['answerable'] and item['origin_type'] == 'verified_markdown_mirror'
             and item.get('provenance', {}).get('verification_status') == 'reviewed_against_official_page'))]
        eligible.sort(key=lambda item: (item['answerability']['answerable'], item['relevance_score'], item['rrf_score']), reverse=True)
        # Select contributing chunks, then verify every requested relation across
        # the selected evidence set. One covered subquestion is not global coverage.
        top_Matches = eligible[:top_k]
        coverage = []
        for requirement in parse_requirements(query)['requirements']:
            supporting = [check for item in top_Matches for check in item['answerability']['coverage']
                          if check['requirement'] == requirement and check['facts']]
            facts = [fact for check in supporting for fact in check['facts']]
            coverage.append({'requirement':requirement,'status':'supported' if facts else 'unknown','facts':facts})
        max_rel = fused_items[0]["relevance_score"] if fused_items else 0.0

        if not top_Matches or any(check['status'] == 'unknown' for check in coverage):
            return {
                "abstained": True,
                "abstention_reason": (
                    f"Nenhum corpo de documento cobre o tema e todas as restrições solicitadas no limiar ({min_thresh:.2f}). "
                    f"Maior score observado: {max_rel:.4f}."
                ),
                "max_relevance_score": max_rel,
                "results": [],
                "citations": [],
                "answerability": {"answerable":False,"coverage":coverage,"parser":"bounded_local_relations_v4","candidate_checks":[{'chunk_id':item['chunk_id'],'missing':item['answerability']['missing']} for item in fused_items[:6]]},
            }

        citations: List[Citation] = []
        for m in top_Matches:
            excerpt = ' '.join(m['answer_snippets'])[:400]
            citations.append(
                Citation(
                    chunk_id=m["chunk_id"],
                    title=m["title"],
                    source=m["source"],
                    source_url=m["source_url"],
                    relevance_score=m["relevance_score"],
                    excerpt=excerpt,
                )
            )

        return {
            "abstained": False,
            "abstention_reason": None,
            "max_relevance_score": max_rel,
            "results": top_Matches,
            "citations": citations,
            "answerability": {"answerable":True,"coverage":coverage,"method":"verified_subject_relation_value_spans","parser":"bounded_local_relations_v4","limit":"Gramática local delimitada, não entailment semântico universal."},
        }

    def search(self, query: str, top_k: int = settings.RAG_TOP_K) -> List[Dict[str, Any]]:
        """
        Interface direta compatível que retorna os chunks aprovados pelo threshold.
        """
        outcome = self.retrieve_hybrid(query=query, top_k=top_k)
        return outcome["results"]


retriever = HybridKnowledgeRetriever()
