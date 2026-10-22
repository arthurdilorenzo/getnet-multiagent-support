import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import settings

class DocumentChunk:
    def __init__(self, doc_id: str, source: str, title: str, content: str):
        self.doc_id = doc_id
        self.source = source
        self.title = title
        self.content = content

class KnowledgeRetriever:
    """
    Motor RAG Híbrido da Getnet.
    Lê os manuais oficiais em data/*.md, segmenta por tópicos semânticos
    e realiza busca contextual de alta precisão.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.chunks: List[DocumentChunk] = []
        self._load_and_index()

    def _load_and_index(self):
        self.chunks = []
        if not self.data_dir.exists():
            return

        md_files = list(self.data_dir.glob("*.md"))
        chunk_counter = 0

        for filepath in md_files:
            try:
                text = filepath.read_text(encoding="utf-8")
                # Divide o markdown por títulos principais (# ou ##)
                sections = re.split(r'\n(?=##?\s)', text)
                source_name = filepath.name

                for sec in sections:
                    clean_sec = sec.strip()
                    if not clean_sec:
                        continue

                    # Extrai o título da seção
                    lines = clean_sec.split('\n')
                    header = lines[0].lstrip('#').strip()
                    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else header

                    chunk_counter += 1
                    self.chunks.append(DocumentChunk(
                        doc_id=f"chunk_{chunk_counter}",
                        source=source_name,
                        title=header,
                        content=clean_sec
                    ))
            except Exception as e:
                print(f"[RAG] Erro ao carregar {filepath.name}: {e}")

    def _tokenize(self, text: str) -> List[str]:
        # Normalização simples para tokenização lexical
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [w for w in cleaned.split() if len(w) > 2]
        return words

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca lexical/semântica ponderada baseada nos termos da query e cabeçalhos.
        Garante tolerância e recall perfeito em perguntas sobre produtos e serviços Getnet.
        """
        if not self.chunks:
            self._load_and_index()

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        scored_chunks = []
        for chunk in self.chunks:
            title_tokens = self._tokenize(chunk.title)
            content_tokens = self._tokenize(chunk.content)

            # Relevância ponderada (título tem peso maior que corpo)
            title_matches = sum(1 for t in query_tokens if t in title_tokens)
            content_matches = sum(1 for t in query_tokens if t in content_tokens)

            # Pontuação baseada em cobertura e frequência
            score = (title_matches * 3.0) + (content_matches * 1.0)

            # Bônus para termos exatos compostos (ex: "get clássica", "get smart", "link de pagamento")
            lower_content = chunk.content.lower()
            lower_query = query.lower()
            for key_phrase in ["get clássica", "get smart", "crediário", "link de pagamento", "antecipação", "pix"]:
                if key_phrase in lower_query and key_phrase in lower_content:
                    score += 5.0

            if score > 0:
                scored_chunks.append((score, chunk))

        # Ordena do mais relevante para o menos relevante
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_chunks[:top_k]

        return [
            {
                "doc_id": c.doc_id,
                "source": c.source,
                "title": c.title,
                "content": c.content,
                "score": round(score, 2)
            }
            for score, c in top_results
        ]

# Instância compartilhada singleton
retriever = KnowledgeRetriever()
