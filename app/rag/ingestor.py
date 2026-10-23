from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx
from app.agents.guardrails import GuardrailManager
from app.config import settings
from app.observability.logger import logger


class _CleanHTMLParser(HTMLParser):
    """
    Parser HTML estruturado que extrai títulos (h1-h4), parágrafos e itens de lista
    de páginas oficiais da Getnet ignorando scripts, menus repetitivos e estilos.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._current_tag = ""
        self.blocks: List[Tuple[str, str]] = []
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript", "svg", "path", "nav", "footer", "header"):
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in ("h1", "h2", "h3", "h4", "p", "li"):
            self._flush()
            self._current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "path", "nav", "footer", "header"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth == 0 and tag in ("h1", "h2", "h3", "h4", "p", "li"):
            self._flush()
            self._current_tag = ""

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._current_tag:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self._buffer.append(cleaned)

    def _flush(self) -> None:
        if self._buffer and self._current_tag:
            text = " ".join(self._buffer).strip()
            if len(text) >= 20:
                self.blocks.append((self._current_tag, text))
        self._buffer = []


class GetnetCorpusIngestor:
    """
    Pipeline de Ingestão Documental e Web Scraping Oficial da Getnet:
    1. Realiza scraping HTTP de URLs canônicas em `site.getnet.com.br` e `www.getnet.net/en`.
    2. Processa os arquivos Markdown estruturados em `data/*.md` com metadados YAML (`source_url`).
    3. Aplica sanitização contra Indirect Prompt Injection, chunking semântico com overlap e hash SHA-256.
    4. Persiste o corpus unificado e verificável em `data/getnet_corpus.json`.
    """

    OFFICIAL_GETNET_URLS: List[Dict[str, str]] = [
        {
            "url": "https://site.getnet.com.br/todas-as-maquininhas/",
            "topic": "Catálogo de Maquininhas Get Clássica e Get Smart",
        },
        {
            "url": "https://site.getnet.com.br/crediario/",
            "topic": "Crediário Getnet e Parcelamento em até 48x",
        },
        {
            "url": "https://site.getnet.com.br/pix/",
            "topic": "Pix Getnet e Recebimento na Maquininha",
        },
        {
            "url": "https://site.getnet.com.br/conta-digital/",
            "topic": "Get Conta Digital Gratuita e Domicílio Bancário",
        },
        {
            "url": "https://site.getnet.com.br/link-de-pagamento/",
            "topic": "Link de Pagamento Get Pay e Vendas pelo WhatsApp",
        },
        {
            "url": "https://site.getnet.com.br/duvidas/",
            "topic": "FAQ Oficial Getnet — Antecipação, Suporte Técnico, Chip e Taxas",
        },
        {
            "url": "https://www.getnet.net/en/",
            "topic": "Getnet Global Payment Solutions",
        },
    ]

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.corpus_path = Path(corpus_path or settings.CORPUS_JSON_PATH)

    @staticmethod
    def _parse_frontmatter(raw_md: str) -> Tuple[Dict[str, str], str]:
        meta: Dict[str, str] = {}
        body = raw_md
        if raw_md.startswith("---"):
            parts = raw_md.split("---", 2)
            if len(parts) == 3:
                fm_text, body = parts[1], parts[2]
                for line in fm_text.strip().splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
        return meta, body.strip()

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def load_markdown_chunks(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not self.data_dir.exists():
            return chunks

        for filepath in sorted(self.data_dir.glob("*.md")):
            raw_text = filepath.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(raw_text)
            source_url = meta.get("source_url", "https://site.getnet.com.br/")
            doc_title = meta.get("title", filepath.stem)

            sections = re.split(r"\n(?=##\s)", body)
            for idx, sec in enumerate(sections, start=1):
                clean_sec = sec.strip()
                if len(clean_sec) < 30:
                    continue
                sanitized_sec, _ = GuardrailManager.sanitize_untrusted_content(clean_sec)
                lines = sanitized_sec.splitlines()
                header = lines[0].lstrip("#").strip() if lines else doc_title
                content_hash = self._sha256(sanitized_sec)
                chunk_id = f"{filepath.stem}_s{idx}_{content_hash[:6]}"

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source_file": filepath.name,
                        "source_url": source_url,
                        "secondary_url": meta.get("secondary_url", ""),
                        "title": header,
                        "content": sanitized_sec,
                        "content_hash": content_hash,
                        "origin_type": "verified_markdown_mirror",
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        return chunks

    def scrape_live_getnet_pages(self, timeout_sec: float = 6.0) -> List[Dict[str, Any]]:
        """
        Coleta seções diretamente das páginas web oficiais da Getnet via HTTP,
        deduplicando blocos de navegação e gerando chunks rastreáveis por URL.
        """
        scraped_chunks: List[Dict[str, Any]] = []
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GetnetFDEIngestor/2.0)"}

        with httpx.Client(follow_redirects=True, timeout=timeout_sec, headers=headers) as client:
            for item in self.OFFICIAL_GETNET_URLS:
                url = item["url"]
                topic = item["topic"]
                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        continue
                    parser = _CleanHTMLParser()
                    parser.feed(resp.text)

                    # Filtra textos repetitivos de menu superior e agrupa em blocos semânticos
                    filtered_lines: List[str] = []
                    seen_lines = set()
                    noise_markers = (
                        "para melhor direcionamento",
                        "aponte a câmera do seu celular",
                        "já sou cliente, e procuro",
                        "peguei a maquininha de outra pessoa",
                    )
                    for tag, text in parser.blocks:
                        low = text.lower()
                        if any(m in low for m in noise_markers):
                            continue
                        if text in seen_lines:
                            continue
                        seen_lines.add(text)
                        prefix = "## " if tag in ("h1", "h2", "h3") else "- "
                        filtered_lines.append(f"{prefix}{text}")

                    # Segmenta em janelas de ~700 caracteres com overlap de 2 linhas
                    window: List[str] = []
                    window_len = 0
                    seg_idx = 0
                    for line in filtered_lines:
                        window.append(line)
                        window_len += len(line)
                        if window_len >= 650:
                            seg_idx += 1
                            raw_chunk = "\n".join(window)
                            sanitized, _ = GuardrailManager.sanitize_untrusted_content(raw_chunk)
                            c_hash = self._sha256(sanitized)
                            scraped_chunks.append(
                                {
                                    "chunk_id": f"web_{seg_idx}_{c_hash[:8]}",
                                    "source_file": url.rstrip("/").split("/")[-1] or "getnet_home",
                                    "source_url": str(resp.url),
                                    "secondary_url": url,
                                    "title": f"{topic} (Seção {seg_idx})",
                                    "content": sanitized,
                                    "content_hash": c_hash,
                                    "origin_type": "live_web_scrape",
                                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                                }
                            )
                            window = window[-2:]
                            window_len = sum(len(x) for x in window)
                            if seg_idx >= 6:  # Limita top-6 blocos densos por URL
                                break
                except Exception as exc:
                    logger.warning("getnet_scrape_url_skipped", url=url, error=str(exc))

        return scraped_chunks

    def build_and_persist_corpus(self, run_live_scrape: bool = True) -> List[Dict[str, Any]]:
        md_chunks = self.load_markdown_chunks()
        web_chunks: List[Dict[str, Any]] = []
        if run_live_scrape:
            web_chunks = self.scrape_live_getnet_pages()

        all_chunks = md_chunks + web_chunks
        payload = {
            "corpus_version": "2.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(all_chunks),
            "markdown_chunks_count": len(md_chunks),
            "live_web_chunks_count": len(web_chunks),
            "canonical_sources": [u["url"] for u in self.OFFICIAL_GETNET_URLS],
            "chunks": all_chunks,
        }
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "getnet_corpus_built",
            total_chunks=len(all_chunks),
            md_chunks=len(md_chunks),
            web_chunks=len(web_chunks),
        )
        return all_chunks
