from datetime import datetime, timezone
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
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
        self._elements: List[Tuple[str, bool]] = []
        self._current_tag = ""
        self.blocks: List[Tuple[str, str]] = []
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            return
        attributes = dict(attrs)
        labels = " ".join(str(attributes.get(key, "")) for key in ("class", "id", "role")).lower()
        skipped = bool(self._skip_depth) or tag in {
            "script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button", "aside"
        } or bool(re.search(r"cookie|privacy|navigation|menu|modal|popup|contentinfo", labels))
        self._elements.append((tag, skipped))
        if skipped:
            if not self._skip_depth:
                self._flush()
                self._current_tag = ""
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in ("h1", "h2", "h3", "h4", "p", "li"):
            self._flush()
            self._current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth == 0 and tag in ("h1", "h2", "h3", "h4", "p", "li"):
            self._flush()
            self._current_tag = ""
        for index in range(len(self._elements) - 1, -1, -1):
            if self._elements[index][0] == tag:
                self._skip_depth -= sum(skipped for _, skipped in self._elements[index:])
                del self._elements[index:]
                break

    def close(self) -> None:
        super().close()
        self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._current_tag:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self._buffer.append(cleaned)

    def _flush(self) -> None:
        if self._buffer and self._current_tag:
            text = " ".join(self._buffer).strip()
            if len(text) >= (3 if self._current_tag.startswith("h") else 20):
                self.blocks.append((self._current_tag, text))
        self._buffer = []


class GetnetCorpusIngestor:
    """
    Pipeline de Ingestão Documental e Web Scraping Oficial da Getnet:
    1. Realiza scraping HTTP de URLs canônicas em `site.getnet.com.br` e `www.getnet.net/en`.
    2. Processa os arquivos Markdown estruturados em `data/*.md` com metadados YAML (`source_url`).
    3. Remove boilerplate, sanitiza diretivas conhecidas e divide blocos em janelas com overlap e SHA-256.
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
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def clean_content(content: str) -> str:
        """Apply the same deterministic hygiene to mirrors, HTTP blocks and snapshots.

        Remove known website boilerplate and public contact numbers. Keep product
        instructions, including legitimate Pix key types and terminal menu steps.
        This is a heuristic filter, not independent fact or injection verification.
        """
        noise_markers = (
            "nos informe cpf",
            "black friday", "cookie", "politica de privacidade", "privacy policy",
            "como podemos te ajudar", "como voce prefere continuar", "como deseja continuar",
            "selecione para ver as taxas", "selecione sua faixa de faturamento",
            "selecione o tipo de cadastro", "escolha a melhor forma para finalizar",
            "recebeu sua maquininha de outra pessoa", "peguei a maquininha de outra pessoa",
            "nao precisa realizar este cadastro", "para melhor direcionamento",
            "aponte a camera do seu celular", "ja sou cliente, e procuro",
            "preencha o formulario", "ja e cliente getnet?", "ao clicar em enviar",
            "palavras de quem acredita", "o time foi bem engajado", "api da genet proporcionou",
            "denise r. de almeida", "a getnet foi a que mais acreditou",
            "concorrencias com tecnologia", "vi mais simplicidade onde mais me preocupava",
            "as taxas sao excelentes", "looking up for our top stories",
        )
        phone_pattern = re.compile(r"\b(?:0800[\s.-]*\d{3}[\s.-]*\d{4}|400[23][\s.-]*4000)\b")
        lines = []
        seen = set()
        for raw_line in content.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            folded = "".join(c for c in unicodedata.normalize("NFKD", line.lower()) if not unicodedata.combining(c))
            if not line or any(marker in folded for marker in noise_markers) or re.match(r"^[#\s*-]*sac\b", folded):
                continue
            if re.match(r"^[#\s*-]*(?:capitais e regioes metropolitanas|demais localidades)\b", folded):
                continue
            # Remove phone-only contact parentheses without discarding the surrounding procedure.
            line = re.sub(r"\([^)]*\)", lambda match: "" if phone_pattern.search(match[0]) else match[0], line)
            line = phone_pattern.sub("", line)
            line = re.sub(r"\bSAC\s*:?", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\(\s*[`\s/.,-]*\)", "", line)
            line = re.sub(r"[` ]*[/][` ]*(?=[.)]|$)", "", line)
            line = re.sub(r"\s+([,.])", r"\1", line).rstrip(" `")
            line = re.sub(r"[ \t]+", " ", line).strip()
            if not line.strip("#- .:`/") or line in seen:
                continue
            seen.add(line)
            lines.append(line)
        cleaned, _ = GuardrailManager.sanitize_untrusted_content("\n".join(lines))
        return cleaned.strip()

    def clean_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preserve source provenance while updating content-derived identifiers."""
        cleaned = self.clean_content(chunk.get("content", ""))
        if len(cleaned) < 30:
            return None
        result = dict(chunk)
        digest = self._sha256(cleaned)
        prefix = result["chunk_id"].rsplit("_", 1)[0]
        suffix_length = 8 if result.get("origin_type") == "live_web_scrape" else 6
        result.update(content=cleaned, content_hash=digest, chunk_id=f"{prefix}_{digest[:suffix_length]}")
        return result

    def clean_existing_corpus(self) -> List[Dict[str, Any]]:
        """Clean the captured snapshot offline, retaining URLs and scrape dates.

        No HTTP request occurs here. Repeating this operation is byte-stable once
        the snapshot has been cleaned; generated_at remains its capture metadata.
        """
        payload = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        chunks = [cleaned for item in payload.get("chunks", []) if (cleaned := self.clean_chunk(item))]
        payload.update(
            corpus_version="2.1.0",
            hygiene_version="1",
            total_chunks=len(chunks),
            markdown_chunks_count=sum(c.get("origin_type") == "verified_markdown_mirror" for c in chunks),
            live_web_chunks_count=sum(c.get("origin_type") == "live_web_scrape" for c in chunks),
            chunks=chunks,
        )
        self.corpus_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return chunks

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
                sanitized_sec = self.clean_content(clean_sec)
                if len(sanitized_sec) < 30:
                    continue
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
                        # A local file with an official URL is not automatically
                        # a verified mirror. Retain the actual review provenance.
                        "origin_type": "verified_markdown_mirror" if meta.get("verification_status") == "reviewed_against_official_page" else "unverified_local_summary",
                        "verification_status": meta.get("verification_status", "unverified_local_summary"),
                        "verified_at": meta.get("verified_at"),
                        "review_method": meta.get("review_method", "none"),
                        "source_location": meta.get("source_location", ""),
                        "source_access_note": meta.get('source_access_note',''),
                        "additional_url": meta.get("additional_url", ""),
                        "scraped_at": None,
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
                    parser.close()

                    # Filtra textos repetitivos de menu superior e agrupa em blocos semânticos
                    filtered_lines: List[str] = []
                    seen_lines = set()
                    for tag, text in parser.blocks:
                        text = self.clean_content(text)
                        if not text:
                            continue
                        if text in seen_lines:
                            continue
                        seen_lines.add(text)
                        prefix = "## " if tag in ("h1", "h2", "h3") else "- "
                        filtered_lines.append(f"{prefix}{text}")

                    # Segmenta em janelas de ~650 caracteres com overlap de 2 linhas.
                    window: List[str] = []
                    window_len = 0
                    seg_idx = 0
                    last_emitted = ""
                    for line in filtered_lines + [None]:
                        if line is not None:
                            window.append(line)
                            window_len += len(line)
                        if window and (window_len >= 650 or line is None):
                            sanitized = self.clean_content("\n".join(window))
                            if len(sanitized) < 30 or sanitized == last_emitted or (line is None and last_emitted.endswith(sanitized)):
                                continue
                            seg_idx += 1
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
                            last_emitted = sanitized
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
            "corpus_version": "2.1.0",
            "hygiene_version": "1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(all_chunks),
            "markdown_chunks_count": sum(c.get('origin_type')=='verified_markdown_mirror' for c in md_chunks),
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


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Build or clean the local Getnet corpus.")
    cli.add_argument("--clean-existing", action="store_true", help="Clean the versioned snapshot offline, preserving provenance.")
    cli.add_argument("--live", action="store_true", help="Fetch official URLs in addition to the local Markdown mirrors.")
    args = cli.parse_args()
    if args.clean_existing and args.live:
        cli.error("--clean-existing and --live are mutually exclusive")
    ingestor = GetnetCorpusIngestor()
    if args.clean_existing:
        ingestor.clean_existing_corpus()
    else:
        ingestor.build_and_persist_corpus(run_live_scrape=args.live)
