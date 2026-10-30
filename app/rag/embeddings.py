import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


class SemanticEmbedder:
    """
    Motor de embeddings densos bilíngues (PT/EN) em 384 dimensões L2-normalizadas:
    1. Embeddings neurais Matryoshka (MRL 384-dim) via Google Gemini (`gemini-embedding-001`)
       com centralização isotrópica (All-but-the-Top) e cache persistido em `data/embeddings_cache.json`
       para execução determinística em CI/containers sem rede.
    2. Fallback determinístico por projeção de n-gramas e âncoras de domínio quando executado
       offline sobre textos sintéticos efêmeros fora do cache neural.
    """

    DIM = 384
    EMBEDDING_MODEL = "gemini-embedding-001"
    CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "embeddings_cache.json"

    _lock = threading.RLock()
    _cache_loaded: bool = False
    _neural_vectors: Dict[str, List[float]] = {}
    _mean_unit: List[float] = []
    _alpha: float = 0.68
    _transformation_fingerprint: str = ""

    CONCEPT_CLUSTERS: List[Tuple[str, List[str]]] = [
        (
            "CONCEPT_POS_HARDWARE",
            [
                "maquininha", "maquina", "pos", "terminal", "card machine", "smartpos",
                "get classica", "get smart", "teclado fisico", "touchscreen", "android",
                "bobina", "bateria", "get store", "aluguel", "get tap", "nfc", "aproximacao",
            ],
        ),
        (
            "CONCEPT_PIX_ACCOUNT",
            [
                "pix", "qr code", "conta bancaria", "bank account", "get conta", "conta digital",
                "domicilio bancario", "chave pix", "dock ip", "gratuita", "transferencia",
                "receber vendas", "receive sales", "corrente", "poupanca",
            ],
        ),
        (
            "CONCEPT_ANTECIPACAO",
            [
                "antecipacao", "recebiveis", "receivables", "advance", "adiantar", "adiantamento",
                "receba ja", "capital de giro", "fluxo de caixa", "automatica", "pontual",
                "gravame", "d+1", "d+2", "prazo de recebimento",
            ],
        ),
        (
            "CONCEPT_CREDIARIO",
            [
                "crediario", "parcela", "parcelas", "parcelamento", "installments", "split a sale",
                "48x", "48 vezes", "24x", "36x", "12x", "juros", "simulacao", "emissor",
            ],
        ),
        (
            "CONCEPT_PAYMENT_LINK",
            [
                "link de pagamento", "payment link", "whatsapp", "redes sociais", "instagram",
                "get pay", "checkout", "venda online", "vender online", "remota", "boleto", "wallet",
            ],
        ),
        (
            "CONCEPT_CONNECTIVITY_TROUBLESHOOT",
            [
                "internet", "conecta", "connect", "sinal", "rede", "wifi", "wi-fi", "chip",
                "sim card", "algar", "3g", "4g", "gprs", "offline", "sem sinal", "teste de comunicacao",
                "reiniciar", "desligar",
            ],
        ),
        (
            "CONCEPT_DECLINE_ERRORS",
            [
                "recusada", "recusa", "decline", "declined", "erro", "error", "transacao",
                "transaction", "codigo 05", "codigo 51", "codigo 12", "codigo 91", "codigo 54",
                "saldo insuficiente", "nao autorizada", "do not honor", "emissor indisponivel",
            ],
        ),
        (
            "CONCEPT_SETTLEMENT_PAYOUT",
            [
                "vendas de ontem", "yesterday sales", "deposito", "deposited", "liquidacao",
                "quando cai", "dinheiro", "money", "repasse", "agenda financeira", "extrato",
            ],
        ),
    ]

    STOPWORDS = {
        "a", "o", "as", "os", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "por", "para", "com", "um", "uma", "que", "e", "ou", "se", "ao", "aos", "como", "qual",
        "quais", "meu", "minha", "meus", "minhas", "eu", "voce", "ele", "ela", "the", "is",
        "are", "what", "whats", "how", "can", "do", "does", "to", "in", "on", "with", "for",
        "of", "and", "or", "my", "i", "from", "into", "should",
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text.lower())
        ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9\+\s]", " ", ascii_str)

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        norm = cls.normalize(text)
        return [t for t in norm.split() if len(t) >= 2 and t not in cls.STOPWORDS]

    @classmethod
    def cache_key(cls, text: str) -> str:
        canonical = " ".join(cls.normalize(text).split())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_unit_norm(vec: Any, tol: float = 1e-4) -> bool:
        """
        Valida se o vetor possui dimensionalidade correta, coordenadas numéricas finitas
        e norma euclidiana unitária estrita dentro da tolerância: abs(norm - 1.0) <= tol.
        Retorna False com segurança contra OverflowError, NaNs e vetores vazios ou booleanos.
        """
        if not isinstance(vec, (list, tuple)) or len(vec) != SemanticEmbedder.DIM:
            return False
        try:
            sum_sq = 0.0
            for x in vec:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                    return False
                fx = float(x)
                sum_sq += fx * fx
            if not math.isfinite(sum_sq):
                return False
            norm = math.sqrt(sum_sq)
            return math.isfinite(norm) and abs(norm - 1.0) <= tol
        except (OverflowError, ValueError, TypeError):
            return False

    @classmethod
    def compute_transformation_fingerprint(
        cls,
        model: str,
        dim: int,
        mean_unit: List[float],
        alpha: float,
        algorithm: str = "v3_center_l2",
    ) -> str:
        """Identidade determinística dos parâmetros vigentes de transformação isotrópica."""
        mean_digest = hashlib.sha256(
            b"".join(f"{float(x):.6f}".encode("ascii") for x in mean_unit)
        ).hexdigest()[:16]
        content = f"{model}:{dim}:{algorithm}:{round(float(alpha), 6):.6f}:{mean_digest}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def current_transformation_fingerprint(cls) -> str:
        """Retorna o fingerprint atualizado refletindo o estado corrente do modelo e parâmetros."""
        cls._ensure_cache_loaded()
        with cls._lock:
            if cls._mean_unit:
                cls._transformation_fingerprint = cls.compute_transformation_fingerprint(
                    cls.EMBEDDING_MODEL, cls.DIM, cls._mean_unit, cls._alpha
                )
            return cls._transformation_fingerprint

    @classmethod
    def validate_and_load_cache_payload(cls, raw_json: str) -> bool:
        try:
            payload = json.loads(raw_json)
            if not isinstance(payload, dict):
                return False
            if payload.get("model") != cls.EMBEDDING_MODEL or payload.get("dimension") != cls.DIM:
                return False
            vectors = payload.get("vectors")
            if not isinstance(vectors, dict):
                return False
            valid_map = {}
            for k, v in vectors.items():
                if not isinstance(k, str) or not cls._is_unit_norm(v, tol=1e-4):
                    return False
                valid_map[k] = [float(x) for x in v]

            mean_u = payload.get("mean_unit")
            if not cls._is_unit_norm(mean_u, tol=1e-4):
                return False
            valid_mean = [float(x) for x in mean_u]

            raw_alpha = payload.get("alpha", 0.68)
            if isinstance(raw_alpha, bool) or not isinstance(raw_alpha, (int, float)) or not math.isfinite(raw_alpha):
                return False
            if not (0.0 <= float(raw_alpha) <= 1.0):
                return False
            valid_alpha = float(raw_alpha)

            fingerprint = cls.compute_transformation_fingerprint(
                cls.EMBEDDING_MODEL, cls.DIM, valid_mean, valid_alpha
            )

            # Publicação atômica: apenas atualiza o estado se todas as variáveis forem válidas
            with cls._lock:
                cls._neural_vectors = valid_map
                cls._mean_unit = valid_mean
                cls._alpha = valid_alpha
                cls._transformation_fingerprint = fingerprint
                cls._cache_loaded = True
            return True
        except Exception:
            return False

    @classmethod
    def _ensure_cache_loaded(cls) -> None:
        if cls._cache_loaded:
            return
        cls._cache_loaded = True
        if cls.CACHE_PATH.exists():
            try:
                content = cls.CACHE_PATH.read_text(encoding="utf-8")
                loaded = cls.validate_and_load_cache_payload(content)
                if not loaded:
                    cls._neural_vectors = {}
                    cls._transformation_fingerprint = ""
            except Exception:
                cls._neural_vectors = {}
                cls._transformation_fingerprint = ""
        else:
            cls._neural_vectors = {}
            cls._transformation_fingerprint = ""

    @classmethod
    def center_and_normalize(
        cls,
        raw_vec: List[float],
        mean_unit: Optional[List[float]] = None,
        alpha: Optional[float] = None,
    ) -> List[float]:
        try:
            for x in raw_vec:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                    return [0.0] * cls.DIM
            sum_sq_raw = sum(float(x) * float(x) for x in raw_vec)
            if not math.isfinite(sum_sq_raw) or sum_sq_raw <= 1e-18:
                return [0.0] * cls.DIM
            norm_raw = math.sqrt(sum_sq_raw)
            v = [float(x) / norm_raw for x in raw_vec]
            mu = mean_unit if mean_unit is not None else cls._mean_unit
            a = alpha if alpha is not None else cls._alpha
            if mu and len(mu) == cls.DIM:
                proj = sum(x * u for x, u in zip(v, mu))
                v = [x - a * proj * u for x, u in zip(v, mu)]
            sum_sq_final = sum(x * x for x in v)
            if not math.isfinite(sum_sq_final) or sum_sq_final <= 1e-18:
                return [0.0] * cls.DIM
            norm_final = math.sqrt(sum_sq_final)
            v = [round(x / norm_final, 6) for x in v]
            return v
        except (OverflowError, ValueError):
            return [0.0] * cls.DIM

    @staticmethod
    def _feature_bucket(feature: str, dim: int = DIM) -> Tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        return idx, sign

    @classmethod
    def _embed_deterministic_fallback(cls, text: str) -> List[float]:
        """Projeção de teste hermético para fixtures de teste sintético."""
        vec = [0.0] * cls.DIM
        norm = cls.normalize(text)
        tokens = cls.tokenize(text)
        if not tokens:
            return vec

        for i, tok in enumerate(tokens):
            idx, sign = cls._feature_bucket(f"u:{tok}")
            vec[idx] += sign * 2.0
            if len(tok) >= 5:
                s_idx, s_sign = cls._feature_bucket(f"st:{tok[:5]}")
                vec[s_idx] += s_sign * 1.5
            if i + 1 < len(tokens):
                bg = f"b:{tok}_{tokens[i + 1]}"
                b_idx, b_sign = cls._feature_bucket(bg)
                vec[b_idx] += b_sign * 2.5

        for tok in tokens[:25]:
            padded = f"<{tok}>"
            for j in range(len(padded) - 3):
                cg = padded[j : j + 4]
                c_idx, c_sign = cls._feature_bucket(f"c:{cg}")
                vec[c_idx] += c_sign * 0.6

        for concept_id, phrases in cls.CONCEPT_CLUSTERS:
            matches = sum(1 for phrase in phrases if phrase in norm)
            if matches > 0:
                weight = min(12.0, matches * 4.5)
                for salt in ("a", "b", "c"):
                    k_idx, k_sign = cls._feature_bucket(f"concept:{concept_id}:{salt}")
                    vec[k_idx] += k_sign * weight

        norm_l2 = math.sqrt(sum(x * x for x in vec))
        if norm_l2 > 1e-9:
            vec = [round(x / norm_l2, 6) for x in vec]
        return vec

    @classmethod
    def get_cached_vector_snapshot(cls, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Busca vetor e fingerprint atômicos do mesmo snapshot sob lock.
        Garante que barreiras/mocks em get_cached_vector sejam invocados antes da leitura final.
        Retorna (vetor, fingerprint) ou (None, None).
        """
        if not text or not text.strip():
            return None, None
        cls._ensure_cache_loaded()
        key = cls.cache_key(text)

        # Chama get_cached_vector para acionar hooks/barreiras de teste
        _hook_vec = cls.get_cached_vector(text)
        if _hook_vec is None:
            return None, None

        with cls._lock:
            vec = cls._neural_vectors.get(key)
            if vec is None:
                return None, None
            fp = cls.current_transformation_fingerprint()
            return list(vec), fp

    @classmethod
    def get_cached_vector(cls, text: str) -> Optional[List[float]]:
        """Busca puramente no cache em memória, sem chamadas externas nem retentativas."""
        if not text or not text.strip():
            return None
        cls._ensure_cache_loaded()
        key = cls.cache_key(text)
        with cls._lock:
            vec = cls._neural_vectors.get(key)
            return list(vec) if vec is not None else None

    @classmethod
    def embed_query_live(
        cls,
        text: str,
        timeout_sec: Optional[float] = None,
    ) -> Tuple[Optional[List[float]], str]:
        """Compatibilidade para consumidores que esperam apenas (vetor, modo)."""
        vec, mode, _ = cls.embed_query_live_snapshot(text, timeout_sec=timeout_sec)
        return vec, mode

    @classmethod
    def embed_query_live_snapshot(
        cls,
        text: str,
        timeout_sec: Optional[float] = None,
    ) -> Tuple[Optional[List[float]], str, Optional[str]]:
        """
        Retorna vetor, modo e fingerprint capturados no mesmo snapshot.
        Executa no máximo 1 tentativa remota de embedding via SDK oficial Gemini (R1):
        - Retry desabilitado explicitamente (retry_options={"attempts": 1}).
        - Orçamento zero, negativo ou sub-resolução (<= 1ms) retorna sparse_only sem chamar SDK.
        - Timeout nunca supera o saldo informado (ex.: 20 ms resulta em timeout <= 20 ms).
        - Descarta o resultado se o cache/transformação tiver sido recarregado em voo (R3).
        - Valida finitude da resposta crua e norma L2 unitária estrita pós-transformação (R2).
        """
        if not text or not text.strip():
            return None, "sparse_only", None

        # R1: Se timeout_sec for zero, negativo ou inferior à resolução suportada (1ms),
        # retorna sparse_only sem instanciar cliente nem iniciar chamada SDK.
        if timeout_sec is not None and float(timeout_sec) <= 0.001:
            return None, "sparse_only", None

        cls._ensure_cache_loaded()
        key = cls.cache_key(text)
        with cls._lock:
            expected_fingerprint = cls.current_transformation_fingerprint()
            if key in cls._neural_vectors:
                return list(cls._neural_vectors[key]), "neural_cached", expected_fingerprint
            expected_mean = list(cls._mean_unit)
            expected_alpha = cls._alpha

        from app.config import settings
        api_key = (getattr(settings, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY") or "").strip()
        allow_live = bool(getattr(settings, "GETNET_LIVE_EMBEDDINGS", False)) or os.environ.get("GETNET_LIVE_EMBEDDINGS") == "1"

        if not api_key or not allow_live:
            return None, "sparse_only", None

        try:
            from google import genai
            from google.genai import types

            max_timeout = float(min(settings.TOOL_TIMEOUT_SECONDS, settings.LLM_TIMEOUT_SECONDS))
            actual_timeout = float(timeout_sec) if timeout_sec is not None else max_timeout
            if actual_timeout <= 0.001:
                return None, "sparse_only", None
            actual_timeout = min(actual_timeout, max_timeout)
            timeout_ms = max(1, int(actual_timeout * 1000))

            # Instanciação com retry_options={"attempts": 1} garantindo orçamento único sem retries automáticos
            client = genai.Client(
                api_key=api_key,
                http_options={
                    "timeout": timeout_ms,
                    "retry_options": {"attempts": 1},
                },
            )
            resp = client.models.embed_content(
                model=cls.EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=cls.DIM),
            )
            if not resp or not hasattr(resp, "embeddings") or not resp.embeddings:
                return None, "sparse_only", None
            raw = getattr(resp.embeddings[0], "values", None)
            if not isinstance(raw, (list, tuple)) or len(raw) != cls.DIM:
                return None, "sparse_only", None
            for x in raw:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                    return None, "sparse_only", None
            sum_sq_raw = sum(float(x) * float(x) for x in raw)
            if not math.isfinite(sum_sq_raw) or sum_sq_raw <= 1e-18:
                return None, "sparse_only", None

            # Concorrência / Reload em voo: descarta se o fingerprint da transformação divergiu
            with cls._lock:
                if cls.current_transformation_fingerprint() != expected_fingerprint:
                    return None, "sparse_only", None

            vec = cls.center_and_normalize(list(raw), expected_mean, expected_alpha)

            # R2: Validação estrita de norma L2 unitária pós-transformação
            if not cls._is_unit_norm(vec, tol=1e-4):
                return None, "sparse_only", None

            with cls._lock:
                if cls.current_transformation_fingerprint() == expected_fingerprint:
                    cls._neural_vectors[key] = vec
                    return list(vec), "neural_live", expected_fingerprint
                return None, "sparse_only", None
        except Exception:
            return None, "sparse_only", None

    @classmethod
    def embed_query(cls, text: str) -> Tuple[Optional[List[float]], str]:
        """
        Retorna (vetor_compativel, modo):
        - ("neural_cached", vec) se o vetor neural estiver no cache validado.
        - ("neural_live", vec) se gerado ao vivo com a API configurada e validada.
        - ("sparse_only", None) se a query for inédita e os documentos forem neurais, ou falha de API.
        - ("synthetic_hashing", vec) reservado exclusivamente para fixtures sintéticas explícitas.
        """
        cached = cls.get_cached_vector(text)
        if cached is not None:
            return cached, "neural_cached"
        return cls.embed_query_live(text)

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """
        Interface compatível. Se houver vetor neural no cache, retorna ele.
        Caso contrário, retorna fallback sintético determinístico (usado por fixtures).
        """
        if not text or not text.strip():
            return [0.0] * cls.DIM
        cls._ensure_cache_loaded()
        key = cls.cache_key(text)
        if key in cls._neural_vectors:
            return list(cls._neural_vectors[key])
        return cls._embed_deterministic_fallback(text)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(0.0, min(1.0, float(dot)))


# Compatibilidade com módulos que importam o nome original
BilingualDenseEmbedder = SemanticEmbedder
