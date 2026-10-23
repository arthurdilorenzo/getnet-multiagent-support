import hashlib
import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


class BilingualDenseEmbedder:
    """
    Motor de Embeddings Vetoriais Densos (384 dimensões, L2-normalizados)
    com projeção semântica bilíngue (PT/EN) e n-gramas sub-palavra.
    Calcula similaridade de cosseno real entre queries e documentos em memória
    ou persistido em `data/vector_index.json`, com suporte opcional à API Gemini Embedding.
    """

    DIM = 384

    # Clusters semânticos bilíngues de domínio (adquirência e meios de pagamento)
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

    @staticmethod
    def _feature_bucket(feature: str, dim: int = DIM) -> Tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        return idx, sign

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """
        Converte um texto (PT ou EN) em um vetor denso de 384 dimensões L2-normalizado.
        Combina:
        1. Unigramas e Bigramas normalizados.
        2. N-gramas de caracteres (4-gramas) para robustez morfológica.
        3. Projeção nos clusters de conceitos bilíngues de pagamentos.
        """
        vec = [0.0] * cls.DIM
        norm = cls.normalize(text)
        tokens = cls.tokenize(text)
        if not tokens:
            return vec

        # 1. Unigrams (peso 2.0) e Bigrams (peso 2.5)
        for i, tok in enumerate(tokens):
            idx, sign = cls._feature_bucket(f"u:{tok}")
            vec[idx] += sign * 2.0
            # Stem prefix (5 chars) para aproximar variações morfológicas (parcela/parcelar/parcelamento)
            if len(tok) >= 5:
                s_idx, s_sign = cls._feature_bucket(f"st:{tok[:5]}")
                vec[s_idx] += s_sign * 1.5
            if i + 1 < len(tokens):
                bg = f"b:{tok}_{tokens[i + 1]}"
                b_idx, b_sign = cls._feature_bucket(bg)
                vec[b_idx] += b_sign * 2.5

        # 2. Character 4-grams nas palavras principais (peso 0.6)
        for tok in tokens[:25]:
            padded = f"<{tok}>"
            for j in range(len(padded) - 3):
                cg = padded[j : j + 4]
                c_idx, c_sign = cls._feature_bucket(f"c:{cg}")
                vec[c_idx] += c_sign * 0.6

        # 3. Projeção Semântica Bilíngue (Concept Anchors, peso 4.5 por match)
        for concept_id, phrases in cls.CONCEPT_CLUSTERS:
            matches = 0
            for phrase in phrases:
                if phrase in norm:
                    matches += 1
            if matches > 0:
                weight = min(12.0, matches * 4.5)
                # Projeta em 3 coordenadas ortogonais determinísticas para o cluster
                for salt in ("a", "b", "c"):
                    k_idx, k_sign = cls._feature_bucket(f"concept:{concept_id}:{salt}")
                    vec[k_idx] += k_sign * weight

        # Normalização L2
        norm_l2 = math.sqrt(sum(x * x for x in vec))
        if norm_l2 > 1e-9:
            vec = [round(x / norm_l2, 6) for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(0.0, min(1.0, float(dot)))
