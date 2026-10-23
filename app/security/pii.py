import re
from typing import Any, Dict


class PIIRedactor:
    """
    Mascaramento por formatos e campos/rótulos reconhecidos nas fronteiras locais.

    Não é detector universal de PII nem certificação PCI-DSS/LGPD. CNPJ sem
    pontuação exige campo/rótulo fiscal explícito; contar 14 dígitos não permite
    distinguir um documento de um identificador opaco de sessão ou terminal.
    """

    # CPF: 000.000.000-00 ou 11 dígitos isolados precedidos por rótulo
    CPF_PATTERN = re.compile(r"\b(\d{3})\.(\d{3})\.(\d{3})-(\d{2})\b")
    CPF_RAW_LABELED = re.compile(
        r"(?i)(\bcpf\s*(?:(?:é|eh|e|is)\b\s*)?[:=]?\s*)(\d{3})(\d{3})(\d{3})(\d{2})\b"
    )

    # CNPJ: 00.000.000/0001-00
    CNPJ_PATTERN = re.compile(r"\b(\d{2})\.(\d{3})\.(\d{3})/(\d{4})-(\d{2})\b")
    CNPJ_VALUE = re.compile(r"([0-9]{2})\.?([0-9]{3})\.?([0-9]{3})/?([0-9]{4})-?([0-9]{2})")
    CNPJ_MASKED = re.compile(r"[0-9]{2}\.\*{3}\.\*{3}/[0-9]{4}-\*{2}")
    CNPJ_RAW_LABELED = re.compile(
        r"(?i)(\b(?:cnpj|(?:company|business|corporate|brazilian)[_\s-]+tax[_\s-]+(?:id|number)|tax[_\s-]+id)\b"
        r"(?:\s*\((?:mascarado|masked)\))?[\s*\"'`]*"
        r"(?:(?:é|eh|e|is)\b\s*)?[:=]?[\s*\"'`]*)([0-9]{14})(?![\w-])"
    )
    CNPJ_KEYS = frozenset({
        "cnpj", "companycnpj", "cnpjnumber", "numerocnpj", "cnpjnumero",
        "braziliantaxid", "businesstaxid", "corporatetaxid", "companytaxid", "taxid",
        "braziliantaxnumber", "businesstaxnumber", "corporatetaxnumber", "companytaxnumber",
    })

    # Cartão de Crédito: 13 a 19 dígitos (agrupados em 4 blocos ou contínuos)
    CARD_GROUPED_PATTERN = re.compile(r"\b(\d{4})[\s\-](\d{4})[\s\-](\d{4})[\s\-](\d{4})\b")
    CARD_CONTINUOUS_PATTERN = re.compile(r"\b([3456]\d{3})(\d{8,11})(\d{4})\b")

    # Senha / CVV explícitos
    PASSWORD_PATTERN = re.compile(
        r"(?i)\b(senha|password|pin|cvv|cvc|token\s+secreto)\s*"
        r"(?::|=|(?:é|eh|e|is)\b\s*[:=]?)\s*"
        r"(?:\"[^\"\n]*\"|'[^'\n]*'|“[^”\n]*”|[^\s,;.!?]+)"
    )

    # Conta corrente explícita
    BANK_ACCOUNT_PATTERN = re.compile(r"(?i)\b(conta[_\s]*corrente|cc|c/c)\b\s*[:=]?\s*(\d{2,6})([\d\-]+)(?!\w)")

    @classmethod
    def mask_text(cls, text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        masked = text
        # Mascara CNPJ mantendo raiz parcial
        masked = cls.CNPJ_PATTERN.sub(r"\1.***.***/\4-**", masked)
        masked = cls.CNPJ_RAW_LABELED.sub(lambda m: m[1] + cls.mask_cnpj(m[2]), masked)
        # Mascara CPF
        masked = cls.CPF_PATTERN.sub(r"***.***. \3-**".replace(" ", ""), masked)
        masked = cls.CPF_RAW_LABELED.sub(r"\1***.***.\4-**", masked)
        # Mascara Cartão de Crédito (PCI-DSS: exibe apenas 4 primeiros e 4 últimos)
        masked = cls.CARD_GROUPED_PATTERN.sub(r"\1 **** **** \4", masked)
        masked = cls.CARD_CONTINUOUS_PATTERN.sub(r"\1********\3", masked)
        # Mascara Senha / CVV
        masked = cls.PASSWORD_PATTERN.sub(r"\1: [REDACTED]", masked)
        # Mascara Conta Bancária
        masked = cls.BANK_ACCOUNT_PATTERN.sub(r"\1: ***\3", masked)
        return masked

    @classmethod
    def mask_cnpj(cls, cnpj: Any) -> str:
        """Mask a value explicitly identified as CNPJ, including JSON numbers.

        Malformed explicit document values fail closed. Already masked values
        stay stable, so repeated egress sanitization does not destroy evidence.
        """
        if cnpj is None or cnpj == "":
            return ""
        value = str(cnpj).strip()
        if value == "[REDACTED]" or cls.CNPJ_MASKED.fullmatch(value):
            return value
        match = cls.CNPJ_VALUE.fullmatch(value)
        if match:
            return f"{match[1]}.***.***/{match[4]}-**"
        return "[REDACTED]"

    @classmethod
    def _mask_cnpj_field(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [cls._mask_cnpj_field(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._mask_cnpj_field(item) for key, item in value.items()}
        return cls.mask_cnpj(value)

    @classmethod
    def mask_bank_account(cls, agencia: str, conta: str) -> Dict[str, str]:
        ag_masked = f"**{agencia[-2:]}" if agencia and len(agencia) >= 2 else "****"
        cc_masked = f"*****{conta[-3:]}" if conta and len(conta) >= 3 else "*****"
        return {"agencia": ag_masked, "conta_corrente": cc_masked}

    @classmethod
    def mask_structure(cls, obj: Any) -> Any:
        """
        Aplica mascaramento recursivo em dicionários, listas e strings.
        """
        if isinstance(obj, str):
            return cls.mask_text(obj)
        if isinstance(obj, dict):
            result: Dict[str, Any] = {}
            for k, v in obj.items():
                key_lower = str(k).lower()
                if key_lower in ("password", "senha", "cvv", "cvc", "cpf", "pin", "secret", "api_key", "token_secreto"):
                    result[k] = "[REDACTED]"
                elif re.sub(r"[\s_-]+", "", key_lower) in cls.CNPJ_KEYS:
                    result[k] = cls._mask_cnpj_field(v)
                elif key_lower in ("conta_corrente", "account_number") and isinstance(v, str):
                    result[k] = f"*****{v[-3:]}" if len(v) >= 3 else "*****"
                else:
                    result[k] = cls.mask_structure(v)
            return result
        if isinstance(obj, list):
            return [cls.mask_structure(item) for item in obj]
        return obj
