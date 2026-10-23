import re
from typing import Any, Dict, List, Union


class PIIRedactor:
    """
    Motor de mascaramento de PII (Dados Pessoais e Sensíveis PCI-DSS / LGPD).
    Sanitiza CPF, CNPJ, PAN de Cartão de Crédito, CVV, Senhas e Contas Bancárias
    antes de gravação em logs, traces, banco de memória ou envio ao LLM.
    """

    # CPF: 000.000.000-00 ou 11 dígitos isolados precedidos por rótulo
    CPF_PATTERN = re.compile(r"\b(\d{3})\.(\d{3})\.(\d{3})-(\d{2})\b")
    CPF_RAW_LABELED = re.compile(
        r"(?i)(\bcpf\s*(?:(?:é|eh|e|is)\b\s*)?[:=]?\s*)(\d{3})(\d{3})(\d{3})(\d{2})\b"
    )

    # CNPJ: 00.000.000/0001-00
    CNPJ_PATTERN = re.compile(r"\b(\d{2})\.(\d{3})\.(\d{3})/(\d{4})-(\d{2})\b")

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
    def mask_cnpj(cls, cnpj: str) -> str:
        if not cnpj:
            return ""
        return cls.CNPJ_PATTERN.sub(r"\1.***.***/\4-**", cnpj)

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
                elif key_lower == "cnpj" and isinstance(v, str):
                    result[k] = cls.mask_cnpj(v)
                elif key_lower in ("conta_corrente", "account_number") and isinstance(v, str):
                    result[k] = f"*****{v[-3:]}" if len(v) >= 3 else "*****"
                else:
                    result[k] = cls.mask_structure(v)
            return result
        if isinstance(obj, list):
            return [cls.mask_structure(item) for item in obj]
        return obj
