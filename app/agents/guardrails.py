import re
from dataclasses import dataclass
from typing import List, Tuple
from app.security.pii import PIIRedactor


@dataclass(frozen=True)
class GuardrailRule:
    rule_id: str
    category: str
    pattern: re.Pattern


class GuardrailManager:
    """
    Camada multicamada de Segurança e Guardrails (Input, Indirect Context & Output).
    - Bloqueia Prompt Injection direto, Jailbreak, SQLi/XSS/RCE e extração de System Prompt.
    - NUNCA expõe expressões regulares ou lógica interna de detecção ao chamador.
    - Sanitiza conteúdo externo (RAG / Web Search) contra Indirect Prompt Injection.
    - Sanitiza a saída final aplicando redaction de PII (CPF, CNPJ, Cartão, Senhas).
    """

    INPUT_RULES: List[GuardrailRule] = [
        GuardrailRule(
            "SEC-INJ-001",
            "prompt_injection_override",
            re.compile(r"ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules)", re.I),
        ),
        GuardrailRule(
            "SEC-INJ-002",
            "prompt_injection_pt",
            re.compile(r"(ignore|esqueça|desconsidere)\s+(todas\s+as\s+)?(instruções|regras)\s+(anteriores|acima)", re.I),
        ),
        GuardrailRule(
            "SEC-INJ-003",
            "system_prompt_extraction",
            re.compile(r"(reveal|show|print|output|exiba|mostre|qual\s+é)\s+(your\s+|the\s+|o\s+seu\s+)?system\s+prompt", re.I),
        ),
        GuardrailRule(
            "SEC-INJ-004",
            "persona_hijack",
            re.compile(r"(você\s+agora\s+é\s+um\s+assistente\s+sem\s+regras|you\s+are\s+now\s+dan\b|do\s+anything\s+now|jailbreak|modo\s+desenvolvedor\s+sem\s+censura)", re.I),
        ),
        GuardrailRule(
            "SEC-INJ-005",
            "security_bypass",
            re.compile(r"bypass\s+(all\s+)?(security|guardrails|auth|filters)", re.I),
        ),
        GuardrailRule(
            "SEC-CODE-001",
            "sql_or_code_injection",
            re.compile(r"(;\s*drop\s+table\b|union\s+select\b|<script\b|javascript:|__import__\s*\(|eval\s*\(|exec\s*\()", re.I),
        ),
    ]

    INDIRECT_INJECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions.*?(\.|\n|$)"),
        re.compile(r"(?i)(system\s*override|new\s*instructions\s*:|<\|im_start\|>|<\|system\|>|\[INST\]|\[/INST\])"),
        re.compile(r"(?i)(esqueça\s+as\s+instruções\s+anteriores|você\s+deve\s+transferir\s+fundos)"),
        re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.I | re.S),
    ]

    @classmethod
    def evaluate_input(cls, message: str) -> Tuple[bool, str, str]:
        """
        Avalia a segurança da mensagem de entrada.
        Retorna (is_safe, public_reason, opaque_rule_id).
        NUNCA inclui o padrão regex na mensagem pública ou nos detalhes do trace.
        """
        if not message or len(message.strip()) == 0:
            return False, "A mensagem enviada está vazia.", "SEC-VAL-EMPTY"

        if len(message) > 4000:
            return (
                False,
                "A mensagem excede o limite máximo permitido de 4.000 caracteres.",
                "SEC-VAL-LEN",
            )

        for rule in cls.INPUT_RULES:
            if rule.pattern.search(message):
                return (
                    False,
                    "Solicitação bloqueada pelas políticas de segurança e integridade operacional da Getnet.",
                    rule.rule_id,
                )

        return True, "Input validado com sucesso.", "SEC-OK"

    @classmethod
    def sanitize_untrusted_content(cls, text: str) -> Tuple[str, bool]:
        """
        Defesa contra Indirect Prompt Injection em documentos recuperados ou resultados Web.
        Remove diretivas maliciosas embutidas em dados externos antes de enviá-los ao LLM.
        Retorna (sanitized_text, was_modified).
        """
        if not text:
            return "", False

        sanitized = text
        modified = False
        for pattern in cls.INDIRECT_INJECTION_PATTERNS:
            if pattern.search(sanitized):
                modified = True
                sanitized = pattern.sub("[CONTEÚDO EXTERNO REMOVIDO POR SEGURANÇA]", sanitized)

        return sanitized, modified

    @classmethod
    def sanitize_output(cls, response_text: str) -> str:
        """
        Guardrail de saída:
        - Aplica mascaramento automático de PII (CPF, CNPJ, Cartões, Senhas).
        - Impede vazamento acidental de chaves ou instruções internas.
        """
        if not response_text:
            return ""
        cleaned = PIIRedactor.mask_text(response_text)
        cleaned = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[API_KEY_REDACTED]", cleaned)
        return cleaned
