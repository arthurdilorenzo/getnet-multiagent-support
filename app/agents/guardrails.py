import re
from typing import Tuple

class GuardrailManager:
    """
    Camada de proteção (Guardrails) para detecção de prompt injections,
    conteúdo hostil ou comandos inseguros antes de acionar a orquestração de agentes.
    """
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s+prompt",
        r"você\s+agora\s+é\s+um\s+assistente\s+sem\s+regras",
        r"jailbreak",
        r"bypass\s+security",
        r"drop\s+table",
        r"<script>",
        r"exec\s*\(",
        r"eval\s*\("
    ]

    @classmethod
    def evaluate_input(cls, message: str) -> Tuple[bool, str]:
        """
        Retorna (is_safe, violation_reason).
        Se is_safe == False, a mensagem deve ser bloqueada com resposta segura.
        """
        lower_msg = message.lower()

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, lower_msg, re.IGNORECASE):
                return False, f"Violação de segurança detectada (padrão de injeção ou tentativa de contorno: '{pattern}')"

        # Mensagens vazias ou excessivamente longas
        if len(message.strip()) == 0:
            return False, "Mensagem vazia."
        if len(message) > 4000:
            return False, "Mensagem excede o limite máximo permitido de 4.000 caracteres."

        return True, "Input seguro."
