import json
import re
from typing import Any, Dict, List, Optional
from app.config import settings
from app.observability.logger import logger
from app.observability.resilience import call_with_resilience
from app.observability.tracer import SystemTracer
from app.rag.embeddings import BilingualDenseEmbedder
from app.schemas import Citation, SpecialistOutput, ToolCallSpec, ToolExecutionResult
from app.security.auth import SecurityContext
from app.security.pii import PIIRedactor
from app.tools.registry import tool_registry
import app.tools.search_tools  # noqa: F401 - garante registro das tools de busca e RAG

KNOWLEDGE_SYSTEM_PROMPT = """Você é o Knowledge & Web Search Agent oficial da Getnet.
Diretrizes inegociáveis de Grounding e Confiabilidade:
1. Utilize APENAS os dados retornados pelas ferramentas (`search_getnet_knowledge`, `get_weather_forecast`, `get_currency_exchange_rate`, `web_search_general`).
2. NUNCA invente taxas, prazos, temperaturas ou cotações de moedas que não estejam explicitamente nos resultados das ferramentas.
3. Se uma ferramenta externa falhar ou retornar indisponibilidade, informe honestamente ao usuário que o serviço externo de consulta em tempo real está temporariamente indisponível.
4. Sempre inclua citações verificáveis ao final da resposta no formato `[Fonte: <título> (<url>)]`.
5. Responda de maneira estruturada, clara e em Português do Brasil.
"""


class KnowledgeAgent:
    """
    Agente 2: Knowledge & External Search Agent.
    Seleção determinística de tools e extração de argumentos por regras locais.
    Executa via ToolRegistry e sintetiza os resultados por Gemini opcional ou
    extração local. O RAG aplica threshold de abstention e inclui URLs de origem;
    a síntese LLM não oferece garantia universal de grounding.
    """

    AVAILABLE_TOOLS = [
        "search_getnet_knowledge",
        "get_weather_forecast",
        "get_currency_exchange_rate",
        "web_search_general",
    ]

    def __init__(self, gemini_client: Optional[Any] = None) -> None:
        self.client = gemini_client

    def run(
        self,
        message: str,
        security_context: SecurityContext,
        is_web_search: bool = False,
        session_id: str = "",
        conversation_context: Optional[Dict[str, Any]] = None,
        tracer: Optional[SystemTracer] = None,
    ) -> SpecialistOutput:
        trace_id = tracer.trace_id if tracer else ""
        contextual_query = self._enrich_query_with_memory(message, conversation_context)

        # 1. Seleção de tools e argumentos por slot extractor determinístico.
        planned_calls = self._plan_tool_calls(
            message=contextual_query,
            is_web_search=is_web_search,
            tracer=tracer,
        )

        tools_called: List[str] = []
        tool_results: List[ToolExecutionResult] = []
        citations: List[Citation] = []
        sources: List[str] = []

        # 2. Loop de Execução de Tools via ToolRegistry
        for call_spec in planned_calls[: settings.MAX_TOOL_TURNS]:
            tools_called.append(call_spec.tool_name)
            res = tool_registry.execute(
                tool_name=call_spec.tool_name,
                raw_arguments=call_spec.arguments,
                security_context=security_context,
                session_id=session_id,
                trace_id=trace_id,
            )
            tool_results.append(res)

            if tracer:
                tracer.add_step(
                    agent="KNOWLEDGE_AGENT_TOOL",
                    action=f"call:{call_spec.tool_name}",
                    status="success" if res.success else "error",
                    details={
                        "arguments": call_spec.arguments,
                        "success": res.success,
                        "error_code": res.error_code,
                        "execution_ms": res.execution_ms,
                    },
                )

            if res.success and isinstance(res.data, dict):
                if call_spec.tool_name == "search_getnet_knowledge":
                    for raw_cit in res.data.get("citations", []):
                        cit = Citation.model_validate(raw_cit)
                        citations.append(cit)
                        if cit.source not in sources:
                            sources.append(cit.source)
                        if cit.source_url and cit.source_url not in sources:
                            sources.append(cit.source_url)
                else:
                    src = res.data.get("source")
                    src_url = res.data.get("source_url")
                    if src and src not in sources:
                        sources.append(src)
                    if src_url and src_url not in sources:
                        sources.append(src_url)

        # 3. Avaliação de Falha de Ferramenta Externa ou Abstention do RAG
        primary_res = tool_results[0] if tool_results else None
        if not primary_res:
            return SpecialistOutput(
                agent_name="WEB_SEARCH_AGENT" if is_web_search else "KNOWLEDGE_AGENT",
                response="Não foi possível processar a consulta por ausência de ferramentas aplicáveis.",
                confidence=0.0,
                abstained=True,
                abstention_reason="Nenhuma tool selecionada.",
            )

        if not primary_res.success:
            # Falha honesta: NUNCA inventa cotação, clima ou dados de fonte falsa!
            honest_msg = (
                f"No momento não foi possível obter os dados atualizados em tempo real porque o serviço externo "
                f"consultado (`{primary_res.tool_name}`) está temporariamente indisponível "
                f"({primary_res.error_code}: {primary_res.error_message}).\n\n"
                f"Por política de integridade e confiabilidade, não exibimos estimativas ou valores não verificados. "
                f"Por favor, tente novamente em alguns instantes."
            )
            return SpecialistOutput(
                agent_name="WEB_SEARCH_AGENT" if is_web_search else "KNOWLEDGE_AGENT",
                response=honest_msg,
                confidence=0.0,
                abstained=True,
                abstention_reason=primary_res.error_message or "Serviço externo indisponível",
                tools_called=tools_called,
                tool_results=tool_results,
                sources=sources,
            )

        if (
            primary_res.tool_name == "search_getnet_knowledge"
            and isinstance(primary_res.data, dict)
            and primary_res.data.get("abstained")
        ):
            max_rel = float(primary_res.data.get("max_relevance_score", 0.0))
            reason = str(primary_res.data.get("abstention_reason", "Cobertura insuficiente na base Getnet."))
            return SpecialistOutput(
                agent_name="KNOWLEDGE_AGENT",
                response=(
                    "Não encontrei informações suficientes sobre esse tema na documentação oficial da Getnet "
                    "para responder com segurança e precisão."
                ),
                confidence=max_rel,
                abstained=True,
                abstention_reason=reason,
                tools_called=tools_called,
                tool_results=tool_results,
                sources=sources,
            )

        # 4. Síntese Grounded a partir dos resultados reais das Tools
        synthesized_text = self._synthesize_from_tool_results(
            user_message=message,
            tool_results=tool_results,
            citations=citations,
            tracer=tracer,
        )

        confidence = 0.92
        if primary_res.tool_name == "search_getnet_knowledge" and isinstance(primary_res.data, dict):
            confidence = max(0.65, min(0.98, float(primary_res.data.get("max_relevance_score", 0.85))))

        return SpecialistOutput(
            agent_name="WEB_SEARCH_AGENT" if is_web_search else "KNOWLEDGE_AGENT",
            response=synthesized_text,
            confidence=round(confidence, 4),
            abstained=False,
            tools_called=tools_called,
            tool_results=tool_results,
            sources=sources,
            citations=citations,
            structured_data=primary_res.data or {},
        )

    def handle(self, message: str, is_web_search: bool = False, user_id: str = "cliente1988") -> Dict[str, Any]:
        """
        Interface de compatibilidade direta.
        """
        sec_ctx = SecurityContext(authenticated_user_id=user_id)
        out = self.run(message=message, security_context=sec_ctx, is_web_search=is_web_search)
        return {
            "response": out.response,
            "sources": out.sources,
            "citations": [c.model_dump() for c in out.citations],
            "tools_called": out.tools_called,
            "abstained": out.abstained,
            "confidence": out.confidence,
        }

    @staticmethod
    def _enrich_query_with_memory(
        message: str, conversation_context: Optional[Dict[str, Any]]
    ) -> str:
        if not conversation_context:
            return message
        words = message.strip().split()
        norm = BilingualDenseEmbedder.normalize(message)
        # Se a pergunta usa pronomes ou é curta de continuação ("e qual delas...", "como funciona isso?", "qual a taxa dele?")
        has_pronoun = any(
            p in norm.split()
            for p in ("dela", "delas", "dele", "deles", "isso", "isto", "essa", "esse", "it", "them", "that")
        )
        if len(words) <= 9 or has_pronoun:
            recent = conversation_context.get("recent_messages", [])
            prev_user_msgs = [m["content"] for m in recent if m.get("role") == "user" and m.get("content") != message]
            if prev_user_msgs:
                return f"{prev_user_msgs[-1]} | Pergunta de acompanhamento: {message}"
        return message

    def _plan_tool_calls(
        self,
        message: str,
        is_web_search: bool,
        tracer: Optional[SystemTracer] = None,
    ) -> List[ToolCallSpec]:
        """
        Seleciona a tool e extrai os argumentos estruturados.
        """
        norm = BilingualDenseEmbedder.normalize(message)
        lower = message.lower()

        if is_web_search:
            # Verifica se é previsão meteorológica
            if any(w in norm for w in ["weather", "previsao", "tempo", "clima", "temperatura", "chover"]):
                city = self._extract_city(message)
                days = 0 if ("hoje" in norm or "today" in norm) and "amanha" not in norm and "tomorrow" not in norm else 1
                return [
                    ToolCallSpec(
                        tool_name="get_weather_forecast",
                        arguments={"city": city, "days_ahead": days},
                        rationale="Consulta meteorológica estruturada via Open-Meteo.",
                    )
                ]

            # Verifica se é cotação cambial
            if any(w in norm for w in ["euro", "eur", "dolar", "usd", "gbp", "libra", "cambio", "cotacao", "exchange"]):
                base = "EUR"
                if any(w in norm for w in ["dolar", "usd", "dollar"]) and "euro" not in norm:
                    base = "USD"
                elif any(w in norm for w in ["libra", "gbp", "pound"]):
                    base = "GBP"
                return [
                    ToolCallSpec(
                        tool_name="get_currency_exchange_rate",
                        arguments={"base_currency": base, "target_currency": "BRL"},
                        rationale=f"Consulta de taxa de câmbio em tempo real para {base}/BRL.",
                    )
                ]

            # Caso contrário, busca web geral
            return [
                ToolCallSpec(
                    tool_name="web_search_general",
                    arguments={"query": message, "max_results": 3},
                    rationale="Pesquisa em fontes públicas da web.",
                )
            ]

        return [
            ToolCallSpec(
                tool_name="search_getnet_knowledge",
                arguments={"query": message, "top_k": settings.RAG_TOP_K},
                rationale="Busca BM25 com expansão léxica, projeção por hashing/âncoras e RRF no corpus Getnet.",
            )
        ]

    @staticmethod
    def _extract_city(message: str) -> str:
        lower = message.lower()
        for known in ("porto alegre", "são paulo", "sao paulo", "rio de janeiro", "curitiba", "belo horizonte", "brasília", "brasilia"):
            if known in lower:
                return known.title()
        # Extração por padrão "em <Cidade>" ou "in <City>"
        match = re.search(
            r"\b(?:em|in|para)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Úa-zà-ú]+){0,2})",
            message,
        )
        if match:
            candidate = match.group(1).strip()
            for stop in ("amanhã", "amanha", "hoje", "tomorrow", "today"):
                candidate = re.sub(rf"\b{stop}\b", "", candidate, flags=re.I).strip()
            if candidate:
                return candidate
        return "Porto Alegre"

    def _synthesize_from_tool_results(
        self,
        user_message: str,
        tool_results: List[ToolExecutionResult],
        citations: List[Citation],
        tracer: Optional[SystemTracer] = None,
    ) -> str:
        primary = tool_results[0]
        data = primary.data or {}

        # Se Gemini API Key estiver ativa, gera síntese natural grounded no payload das tools
        if self.client and settings.GEMINI_API_KEY:
            try:
                prompt = (
                    f"Pergunta do usuário:\n\"{PIIRedactor.mask_text(user_message)}\"\n\n"
                    f"Resultado estruturado da ferramenta `{primary.tool_name}`:\n"
                    f"{json.dumps(data, ensure_ascii=False, indent=2)}\n\n"
                    f"Elabore uma resposta precisa baseada exclusivamente nos dados acima e inclua as fontes/URLs."
                )

                def _call_llm() -> Any:
                    return self.client.models.generate_content(
                        model=settings.GEMINI_MODEL,
                        contents=prompt,
                        config={"system_instruction": KNOWLEDGE_SYSTEM_PROMPT, "temperature": 0.1},
                    )

                resp = call_with_resilience("gemini_api", _call_llm, max_retries=1)
                if resp and resp.text:
                    if tracer:
                        tracer.record_tokens(prompt_text=prompt, completion_text=resp.text)
                    return resp.text.strip()
            except Exception as exc:
                logger.warning("knowledge_llm_synthesis_fallback", error=str(exc))

        # Síntese extrativa local dos dados retornados pela tool.
        if primary.tool_name == "get_weather_forecast":
            city = data.get("city", "Porto Alegre, RS")
            horizon = data.get("forecast_horizon", "Amanhã")
            t_date = data.get("target_date", "")
            min_t = data.get("min_temp_celsius")
            max_t = data.get("max_temp_celsius")
            rain = data.get("rain_probability_percent", 0)
            cond = data.get("condition", "")
            src = data.get("source", "Open-Meteo")
            src_url = data.get("source_url", "https://api.open-meteo.com/v1/forecast")
            date_label = f" ({t_date})" if t_date else ""
            out = (
                f"A previsão do tempo para **{horizon.lower()}{date_label}** em **{city}** indica:\n\n"
                f"- **Temperatura:** mínima de **{min_t}°C** e máxima de **{max_t}°C**\n"
                f"- **Condição meteorológica:** {cond}\n"
                f"- **Probabilidade máxima de precipitação:** {rain}%\n\n"
                f"[Fonte: {src} ({src_url})]"
            )
            if tracer:
                tracer.record_tokens(prompt_text=user_message, completion_text=out)
            return out

        if primary.tool_name == "get_currency_exchange_rate":
            pair = data.get("currency_pair", "EUR/BRL")
            rate = data.get("current_rate_brl")
            high = data.get("high_today_brl")
            low = data.get("low_today_brl")
            updated = data.get("updated_at", "tempo real")
            src = data.get("source", "")
            src_url = data.get("source_url", "")
            out = (
                f"A cotação comercial atual para **{pair}** está em **R$ {rate:.2f}** (taxa exata: `{rate:.4f}`).\n\n"
                f"- **Máxima do dia:** R$ {high:.2f}\n"
                f"- **Mínima do dia:** R$ {low:.2f}\n"
                f"- **Última atualização:** {updated}\n\n"
                f"[Fonte: {src} ({src_url})]"
            )
            if tracer:
                tracer.record_tokens(prompt_text=user_message, completion_text=out)
            return out

        if primary.tool_name == "web_search_general":
            results = data.get("results", [])
            lines = [f"Resultados verificados na busca web para **\"{data.get('query', user_message)}\"**:\n"]
            for r in results:
                lines.append(f"- **{r.get('title')}:** {r.get('snippet')} ([Fonte]({r.get('url')}))")
            out = "\n".join(lines)
            if tracer:
                tracer.record_tokens(prompt_text=user_message, completion_text=out)
            return out

        # Síntese Extrativa-Estruturada a partir dos Chunks recuperados pelo RAG Híbrido
        chunks: List[Dict[str, Any]] = data.get("chunks", [])
        if not chunks:
            return "Não foram encontrados trechos correspondentes na documentação oficial da Getnet."

        top_chunk = chunks[0]
        raw_content = top_chunk.get("content", "").strip()
        # Remove apenas a primeira linha de cabeçalho `## ...` para apresentar o corpo limpo com o título em negrito
        content_lines = raw_content.splitlines()
        title_line = top_chunk.get("title", "Documentação Oficial Getnet")
        if content_lines and content_lines[0].lstrip().startswith("#"):
            body_text = "\n".join(content_lines[1:]).strip()
        else:
            body_text = raw_content

        # Se o segundo chunk pertencer ao mesmo documento/tema e complementar a pergunta, agrega os pontos-chave
        supplementary_text = ""
        if len(chunks) > 1 and chunks[1].get("relevance_score", 0) >= 0.52:
            second = chunks[1]
            if second.get("chunk_id") != top_chunk.get("chunk_id") and second.get("source") == top_chunk.get("source"):
                sec_lines = [
                    l.strip()
                    for l in second.get("content", "").splitlines()
                    if l.strip() and not l.strip().startswith("#")
                ]
                if sec_lines:
                    supplementary_text = f"\n\n**{second.get('title')}:**\n" + "\n".join(sec_lines[:4])

        citation_lines: List[str] = []
        seen_urls = set()
        for c in citations[:2]:
            key = f"{c.source}:{c.source_url}"
            if key not in seen_urls:
                seen_urls.add(key)
                citation_lines.append(
                    f"[Fonte: {c.title} — {c.source} ({c.source_url})]"
                )

        citations_block = "\n".join(citation_lines)
        out = f"**{title_line}**\n\n{body_text}{supplementary_text}\n\n{citations_block}".strip()
        if tracer:
            tracer.record_tokens(prompt_text=raw_content, completion_text=out)
        return out
