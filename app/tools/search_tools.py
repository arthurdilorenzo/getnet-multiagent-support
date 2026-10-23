from html.parser import HTMLParser
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import httpx
from pydantic import BaseModel, Field
from app.agents.guardrails import GuardrailManager
from app.config import settings
from app.observability.resilience import ExternalServiceUnavailableError, call_with_resilience
from app.rag.retriever import retriever
from app.tools.registry import tool_registry


# ==============================================================================
# Pydantic Input Schemas
# ==============================================================================
class SearchGetnetKnowledgeInput(BaseModel):
    query: str = Field(..., min_length=2, description="Consulta técnica ou comercial sobre produtos e serviços Getnet")
    top_k: int = Field(default=3, ge=1, le=6, description="Número máximo de trechos relevantes a recuperar")


class WeatherForecastInput(BaseModel):
    city: str = Field(default="Porto Alegre", description="Nome da cidade para previsão meteorológica")
    days_ahead: int = Field(default=1, ge=0, le=5, description="Dias à frente (0=hoje, 1=amanhã)")


class CurrencyExchangeInput(BaseModel):
    base_currency: str = Field(default="EUR", min_length=3, max_length=3, description="Moeda de origem ISO-4217 (ex: EUR, USD, GBP)")
    target_currency: str = Field(default="BRL", min_length=3, max_length=3, description="Moeda de destino ISO-4217 (ex: BRL)")


class GeneralWebSearchInput(BaseModel):
    query: str = Field(..., min_length=2, description="Termos de pesquisa para busca web geral")
    max_results: int = Field(default=3, ge=1, le=5, description="Quantidade máxima de resultados")


# ==============================================================================
# WMO Weather Code Mapping (Official Open-Meteo WMO 4677 Standard)
# ==============================================================================
WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Céu limpo e ensolarado",
    1: "Predominantemente limpo",
    2: "Parcialmente nublado",
    3: "Céu encoberto / nublado",
    45: "presença de nevoeiro",
    48: "Nevoeiro com formação de geada",
    51: "Chuvisco leve",
    53: "Chuvisco moderado",
    55: "Chuvisco intenso",
    61: "Chuva fraca",
    63: "Chuva moderada",
    65: "Chuva forte",
    71: "Queda de neve fraca",
    80: "Pancadas de chuva leves",
    81: "Pancadas de chuva moderadas",
    82: "Pancadas de chuva fortes",
    95: "Tempestade com trovoadas",
    96: "Tempestade com possibilidade de granizo leve",
    99: "Tempestade severa com granizo",
}

KNOWN_CITY_COORDS: Dict[str, Tuple[float, float, str]] = {
    "porto alegre": (-30.0346, -51.2177, "Porto Alegre, RS"),
    "sao paulo": (-23.5505, -46.6333, "São Paulo, SP"),
    "são paulo": (-23.5505, -46.6333, "São Paulo, SP"),
    "rio de janeiro": (-22.9068, -43.1729, "Rio de Janeiro, RJ"),
    "curitiba": (-25.4284, -49.2733, "Curitiba, PR"),
    "belo horizonte": (-19.9167, -43.9345, "Belo Horizonte, MG"),
    "brasilia": (-15.7975, -47.8919, "Brasília, DF"),
    "brasília": (-15.7975, -47.8919, "Brasília, DF"),
}


def decode_wmo_weather_code(code: Optional[int]) -> str:
    if code is None:
        return "Condição meteorológica variável"
    return WMO_WEATHER_CODES.get(int(code), f"Condição meteorológica WMO código {code}")


# ==============================================================================
# Tool 1: Hybrid RAG Knowledge Search
# ==============================================================================
def search_getnet_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Executa busca híbrida (BM25 + Dense Embeddings + RRF) sobre o corpus oficial Getnet.
    """
    outcome = retriever.retrieve_hybrid(query=query, top_k=top_k)
    return {
        "abstained": outcome["abstained"],
        "abstention_reason": outcome["abstention_reason"],
        "max_relevance_score": outcome["max_relevance_score"],
        "chunks_found": len(outcome["results"]),
        "chunks": outcome["results"],
        "citations": [c.model_dump() for c in outcome["citations"]],
    }


# ==============================================================================
# Tool 2: Live Weather Forecast (Zero Fabricated Fallbacks)
# ==============================================================================
def search_weather(city: str = "Porto Alegre", days_ahead: int = 1) -> Dict[str, Any]:
    """
    Consulta previsão meteorológica real via Open-Meteo API sob proteção de Circuit Breaker + Retries.
    NUNCA inventa temperaturas nem fontes falsas se a API estiver indisponível.
    """
    clean_city = (city or "Porto Alegre").strip()
    city_key = clean_city.lower()

    def _fetch_open_meteo() -> Dict[str, Any]:
        with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            if city_key in KNOWN_CITY_COORDS:
                lat, lon, resolved_name = KNOWN_CITY_COORDS[city_key]
            else:
                geo_url = (
                    f"https://geocoding-api.open-meteo.com/v1/search"
                    f"?name={quote_plus(clean_city)}&count=1&language=pt&format=json"
                )
                geo_resp = client.get(geo_url)
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()
                results = geo_data.get("results") or []
                if not results:
                    raise ValueError(f"Cidade '{clean_city}' não encontrada no serviço de geocodificação.")
                first = results[0]
                lat = float(first["latitude"])
                lon = float(first["longitude"])
                resolved_name = f"{first.get('name', clean_city)}, {first.get('admin1', '')}".strip(", ")

            forecast_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
                f"&timezone=America%2FSao_Paulo"
            )
            resp = client.get(forecast_url)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})

            max_list = daily.get("temperature_2m_max") or []
            min_list = daily.get("temperature_2m_min") or []
            precip_list = daily.get("precipitation_probability_max") or []
            code_list = daily.get("weathercode") or []
            dates_list = daily.get("time") or []

            idx = min(max(0, days_ahead), len(max_list) - 1) if max_list else -1
            if idx < 0:
                raise RuntimeError("Payload meteorológico da Open-Meteo não retornou séries diárias.")

            wmo_code = code_list[idx] if idx < len(code_list) else None
            condition_desc = decode_wmo_weather_code(wmo_code)
            target_date = dates_list[idx] if idx < len(dates_list) else ""

            return {
                "source": "Open-Meteo Weather Forecast API",
                "source_url": "https://api.open-meteo.com/v1/forecast",
                "city": resolved_name,
                "target_date": target_date,
                "forecast_horizon": "Amanhã" if days_ahead == 1 else f"D+{days_ahead}",
                "max_temp_celsius": float(max_list[idx]),
                "min_temp_celsius": float(min_list[idx]),
                "rain_probability_percent": int(precip_list[idx]) if idx < len(precip_list) and precip_list[idx] is not None else 0,
                "wmo_weather_code": wmo_code,
                "condition": condition_desc,
                "live": True,
            }

    return call_with_resilience("weather_api", _fetch_open_meteo)


# ==============================================================================
# Tool 3: Live Currency Exchange Rate (Zero Fabricated Fallbacks)
# ==============================================================================
def search_currency_rate(base_currency: str = "EUR", target_currency: str = "BRL") -> Dict[str, Any]:
    """
    Consulta cotação de câmbio comercial em tempo real (AwesomeAPI com failover para Frankfurter/ECB)
    sob proteção de Circuit Breaker + Retries.
    NUNCA inventa valores (ex: R$ 6,18) nem atribui a fontes fictícias se os provedores falharem.
    """
    base = (base_currency or "EUR").strip().upper()
    target = (target_currency or "BRL").strip().upper()

    def _fetch_live_rate() -> Dict[str, Any]:
        with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            # Provedor Primário: AwesomeAPI
            try:
                awesome_url = f"https://economia.awesomeapi.com.br/last/{base}-{target}"
                resp = client.get(awesome_url)
                if resp.status_code == 200:
                    data = resp.json()
                    pair_data = data.get(f"{base}{target}")
                    if pair_data and "bid" in pair_data:
                        return {
                            "source": "AwesomeAPI (Mercado de Câmbio em Tempo Real)",
                            "source_url": awesome_url,
                            "currency_pair": f"{base}/{target}",
                            "current_rate_brl": round(float(pair_data["bid"]), 4),
                            "high_today_brl": round(float(pair_data.get("high", pair_data["bid"])), 4),
                            "low_today_brl": round(float(pair_data.get("low", pair_data["bid"])), 4),
                            "updated_at": pair_data.get("create_date", ""),
                            "live": True,
                        }
            except Exception:
                pass

            # Provedor Secundário Real (Failover): Frankfurter API (European Central Bank rates)
            frankfurter_url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
            f_resp = client.get(frankfurter_url)
            f_resp.raise_for_status()
            f_data = f_resp.json()
            rates = f_data.get("rates") or {}
            if target not in rates:
                raise RuntimeError(f"Par de moedas {base}/{target} não retornado pelo provedor secundário.")
            rate_val = float(rates[target])
            return {
                "source": "Frankfurter API (European Central Bank Reference Rates)",
                "source_url": frankfurter_url,
                "currency_pair": f"{base}/{target}",
                "current_rate_brl": round(rate_val, 4),
                "high_today_brl": round(rate_val, 4),
                "low_today_brl": round(rate_val, 4),
                "updated_at": f_data.get("date", ""),
                "live": True,
            }

    return call_with_resilience("currency_api", _fetch_live_rate)


# ==============================================================================
# Tool 4: Real General-Purpose Web Search (DuckDuckGo HTML + Wikipedia API)
# ==============================================================================
class _DDGResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_snippet = False
        self.snippets: List[str] = []
        self._buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)
        cls_name = attr_dict.get("class", "")
        if "result__snippet" in cls_name:
            self.in_snippet = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_snippet and tag in ("a", "td", "div"):
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.snippets.append(text)
            self.in_snippet = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self.in_snippet:
            self._buf.append(data)


def general_web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Executa busca web geral real via HTTP (Wikipedia Opensearch API + DuckDuckGo HTML),
    aplicando sanitização contra Indirect Prompt Injection nos snippets retornados.
    Se os provedores externos estiverem inacessíveis, levanta ExternalServiceUnavailableError.
    """
    clean_query = query.strip()
    lower = clean_query.lower()

    # Despacho direto caso o LLM chame general_web_search para clima ou câmbio
    if any(w in lower for w in ["clima", "previsão do tempo", "weather", "temperatura"]):
        city = "Porto Alegre"
        if "são paulo" in lower or "sao paulo" in lower:
            city = "São Paulo"
        elif "rio de janeiro" in lower:
            city = "Rio de Janeiro"
        return search_weather(city=city, days_ahead=1)

    if any(w in lower for w in ["euro", "câmbio", "exchange rate", "cotação", "dólar", "dolar", "usd"]):
        base = "USD" if ("dólar" in lower or "dolar" in lower or "usd" in lower) and "euro" not in lower else "EUR"
        return search_currency_rate(base_currency=base, target_currency="BRL")

    def _perform_search() -> Dict[str, Any]:
        results: List[Dict[str, str]] = []
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GetnetFDEWebSearch/2.0)"}

        with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS, follow_redirects=True, headers=headers) as client:
            # 1. Tenta Wikipedia MediaWiki Search API (PT e EN)
            for lang in ("pt", "en"):
                if len(results) >= max_results:
                    break
                wiki_url = (
                    f"https://{lang}.wikipedia.org/w/api.php"
                    f"?action=query&list=search&srsearch={quote_plus(clean_query)}&utf8=&format=json&srlimit={max_results}"
                )
                try:
                    w_resp = client.get(wiki_url)
                    if w_resp.status_code == 200:
                        items = w_resp.json().get("query", {}).get("search", [])
                        for item in items[:max_results]:
                            raw_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                            sanitized_snippet, _ = GuardrailManager.sanitize_untrusted_content(raw_snippet)
                            title = item.get("title", "")
                            page_url = f"https://{lang}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                            if sanitized_snippet:
                                results.append(
                                    {
                                        "title": title,
                                        "snippet": sanitized_snippet,
                                        "url": page_url,
                                    }
                                )
                except Exception:
                    pass

            # 2. Complementa com DuckDuckGo HTML se necessário
            if not results:
                ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(clean_query)}"
                d_resp = client.get(ddg_url)
                d_resp.raise_for_status()
                parser = _DDGResultParser()
                parser.feed(d_resp.text)
                for idx, snip in enumerate(parser.snippets[:max_results], start=1):
                    sanitized_snip, _ = GuardrailManager.sanitize_untrusted_content(snip)
                    results.append(
                        {
                            "title": f"Resultado Web #{idx}",
                            "snippet": sanitized_snip,
                            "url": ddg_url,
                        }
                    )

        if not results:
            raise ExternalServiceUnavailableError(
                f"Nenhum resultado público verificável encontrado na busca web para '{clean_query}'."
            )

        return {
            "source": "Live Web Search (Wikipedia / DuckDuckGo)",
            "query": clean_query,
            "results_count": len(results),
            "results": results,
            "live": True,
        }

    return call_with_resilience("web_search_api", _perform_search)


# ==============================================================================
# Register Knowledge & Search Tools in Global ToolRegistry
# ==============================================================================
def register_search_tools() -> None:
    tool_registry.register(
        name="search_getnet_knowledge",
        description="Executa busca RAG híbrida (Okapi BM25 + Embeddings Vetoriais Densos + RRF) na base oficial Getnet (site.getnet.com.br) retornando trechos fundamentados e citações verificáveis.",
        args_schema=SearchGetnetKnowledgeInput,
        handler=search_getnet_knowledge,
        requires_tenant_context=False,
        is_write_action=False,
    )
    tool_registry.register(
        name="get_weather_forecast",
        description="Consulta previsão do tempo em tempo real para qualquer cidade via Open-Meteo API com código meteorológico WMO real.",
        args_schema=WeatherForecastInput,
        handler=search_weather,
        requires_tenant_context=False,
        is_write_action=False,
    )
    tool_registry.register(
        name="get_currency_exchange_rate",
        description="Consulta cotação comercial em tempo real de moedas internacionais (EUR, USD, GBP) frente ao Real (BRL) via AwesomeAPI e Frankfurter ECB.",
        args_schema=CurrencyExchangeInput,
        handler=search_currency_rate,
        requires_tenant_context=False,
        is_write_action=False,
    )
    tool_registry.register(
        name="web_search_general",
        description="Realiza pesquisa em fontes públicas da web para perguntas gerais de conhecimento externo que não dizem respeito à Getnet.",
        args_schema=GeneralWebSearchInput,
        handler=general_web_search,
        requires_tenant_context=False,
        is_write_action=False,
    )


register_search_tools()
