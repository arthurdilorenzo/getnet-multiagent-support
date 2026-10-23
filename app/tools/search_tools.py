from html.parser import HTMLParser
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import math
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
        return "Condição meteorológica não informada pelo provedor"
    return WMO_WEATHER_CODES.get(int(code), f"Condição meteorológica WMO código {code}")


def _number(value: Any, label: str, minimum: float, maximum: float, optional: bool = False) -> Optional[float]:
    """Missing is unknown; malformed or non-finite observations are rejected."""
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label}: valor booleano inválido.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label}: valor não finito ou fora do intervalo válido.")
    return number


def _iso_date(value: Any, label: str) -> str:
    text = str(value or "")
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label}: data ausente ou inválida.") from exc
    return text


def _today_local() -> date:
    return datetime.now(ZoneInfo('America/Sao_Paulo')).date()


# ==============================================================================
# Tool 1: Hybrid RAG Knowledge Search
# ==============================================================================
def search_getnet_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Return raw candidates separately; only the specialist authorizes evidence."""
    candidates = retriever.retrieve_candidates(query=query)
    if settings.GEMINI_API_KEY:
        # Native semantics must see raw evidence even when the legacy grammar
        # would abstain. Candidates themselves never count as approved facts.
        return {**candidates,'abstained':True,'abstention_reason':'Native semantic authorization pending',
                'max_relevance_score':max((c.get('relevance_score',0.0) for c in candidates['candidate_chunks']),default=0.0),
                'chunks_found':0,'chunks':[],'citations':[],'answerability':{'answerable':False,'method':'pending_native_semantics'}}
    outcome = retriever.retrieve_hybrid(query=query, top_k=top_k)
    return {**candidates,
        "abstained": outcome["abstained"], "abstention_reason": outcome["abstention_reason"],
        "max_relevance_score": outcome["max_relevance_score"], "chunks_found": len(outcome["results"]),
        "chunks": outcome["results"], "citations": [c.model_dump() for c in outcome["citations"]],
        "answerability": outcome.get("answerability", {}),
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
                lat = _number(first["latitude"], "latitude", -90, 90)
                lon = _number(first["longitude"], "longitude", -180, 180)
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

            idx = days_ahead
            if not 0 <= idx < len(dates_list):
                raise RuntimeError("O provedor não retornou a data solicitada; não substituímos por outro dia.")

            def observation(values, label, minimum, maximum):
                return _number(values[idx] if idx < len(values) else None, label, minimum, maximum, optional=True)

            max_t = observation(max_list, "temperatura máxima", -100, 65)
            min_t = observation(min_list, "temperatura mínima", -100, 65)
            rain = observation(precip_list, "probabilidade de precipitação", 0, 100)
            if max_t is not None and min_t is not None and min_t > max_t:
                raise ValueError("Temperatura mínima superior à máxima.")

            wmo_code = code_list[idx] if idx < len(code_list) else None
            if wmo_code is not None:
                numeric_code = _number(wmo_code, "código WMO", 0, 99)
                if numeric_code != int(numeric_code):
                    raise ValueError("Código WMO deve ser inteiro.")
                wmo_code = int(numeric_code)
            condition_desc = decode_wmo_weather_code(wmo_code)
            target_date = _iso_date(dates_list[idx], "data meteorológica")
            requested_date=_today_local()+timedelta(days=days_ahead)
            if target_date[:10]!=requested_date.isoformat():
                raise ValueError('A data retornada não corresponde ao dia solicitado; previsão antiga não atende a consulta.')

            return {
                "source": "Open-Meteo Weather Forecast API",
                "source_url": forecast_url,
                "city": resolved_name,
                "target_date": target_date,
                "forecast_horizon": "Hoje" if days_ahead == 0 else "Amanhã" if days_ahead == 1 else f"D+{days_ahead}",
                "max_temp_celsius": max_t,
                "min_temp_celsius": min_t,
                "rain_probability_percent": rain,
                "wmo_weather_code": wmo_code,
                "condition": condition_desc,
                "live": True,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }

    return call_with_resilience("weather_api", _fetch_open_meteo)


# ==============================================================================
# Tool 3: Live Currency Exchange Rate (Zero Fabricated Fallbacks)
# ==============================================================================
def search_currency_rate(base_currency: str = "EUR", target_currency: str = "BRL") -> Dict[str, Any]:
    """
    Consulta cotação informada pela AwesomeAPI com failover para referência diária Frankfurter.
    sob proteção de Circuit Breaker + Retries.
    NUNCA inventa valores (ex: R$ 6,18) nem atribui a fontes fictícias se os provedores falharem.
    """
    base = (base_currency or "EUR").strip().upper()
    target = (target_currency or "BRL").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", base) or not re.fullmatch(r"[A-Z]{3}", target):
        raise ValueError("Moeda deve ser um código ISO de três letras.")

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
                        if (("code" in pair_data and pair_data["code"] != base)
                                or ("codein" in pair_data and pair_data["codein"] != target)):
                            raise ValueError("Identidade do par cambial divergente no provedor primário.")
                        rate = _number(pair_data["bid"], "cotação", 1e-10, 1e10)
                        high = _number(pair_data.get("high"), "máxima", 1e-10, 1e10, optional=True)
                        low = _number(pair_data.get("low"), "mínima", 1e-10, 1e10, optional=True)
                        if high is not None and low is not None and high < low:
                            raise ValueError("Máxima cambial inferior à mínima.")
                        observed=pair_data.get('create_date') or None
                        try:
                            is_fresh=bool(observed and date.fromisoformat(str(observed)[:10])==_today_local())
                        except ValueError:
                            observed=None
                            is_fresh=False
                        return {
                            "source": "AwesomeAPI (cotação informada pelo provedor)",
                            "source_url": awesome_url,
                            "currency_pair": f"{base}/{target}",
                            "current_rate_brl": rate,
                            "high_today_brl": high,
                            "low_today_brl": low,
                            "updated_at": observed,
                            "rate_type": "market_quote" if is_fresh else "historical_market_quote" if observed else "undated_market_quote",
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "live": is_fresh,
                        }
            except Exception:
                pass

            # Provedor secundário: referência diária Frankfurter, sem inferir
            # um banco-fonte que o payload não identificou.
            frankfurter_url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
            f_resp = client.get(frankfurter_url)
            f_resp.raise_for_status()
            f_data = f_resp.json()
            if f_data.get("base") != base:
                raise ValueError("Moeda-base ausente ou divergente no provedor secundário.")
            if "amount" not in f_data or f_data["amount"] is None:
                raise ValueError("Unidade cambial ausente no provedor secundário.")
            amount = _number(f_data["amount"], "unidade cambial", 1e-10, 1e10)
            if amount != 1:
                raise ValueError("Unidade cambial divergente: a consulta exige uma unidade da moeda-base.")
            rates = f_data.get("rates") or {}
            if target not in rates:
                raise RuntimeError(f"Par de moedas {base}/{target} não retornado pelo provedor secundário.")
            rate_val = _number(rates[target], "cotação de referência", 1e-10, 1e10)
            return {
                "source": "Frankfurter API (referência diária)",
                "source_url": frankfurter_url,
                "currency_pair": f"{base}/{target}",
                "current_rate_brl": rate_val,
                "high_today_brl": None,
                "low_today_brl": None,
                "updated_at": _iso_date(f_data.get("date"), "data da referência cambial"),
                "rate_type": "daily_reference",
                "live": False,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
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
    # Structured weather/currency calls belong to the explicit slot extractor.
    # A general search never silently substitutes Porto Alegre or EUR for an
    # unrecognized city/currency from the query.

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
        description="Busca no corpus de produtos Getnet por BM25 com expansão léxica, projeção vetorial por hashing/âncoras conceituais e RRF, retornando trechos e URLs de origem.",
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
        description="Consulta cotação de um par ISO de moedas via AwesomeAPI, com fallback Frankfurter de referência diária, data e campos ausentes explícitos.",
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
