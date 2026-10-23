"""Native Web proposals must respect verified Getnet Pix evidence and private scope."""
import pytest
from google.genai import types

from app.agents.router_agent import RouterAgent
from app.config import settings


class WebProposal:
    def __init__(self, assignments):
        self.assignments = assignments
        self.models = self

    def generate_content(self, **kwargs):
        parts = [types.Part(function_call=types.FunctionCall(
            name="WEB_SEARCH_AGENT",
            args={"sub_query": query, "rationale": "Model proposed public web", "request_unit_ids": ids},
        )) for query, ids in self.assignments]
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts), finish_reason="STOP")])


def test_verified_getnet_account_categories_override_model_web(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "Name the account types accepted for Pix in Brazil."
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
    ]


@pytest.mark.parametrize("message", [
    "Find the current Banco Central regulation on Pix account eligibility in Brazil.",
    "Find the latest external regulations for Pix account eligibility in Brazil.",
])
def test_explicit_current_external_pix_regulation_stays_web(monkeypatch, message):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("WEB_SEARCH_AGENT", ["u1"]),
    ]


def test_mixed_pix_and_private_firmware_get_distinct_assignments(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = ("Name the account types accepted for Pix in Brazil plus inspect the installed "
               "firmware on GET-CLASSICA-QA-84257.")
    plan = RouterAgent(WebProposal([(message, ["u1", "u2"])] )).plan(message, "synthetic")
    assert [(s.target_agent, s.request_unit_ids) for s in plan.plan_steps] == [
        ("KNOWLEDGE_AGENT", ["u1"]),
        ("CUSTOMER_SUPPORT_AGENT", ["u2"]),
    ]
    assert "firmware" not in plan.plan_steps[0].sub_query.lower()
    assert "GET-CLASSICA-QA-84257" not in plan.plan_steps[0].sub_query
    assert "Pix" not in plan.plan_steps[1].sub_query


def test_unrelated_current_weather_stays_web(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "synthetic-never-sent")
    message = "Find the current weather in Campinas."
    plan = RouterAgent(WebProposal([(message, ["u1"])] )).plan(message, "synthetic")
    assert plan.primary_agent == "WEB_SEARCH_AGENT"
