"""Explicit historical component binding for legacy HTTP regression fixtures.

This adapter exists only in tests. Production keeps the V2 runtime exclusively;
these fixtures measure legacy orchestrator behavior and retain its assertions.
"""
from app.schemas import ChatRequest


def bind_legacy_runtime(monkeypatch, main, orchestrator, *, store=None, queue=None):
    class LegacyComponentRuntime:
        def process(self, message, security_context, session_id=None):
            return orchestrator.process(ChatRequest(message=message,
                user_id=security_context.authenticated_user_id, session_id=session_id),
                security_context=security_context)

        def resume(self, ticket_id, operator):
            active_queue = queue or main.hitl_queue
            ticket = active_queue.get_ticket(ticket_id)
            return orchestrator.resume_session_from_hitl(
                session_id=ticket.session_id, user_id=ticket.user_id,
                ticket_id=ticket.ticket_id, operator_reply=ticket.operator_response or '',
                operator_id=operator.authenticated_user_id, store=store or main.state_store)

    from app.tools.registry import tool_registry
    if store is not None:
        # Registry captures its dependency at construction; updating the module
        # global alone no longer isolates persisted operations between fixtures.
        monkeypatch.setattr(tool_registry, '_store', store)
    runtime = LegacyComponentRuntime()
    monkeypatch.setattr(main, 'get_integrated_runtime', lambda: runtime)
    # Never let a V2 instance from a different integration fixture leak into
    # legacy ownership/store lookup. monkeypatch restores it after this fixture.
    monkeypatch.setattr(main, '_integrated_runtime', None)
    return runtime
