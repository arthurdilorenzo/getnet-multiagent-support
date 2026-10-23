"""Server-owned terminal field projection at specialist and final public egress.

Internal observations can remain rich. Model prose never supplies field authority.
Unknown independent text cannot disable a recognized target/field contract.
"""
import re
from app.schemas import ChatResponse, SpecialistOutput
from app.security.request_constraints import extract_constraints, _normalize
from app.security.request_units import request_units

FIELDS = {
    'battery': (r'\b(?:battery|bateria)\b', 'Bateria'),
    'connection_type': (r'\b(?:connection(?: type)?|connectivity|conexao|conectividade)\b', 'Tipo de conexão'),
    'signal_strength': (r'\b(?:signal(?: strength)?|sinal)\b', 'Força do sinal'),
    'firmware_version': (r'\b(?:firmware|versao|version)\b', 'Firmware'),
    'status': (r'\b(?:status|estado|situacao)\b', 'Status'),
    'model': (r'\b(?:modelo|model)\b', 'Modelo'),
}


def terminal_field_contract(original_message: str) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for unit in request_units(original_message):
        ids = extract_constraints(unit['text']).terminal_ids
        if len(ids) != 1:
            continue
        # Read the original text, not generated subqueries/rationales. A field
        # mentioned solely in a negative restriction grants no disclosure.
        norm = _normalize(unit['text'])
        norm = re.split(r'\b(?:sem incluir|sem mostrar|without including|do not include|nao inclua|nao mostre)\b', norm)[0]
        fields = {name for name, (pattern, _) in FIELDS.items() if re.search(pattern, norm)}
        if fields:
            pairs.setdefault(ids[0], set()).update(fields)
    return pairs


def project_specialist(output: SpecialistOutput, original_message: str,
                       authenticated_user_id: str | None = None) -> SpecialistOutput:
    """Project only diagnostic results, preserving independent verified tools."""
    from app.agents.customer_support_agent import CustomerSupportAgent
    from app.security.fulfillment import evaluate_operational_unit
    contract = terminal_field_contract(original_message)
    if not contract or output.agent_name != 'CUSTOMER_SUPPORT_AGENT':
        return output
    output = output.model_copy(deep=True)
    broad_targets = {target for unit in request_units(original_message)
        if re.search(r'\b(?:diagnostique|diagnosticar|diagnose|diagnosis|diagnostico|troubleshoot)\b', _normalize(unit['text']))
        for target in extract_constraints(unit['text']).terminal_ids if target not in contract}
    safe_results, rich_for_coverage, other_results = [], [], []
    rendered, seen = [], set()
    profile_missing = False
    for result in output.tool_results:
        if result.tool_name != 'diagnose_terminal_issue':
            if result.tool_name == 'get_customer_profile':
                data = result.data or {}
                profile_requested = bool(re.search(r'\b(?:full profile|my profile|customer profile|perfil completo|meu perfil|perfil cadastral|dados cadastrais)\b', _normalize(original_message)))
                if not profile_requested or (authenticated_user_id is not None and data.get('user_id') != authenticated_user_id):
                    profile_missing = profile_requested
                    continue
            other_results.append(result)
            continue
        if not result.success:
            safe_results.append(result.model_copy(update={'data':None, 'error_message':'Consulta operacional recusada ou indisponível.'}))
            continue
        data = result.data or {}
        terminal = data.get('terminal') or {}
        target = terminal.get('terminal_id') or data.get('requested_terminal_id')
        if target in broad_targets and (authenticated_user_id is None or data.get('user_id') == authenticated_user_id):
            other_results.append(result)
            continue
        if target not in contract:
            continue
        owned = authenticated_user_id is None or data.get('user_id') == authenticated_user_id
        # Idempotent projection retains tenant proof internally in the dedicated
        # structured result. It is stripped from final public evidence below.
        valid = owned and data.get('found') is True and terminal.get('terminal_id') == target
        clean = {'found': bool(valid), 'requested_terminal_id':target}
        if valid:
            selected = {field:terminal[field] for field in contract[target]
                        if isinstance(terminal.get(field), str) and terminal[field].strip()}
            clean = {'found':True, 'requested_terminal_id':target, 'user_id': data.get('user_id'),
                     'terminal':{'terminal_id':target, **selected}}
        else:
            selected = {}
        safe_results.append(result.model_copy(update={'data':clean}))
        rich_for_coverage.append(result)
        if valid and selected:
            values = ' | '.join(f"**{FIELDS[field][1]}:** {selected.get(field, 'não informado')}" for field in FIELDS if field in contract[target])
            rendered.append(f'- `{target}` — {values}')
        else:
            missing = ' | '.join(f'**{FIELDS[field][1]}:** não informado' for field in FIELDS if field in contract[target])
            rendered.append(f'- `{target}` — dado solicitado não disponível na consulta. {missing}')
        seen.add(target)
    # A separate delegation must not report the other delegation as missing.
    assigned = set(output.request_unit_ids)
    active_targets = set(contract) if not assigned else {target for unit in request_units(original_message)
        if unit['unit_id'] in assigned for target in extract_constraints(unit['text']).terminal_ids}
    for target in contract:
        if target not in seen and target in active_targets:
            rendered.append(f'- `{target}` — dado solicitado não disponível na consulta.')
    renderer = CustomerSupportAgent()
    other_text = renderer._render_verified_results(original_message, authenticated_user_id or '', other_results) if other_results else ''
    # Typed supervisor predicates consume the original rich observation. These
    # results are internal; ChatResponse has no tool_results field.
    public_results = [*other_results, *safe_results]
    output.structured_data = {r.tool_name:r.data for r in other_results if r.success and r.data}
    if any(result.tool_name == 'diagnose_terminal_issue' for result in output.tool_results):
        output.structured_data['diagnose_terminal_issue'] = [r.data for r in safe_results if r.data]
    if profile_missing:
        other_text = 'A consulta de perfil solicitada não pôde ser concluída.'
    output.response = '\n\n'.join(part for part in (other_text, '**Campos solicitados por terminal**\n\n'+'\n'.join(rendered) if rendered else '') if part)
    output.evidence = renderer._result_evidence(public_results)
    if not broad_targets and not re.search(r'\b(?:hardware|defeito|leitor|broken|replacement|substituicao)\b', _normalize(original_message)):
        output.requires_escalation = False
        output.escalation_reason = None
    units = [unit for unit in request_units(original_message) if not assigned or unit['unit_id'] in assigned]
    output.request_unit_ids = [unit['unit_id'] for unit in units]
    output.covered_unit_ids = [unit['unit_id'] for unit in units if evaluate_operational_unit(unit['text'],
        [*other_results,*rich_for_coverage], original=original_message, authenticated_user_id=authenticated_user_id).fulfilled]
    output.unresolved_unit_ids = [uid for uid in output.request_unit_ids if uid not in output.covered_unit_ids]
    output.abstained = bool(output.unresolved_unit_ids)
    observed_own_target = any((result.data or {}).get('found') is True for result in safe_results)
    output.answer_status = 'partial' if output.unresolved_unit_ids and (output.covered_unit_ids or observed_own_target) else 'abstained' if output.unresolved_unit_ids else 'complete'
    return output


def _public_trace_details(step, requested_targets: set[str]) -> dict:
    """Typed, action-specific server telemetry; never arbitrary model strings."""
    details = step.details or {}
    clean = {'public_projection':'requested_terminal_fields'}
    agents = {'CUSTOMER_SUPPORT_AGENT', 'KNOWLEDGE_AGENT', 'WEB_SEARCH_AGENT', 'HUMAN_ESCALATION_AGENT'}
    def booleans(*keys):
        for key in keys:
            if type(details.get(key)) is bool:
                clean[key] = details[key]
    def counters(*keys):
        for key in keys:
            if type(details.get(key)) is int and details[key] >= 0:
                clean[key] = details[key]
    def enum(key, values):
        if isinstance(details.get(key), str) and details[key] in values:
            clean[key] = details[key]
    def agent_list(key):
        value = details.get(key)
        if isinstance(value, list) and all(isinstance(item, str) and item in agents for item in value):
            clean[key] = list(value)
    if step.agent == 'SUPERVISOR_ORCHESTRATOR' and step.action == 'evaluate_conditional_dependency':
        booleans('predicate_authorized')
        enum('terminal_id', requested_targets)
        enum('reason', {'exact_typed_observation','predicate_false_or_unverified'})
    elif step.agent == 'SUPERVISOR_ORCHESTRATOR' and step.action == 'evaluate_stopping_condition':
        counters('iterations_used','pending_steps')
        enum('stopping_reason', {'iteration_limit','all_planned_and_replanned_steps_completed'})
        agent_list('agents_executed')
    elif step.agent == 'CUSTOMER_SUPPORT_TOOL' and step.action in {
            'call:diagnose_terminal_issue', 'call:get_customer_profile',
            'call:get_sales_settlement', 'call:execute_terminal_remote_action'}:
        booleans('success','idempotent_replay')
        counters('turn')
    elif step.agent == 'ROUTER_AGENT' and step.action == 'plan_and_route':
        enum('primary_agent', agents)
        agent_list('planned_steps')
        booleans('is_multi_step','requires_user_context')
    elif step.agent == 'SYSTEM_EGRESS' and step.action == 'sanitize_and_persist_response':
        booleans('abstained','escalated')
        counters('response_char_count')
        enum('answer_status', {'complete','partial','abstained',''})
        enum('semantic_mode', {'native_semantic','limited_offline',''})
    return clean


def project_public_response(response: ChatResponse, original_message: str,
                            outputs: list[SpecialistOutput], authenticated_user_id: str,
                            *, trusted_segments: list[str] | None = None) -> ChatResponse:
    """Call after aggregation, BEFORE persisting response.response. Idempotent.

    Every free-form model trace/intent is excluded under a field contract because
    those strings may repeat rich internal observations. Independent specialists'
    verified text/evidence/citations/proposals remain in their original order.
    """
    if not terminal_field_contract(original_message):
        return response
    safe = response.model_copy(deep=True)
    projected = [project_specialist(out, original_message, authenticated_user_id) for out in outputs]
    safe.response = '\n\n---\n\n'.join([*(trusted_segments or []), *(out.response for out in projected if out.response)])
    safe.intent = 'Consulta com divulgação limitada aos campos solicitados por terminal.'
    safe.evidence = [item for out in projected for item in out.evidence]
    safe.citations = [citation for out in projected if out.agent_name != 'CUSTOMER_SUPPORT_AGENT' for citation in out.citations]
    safe.sources = list(dict.fromkeys(source for out in projected if out.agent_name != 'CUSTOMER_SUPPORT_AGENT' for source in out.sources))
    # Proposals are reconstructed from the server result, not the aggregate.
    safe.action_proposals = [r.data for out in projected for r in out.tool_results
        if r.success and r.tool_name == 'execute_terminal_remote_action' and r.data and r.data.get('proposal_id')]
    for step in safe.trace:
        # Keep typed event identity/status/timing; arbitrary details and rationales
        # can contain any unrequested field or a copied internal tool snapshot.
        step.details = _public_trace_details(step, set(extract_constraints(original_message).terminal_ids))
    from app.security.fulfillment import canonical_terminal_read_text, _useful
    affected_field_ids, proven_field_ids = set(), set()
    for out in projected:
        if out.agent_name != 'CUSTOMER_SUPPORT_AGENT':
            continue
        for unit in request_units(original_message):
            if unit['unit_id'] not in out.request_unit_ids:
                continue
            canonical = canonical_terminal_read_text(unit['text'].lstrip(' ;,.'), original_message)
            fields_by_target = terminal_field_contract(canonical)
            if not fields_by_target:
                continue
            affected_field_ids.add(unit['unit_id'])
            proven = all(any(
                result.tool_name == 'diagnose_terminal_issue' and result.success and result.outcome_state == 'completed'
                and (result.data or {}).get('found') is True
                and (result.data or {}).get('user_id') == authenticated_user_id
                and ((result.data or {}).get('terminal') or {}).get('terminal_id') == target
                and all(_useful(((result.data or {}).get('terminal') or {}).get(field), 'terminal.'+field)
                        for field in fields)
                for result in out.tool_results)
                for target, fields in fields_by_target.items())
            if proven:
                proven_field_ids.add(unit['unit_id'])
    missing_field_ids = affected_field_ids - proven_field_ids
    original_units = {unit['unit_id']: unit for unit in request_units(original_message)}
    for row in safe.request_coverage:
        unit = original_units.get(row.get('unit_id'))
        if unit:
            row['text'] = unit['text']
        if row.get('unit_id') in missing_field_ids and row.get('state') != 'exempt':
            row['state'] = 'unresolved'
        # Preserve server-owned unit identity/state/text; omit model annotations.
        allowed = {'unit_id', 'text', 'state', 'start', 'end', 'assigned_agents'}
        for key in list(row):
            if key not in allowed:
                row.pop(key)
    if missing_field_ids:
        safe.abstained = True
        independent_verified = any(out.evidence for out in projected if out.agent_name != 'CUSTOMER_SUPPORT_AGENT')
        safe.answer_status = 'partial' if (proven_field_ids or independent_verified or any(row.get('state') == 'fulfilled' for row in safe.request_coverage)) else 'abstained'
    return safe
