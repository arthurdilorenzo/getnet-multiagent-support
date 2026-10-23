"""Literal server-owned units and an evidence-report ledger; no SDK dependency."""
import re
from typing import Any


# Additive markers only introduce a fresh bounded read when followed by a
# property/request shape. Do not split brand names such as Plus Payments or
# Mais Vantagens, nor a bare second identifier with an unclear relation.
ADDITIVE_CONNECTOR = r'(?:plus|mais|al[eé]m de|as well as)'
_ADDITIVE_START = re.compile(
    r'\b' + ADDITIVE_CONNECTOR + r'\s+(?=(?:(?:the|a|o)\s+)?(?:'
    r'connection|connectivity|signal|firmware|status|hardware|telemetry|code|decline|'
    r'conex[aã]o|conectividade|sinal|estado|situa[cç][aã]o|telemetria|c[oó]digo|'
    r'check|inspect|verify|confirm|consult|examine|diagnose|consulte|verifique|'
    r'inspecione|confirme|examine|diagnostique|give|tell|show|me informe|me diga|me mostre)\b)', re.I)


def operational_unit_text(text: str, original: str) -> str:
    """Consume only a newly recognized literal coordination prefix.

    Raw offsets and ledger text stay untouched. This is a syntactic marker, not
    permission to discard a predicate, condition, negation or target.
    """
    literal = text.lstrip(' ;,.').strip()
    for unit in request_units(original):
        if unit['start'] and unit['text'].lstrip(' ;,.').strip() == literal:
            marker = _ADDITIVE_START.match(literal)
            if marker:
                return literal[marker.end():]
    return text


def request_units(message: str) -> list[dict[str, Any]]:
    boundaries={0,len(message)}
    for match in re.finditer(r';|(?<=[.!?])\s+(?=\S)|\b(?:e|and|mas|but|tambem|also|exceto|except|unless|se|if|desde que|provided that)\b',message,flags=re.I):
        boundaries.add(match.start())
    for match in _ADDITIVE_START.finditer(message):
        boundaries.add(match.start())
    ordered=sorted(boundaries)
    return [{'unit_id':f'u{i+1}','start':a,'end':b,'text':message[a:b]}
            for i,(a,b) in enumerate(zip(ordered,ordered[1:])) if a<b]


def validate_scope(message: str, scope: dict | None = None) -> dict[str,list[str]]:
    known={unit['unit_id'] for unit in request_units(message)}
    if scope is None or scope == {} or scope == {"assigned_unit_ids":[],"other_unit_ids":[]}:
        return {'assigned_unit_ids':list(unit['unit_id'] for unit in request_units(message)), 'other_unit_ids':[]}
    if not isinstance(scope,dict) or set(scope)-{'assigned_unit_ids','other_unit_ids'}:
        raise ValueError('Invalid delegation scope')
    result={}
    for field in ('assigned_unit_ids','other_unit_ids'):
        ids=scope.get(field,[])
        if not isinstance(ids,list) or any(not isinstance(value,str) for value in ids) or len(set(ids))!=len(ids) or not set(ids)<=known:
            raise ValueError('Unknown or duplicate request unit IDs')
        result[field]=list(ids)
    if set(result["assigned_unit_ids"]+result["other_unit_ids"]) != known:
        raise ValueError("Delegation scope omits request units")
    return result


class RequestCoverageLedger:
    """Credit only a known, assigned, evidence-backed specialist report.

    This validates attribution, not semantic truth. Specialists remain responsible
    for proving their coverage using typed observations or verified source spans.
    """
    def __init__(self,message: str):
        self._rows={unit['unit_id']:dict(unit,assigned_agents=[],fulfilled_by=[],evidence_reported=[],state='unresolved',pertinent=True,exemption_by={}) for unit in request_units(message)}

    def assign(self,agent: str,ids: list[str]) -> None:
        if len(set(ids))!=len(ids) or not set(ids)<=self._rows.keys():
            raise ValueError('Unknown or duplicate assignment IDs')
        for unit_id in ids:
            row=self._rows[unit_id]
            if agent not in row['assigned_agents']: row['assigned_agents'].append(agent)
            if not set(row['assigned_agents'])<=set(row['fulfilled_by']):
                row['state']='exempt' if set(row['assigned_agents'])<=set(row['exemption_by']) else 'delegated'
                row['pertinent']=row['state']!='exempt'

    def report(self,agent: str,ids: list[str],covered: list[str],unresolved: list[str],evidence_count: int=0,exemptions: dict[str,str] | None=None) -> bool:
        exemptions=exemptions or {}
        if not isinstance(exemptions,dict) or any(kind not in {'context','courtesy'} for kind in exemptions.values()): return False
        if exemptions and agent!='KNOWLEDGE_AGENT': return False
        if not set(exemptions)<=set(ids) or set(exemptions)&set(covered+unresolved): return False
        groups=(ids,covered,unresolved)
        if any(len(set(values))!=len(values) for values in groups): return False
        if not set(covered+unresolved)<=set(ids) or set(covered)&set(unresolved): return False
        if any(unit_id not in self._rows or agent not in self._rows[unit_id]['assigned_agents'] for unit_id in ids): return False
        for unit_id in ids:
            row=self._rows[unit_id]
            if unit_id in exemptions:
                row['exemption_by'][agent]=exemptions[unit_id]
                row['state']='exempt' if set(row['assigned_agents'])<=set(row['exemption_by']) else 'delegated'
                row['pertinent']=row['state']!='exempt'
            elif unit_id in covered and evidence_count>0:
                if agent not in row['fulfilled_by']: row['fulfilled_by'].append(agent)
                row['evidence_reported'].append({'agent':agent,'count':evidence_count})
                row['state']='fulfilled' if set(row['assigned_agents'])<=set(row['fulfilled_by']) else 'delegated'
            elif not set(row['assigned_agents'])<=set(row['fulfilled_by']): row['state']='unresolved'
        return True

    def snapshot(self) -> list[dict[str,Any]]:
        return [dict(row,exemption_by=dict(row['exemption_by']),assigned_agents=list(row['assigned_agents']),fulfilled_by=list(row['fulfilled_by']),evidence_reported=list(row['evidence_reported'])) for row in self._rows.values()]
