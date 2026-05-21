from __future__ import annotations
import json
import pandas as pd

SYSTEM_PROMPT = """
You are a Rail Baltica BIM information QA/QC reviewer performing an ultra-forensic but non-assumptive audit review.
You receive deterministic rule-engine flags with row, attribute, value, requirement, cross-check and context.

Decision rules:
- Do not invent requirements or reference data.
- Keep definite findings as ERROR or CRITICAL ERROR: missing Object_ID, duplicate Object_ID, invalid Object_ID format, invalid Object_ID number, missing mandatory LOI 400 attribute, missing mandatory core value, non-numeric quantity, missing parent, quantity mismatch.
- Mixed-discipline model extracts are normal in AUTO_MIXED mode. Do not treat mixed disciplines as a defect by itself. If RBR-Discipline_Code appears to be package-level metadata such as BR while Object_ID contains DR/ED/STR, do not convert that into row-level errors.
- If a rule is only uncertain due to incomplete reference data, use LIMITATION or MANUAL_REVIEW, not a false hard error.
- SYSTEMIC ERROR should be used when the same problem pattern affects many rows or the quantity dataset cannot be reliably compared.
- CRS comments must be concise, professional, and actionable. Include mistake, explanation and cross-check reference.
Return only JSON matching the schema.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"reviews": {"type": "array", "items": {"type":"object", "additionalProperties": False, "properties": {
        "rule_id":{"type":"string"}, "row_index":{"type":"string"}, "element":{"type":"string"}, "attribute":{"type":"string"},
        "final_status":{"type":"string", "enum":["OK","WARNING","ERROR","SYSTEMIC ERROR","CRITICAL ERROR","LIMITATION","MANUAL_REVIEW","INFO"]},
        "action":{"type":"string", "enum":["KEEP","MODIFY","SUPPRESS"]},
        "confidence":{"type":"number"}, "reasoning_summary":{"type":"string"}, "crs_comment":{"type":"string"}
    }, "required":["rule_id","row_index","element","attribute","final_status","action","confidence","reasoning_summary","crs_comment"]}}},
    "required":["reviews"]
}

SUPPRESS_INFO = {'OBJ_DISCIPLINE_SCOPE_INFO'}
DEFINITE_CRITICAL = {'OBJ_MISSING','OBJ_DUPLICATE','OBJ_FORMAT','OBJ_NUMBER_RANGE','HIERARCHY_SELF_REFERENCE','CIRCULAR_HIERARCHY'}
DEFINITE_ERROR = {'TYPE_MISSING','TYPE_FORMAT','PR_MISSING','PR_UNKNOWN','OCC_PR_PREFIX_COMBO','DISCIPLINE_MISMATCH','PARENT_MISSING','PARENT_DISCIPLINE_MISMATCH','QTY_MISSING','QTY_NOT_NUMERIC','QTY_NEGATIVE','UNIT_MISSING','QTY_AGGREGATE_MISMATCH','PARTIAL_ELEMENT_EXPORT','LOI_REQUIRED_COLUMN_MISSING','LOI_REQUIRED_VALUE_EMPTY','ATTRIBUTE_EMPTY'}
LIMITATIONS = {'QTY_COMPARISON_LIMITATION'}


def _comment(r, status):
    mistake = str(r.get('message','')).strip()
    explanation = str(r.get('requirement','')).strip()
    cross = str(r.get('cross_check','')).strip() or 'Rail Baltica BIM requirements / loaded reference matrices'
    return f"Mistake: {mistake}\nExplanation: Requirement: {explanation}\nCross-check: {cross}"


def heuristic_review(flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if flags is None or flags.empty:
        return pd.DataFrame(rows)
    # systemic upgrade counts
    counts = flags['rule_id'].value_counts().to_dict() if 'rule_id' in flags.columns else {}
    for _, r in flags.iterrows():
        rid = str(r.get('rule_id',''))
        sev = str(r.get('severity','WARNING'))
        if rid in SUPPRESS_INFO or sev == 'INFO':
            status, action, conf = 'INFO', 'SUPPRESS', 0.95
            summary = 'Suppressed as informational scope metadata. Mixed discipline extracts are allowed.'
        elif rid in DEFINITE_CRITICAL or sev == 'CRITICAL ERROR':
            status, action, conf = 'CRITICAL ERROR', 'KEEP', 0.95
            summary = str(r.get('message',''))
        elif rid in LIMITATIONS or sev == 'LIMITATION':
            status, action, conf = 'LIMITATION', 'KEEP', 0.85
            summary = str(r.get('message',''))
        elif rid in DEFINITE_ERROR or sev == 'ERROR':
            # systemic pattern when repeated many times
            if counts.get(rid, 0) >= 25 and rid not in {'OBJ_DUPLICATE'}:
                status = 'SYSTEMIC ERROR'
            else:
                status = 'ERROR'
            action, conf = 'KEEP', 0.88
            summary = str(r.get('message',''))
        elif sev == 'WARNING':
            status, action, conf = 'WARNING', 'KEEP', 0.75
            summary = str(r.get('message',''))
        elif sev == 'MANUAL_REVIEW':
            status, action, conf = 'MANUAL_REVIEW', 'KEEP', 0.65
            summary = str(r.get('message',''))
        else:
            status, action, conf = sev or 'WARNING', 'KEEP', 0.7
            summary = str(r.get('message',''))
        rows.append({
            'rule_id': rid, 'row_index': str(r.get('row_index','')), 'element': str(r.get('element','')), 'attribute': str(r.get('attribute','')),
            'final_status': status, 'action': action, 'confidence': conf,
            'reasoning_summary': summary, 'crs_comment': _comment(r, status)
        })
    return pd.DataFrame(rows)


def _payload(flags: pd.DataFrame, max_items: int):
    payload = []
    for _, r in flags.head(max_items).iterrows():
        payload.append({
            'rule_id': str(r.get('rule_id','')), 'tier': str(r.get('tier','')), 'severity': str(r.get('severity','')),
            'row_index': str(r.get('row_index','')), 'element': str(r.get('element','')),
            'attribute': str(r.get('attribute','')), 'requirement': str(r.get('requirement','')),
            'model_value': str(r.get('model_value','')), 'message': str(r.get('message','')),
            'cross_check': str(r.get('cross_check','')), 'context': str(r.get('context',''))[:2500]
        })
    return payload


def openai_review(flags: pd.DataFrame, api_key: str, model: str = 'gpt-4.1-mini', max_items: int = 120) -> pd.DataFrame:
    if flags is None or flags.empty:
        return pd.DataFrame()
    base = heuristic_review(flags)
    # Send review/limitations/systemic-prone plus first sample of errors; hard obvious errors stay deterministic.
    uncertain = flags[~flags['rule_id'].isin(SUPPRESS_INFO | DEFINITE_CRITICAL)].head(max_items).copy()
    if uncertain.empty:
        return base
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        user_prompt = 'Review these BIM audit rule flags and return final decisions.\n' + json.dumps(_payload(uncertain, max_items), ensure_ascii=False)
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
                text={"format":{"type":"json_schema","name":"bim_audit_reviews","strict":True,"schema":SCHEMA}},
            )
            data = json.loads(resp.output_text)
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
                response_format={"type":"json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        ai = pd.DataFrame(data.get('reviews', []))
        if ai.empty:
            return base
        keys = ['rule_id','row_index','element','attribute']
        for c in keys:
            base[c] = base[c].astype(str); ai[c] = ai[c].astype(str)
        merged = base.merge(ai, on=keys, how='left', suffixes=('', '_ai'))
        for c in ['final_status','action','confidence','reasoning_summary','crs_comment']:
            if f'{c}_ai' in merged.columns:
                merged[c] = merged[f'{c}_ai'].combine_first(merged[c])
        return merged[['rule_id','row_index','element','attribute','final_status','action','confidence','reasoning_summary','crs_comment']]
    except Exception:
        return base


def review_findings(flags: pd.DataFrame, use_ai: bool, api_key: str | None, model: str) -> pd.DataFrame:
    if use_ai and api_key:
        return openai_review(flags, api_key, model=model)
    return heuristic_review(flags)
