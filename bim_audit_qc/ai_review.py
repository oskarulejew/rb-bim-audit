from __future__ import annotations
import json
import pandas as pd

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a Rail Baltica BIM information QA/QC reviewer performing an ultra-forensic audit review.
You receive deterministic rule-engine flags with row, attribute, value, requirement and cross-check.

Decision rules:
- Do not invent requirements or reference data.
- Keep definite findings as ERROR or CRITICAL ERROR: missing Object_ID, duplicate Object_ID,
  invalid Object_ID format, missing mandatory LOI 400 attribute, missing core value,
  non-numeric quantity, missing parent, quantity mismatch.
- Mixed-discipline extracts are normal in AUTO_MIXED mode. Do not flag mixed disciplines.
- If a rule is uncertain due to incomplete reference data, use LIMITATION or MANUAL_REVIEW.
- SYSTEMIC ERROR when the same problem pattern affects many rows.
- CRS comments must be professional and actionable.
  Format: "Mistake: X\\nExplanation: Y\\nCross-check: Z"
Return only JSON matching the schema.
"""

RAG_SYSTEM_PROMPT = """
You are a Rail Baltica BIM information QA/QC reviewer performing an ultra-forensic audit review.
Each flag includes REFERENCE_CONTEXT — rows retrieved from the official Rail Baltica BIM standard
documents (LOI matrices, ObjectID matrices, Uniclass tables, naming conventions, kontrolltabel).

Decision rules:
- Use the REFERENCE_CONTEXT to verify or refute the flag. Always cite the source file in your reasoning.
- Do not invent requirements not present in the provided reference context.
- Keep definite findings as ERROR or CRITICAL ERROR: missing Object_ID, duplicate Object_ID,
  invalid format, missing mandatory LOI 400 attribute, missing core value, non-numeric quantity,
  missing parent, quantity mismatch.
- Mixed-discipline extracts are normal in AUTO_MIXED mode. Do not flag mixed disciplines.
- If the reference context does not cover the specific case, use LIMITATION or MANUAL_REVIEW.
- SYSTEMIC ERROR when the same problem pattern affects many rows.
- CRS comments must cite the specific source file from REFERENCE_CONTEXT.
  Format: "Mistake: X\\nExplanation: Y\\nCross-check: [cite exact source file from context]"
Return only JSON matching the schema.
"""

_SCHEMA_HINT = (
    'Return a JSON object with a single key "reviews" containing an array. '
    'Each item must have exactly these fields: '
    'rule_id (string), row_index (string), element (string), attribute (string), '
    'final_status (one of: OK / WARNING / ERROR / SYSTEMIC ERROR / CRITICAL ERROR / LIMITATION / MANUAL_REVIEW / INFO), '
    'action (one of: KEEP / MODIFY / SUPPRESS), '
    'confidence (number between 0 and 1), '
    'reasoning_summary (string), '
    'crs_comment (string formatted as "Mistake: ...\\nExplanation: ...\\nCross-check: ...").'
)

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"reviews": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "rule_id": {"type": "string"}, "row_index": {"type": "string"},
            "element": {"type": "string"}, "attribute": {"type": "string"},
            "final_status": {"type": "string", "enum": [
                "OK", "WARNING", "ERROR", "SYSTEMIC ERROR", "CRITICAL ERROR",
                "LIMITATION", "MANUAL_REVIEW", "INFO",
            ]},
            "action": {"type": "string", "enum": ["KEEP", "MODIFY", "SUPPRESS"]},
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            "crs_comment": {"type": "string"},
        },
        "required": ["rule_id", "row_index", "element", "attribute",
                     "final_status", "action", "confidence", "reasoning_summary", "crs_comment"],
    }}},
    "required": ["reviews"],
}

# ── Rule sets ─────────────────────────────────────────────────────────────────

SUPPRESS_INFO = {'OBJ_DISCIPLINE_SCOPE_INFO'}
DEFINITE_CRITICAL = {
    'OBJ_MISSING', 'OBJ_DUPLICATE', 'OBJ_FORMAT', 'OBJ_NUMBER_RANGE',
    'HIERARCHY_SELF_REFERENCE', 'CIRCULAR_HIERARCHY',
}
DEFINITE_ERROR = {
    'TYPE_MISSING', 'TYPE_FORMAT', 'PR_MISSING', 'PR_UNKNOWN',
    'OCC_PR_PREFIX_COMBO', 'DISCIPLINE_MISMATCH', 'PARENT_MISSING',
    'PARENT_DISCIPLINE_MISMATCH', 'QTY_MISSING', 'QTY_NOT_NUMERIC',
    'QTY_NEGATIVE', 'UNIT_MISSING', 'QTY_AGGREGATE_MISMATCH',
    'PARTIAL_ELEMENT_EXPORT', 'LOI_REQUIRED_COLUMN_MISSING',
    'LOI_REQUIRED_VALUE_EMPTY', 'ATTRIBUTE_EMPTY',
    'ELEMENT_COUNT_MISMATCH', 'ELEMENT_QTY_MISMATCH',
    'TOTAL_QTY_BY_UNIT_MISMATCH',
}
LIMITATIONS = {'QTY_COMPARISON_LIMITATION'}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _comment(r, status: str) -> str:
    mistake = str(r.get('message', '')).strip()
    explanation = str(r.get('requirement', '')).strip()
    cross = str(r.get('cross_check', '')).strip() or 'Rail Baltica BIM requirements / loaded reference matrices'
    return f"Mistake: {mistake}\nExplanation: Requirement: {explanation}\nCross-check: {cross}"


def _rag_query(r: pd.Series) -> str:
    element = str(r.get('element', ''))
    prefix = '-'.join(element.split('-')[:3]) if '-' in element else element
    parts = [
        str(r.get('rule_id', '')),
        str(r.get('attribute', '')),
        prefix,
        str(r.get('requirement', ''))[:120],
        str(r.get('model_value', ''))[:60],
    ]
    return ' '.join(p for p in parts if p and p.lower() not in {'nan', 'none', ''})


def _enrich_with_rag(flags: pd.DataFrame, rag) -> list[dict]:
    query_cache: dict[str, list] = {}
    payload = []
    for _, r in flags.iterrows():
        query = _rag_query(r)
        if query not in query_cache:
            query_cache[query] = rag.retrieve(query, k=6)
        ctx = query_cache[query]
        ctx_text = (
            '\n'.join(
                f"  [{i+1}] (source: {c['source']}, relevance: {c['score']:.2f})\n      {c['text']}"
                for i, c in enumerate(ctx)
            ) if ctx else '  No matching reference entry found.'
        )
        payload.append({
            'rule_id': str(r.get('rule_id', '')),
            'tier': str(r.get('tier', '')),
            'severity': str(r.get('severity', '')),
            'row_index': str(r.get('row_index', '')),
            'element': str(r.get('element', '')),
            'attribute': str(r.get('attribute', '')),
            'requirement': str(r.get('requirement', '')),
            'model_value': str(r.get('model_value', ''))[:200],
            'message': str(r.get('message', ''))[:500],
            'cross_check': str(r.get('cross_check', '')),
            'REFERENCE_CONTEXT': ctx_text,
        })
    return payload


def _plain_payload(flags: pd.DataFrame, max_items: int) -> list[dict]:
    rows = []
    for _, r in flags.head(max_items).iterrows():
        rows.append({
            'rule_id': str(r.get('rule_id', '')), 'tier': str(r.get('tier', '')),
            'severity': str(r.get('severity', '')), 'row_index': str(r.get('row_index', '')),
            'element': str(r.get('element', '')), 'attribute': str(r.get('attribute', '')),
            'requirement': str(r.get('requirement', '')),
            'model_value': str(r.get('model_value', '')),
            'message': str(r.get('message', ''))[:500],
            'cross_check': str(r.get('cross_check', '')),
        })
    return rows


def _merge_ai_into_base(base: pd.DataFrame, ai: pd.DataFrame) -> pd.DataFrame:
    if ai.empty:
        return base
    keys = ['rule_id', 'row_index', 'element', 'attribute']
    for c in keys:
        base[c] = base[c].astype(str)
        ai[c] = ai[c].astype(str)
    merged = base.merge(ai, on=keys, how='left', suffixes=('', '_ai'))
    for c in ['final_status', 'action', 'confidence', 'reasoning_summary', 'crs_comment']:
        col_ai = f'{c}_ai'
        if col_ai in merged.columns:
            merged[c] = merged[col_ai].combine_first(merged[c])
    return merged[['rule_id', 'row_index', 'element', 'attribute',
                   'final_status', 'action', 'confidence', 'reasoning_summary', 'crs_comment']]

# ── Heuristic review (no API) ─────────────────────────────────────────────────


def heuristic_review(flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if flags is None or flags.empty:
        return pd.DataFrame(rows)
    counts = flags['rule_id'].value_counts().to_dict() if 'rule_id' in flags.columns else {}
    for _, r in flags.iterrows():
        rid = str(r.get('rule_id', ''))
        sev = str(r.get('severity', 'WARNING'))
        if rid in SUPPRESS_INFO or sev == 'INFO':
            status, action, conf = 'INFO', 'SUPPRESS', 0.95
            summary = 'Suppressed: informational scope metadata. Mixed discipline extracts are allowed.'
        elif rid in DEFINITE_CRITICAL or sev == 'CRITICAL ERROR':
            status, action, conf = 'CRITICAL ERROR', 'KEEP', 0.95
            summary = str(r.get('message', ''))
        elif rid in LIMITATIONS or sev == 'LIMITATION':
            status, action, conf = 'LIMITATION', 'KEEP', 0.85
            summary = str(r.get('message', ''))
        elif rid in DEFINITE_ERROR or sev == 'ERROR':
            status = 'SYSTEMIC ERROR' if counts.get(rid, 0) >= 25 and rid not in {'OBJ_DUPLICATE'} else 'ERROR'
            action, conf = 'KEEP', 0.88
            summary = str(r.get('message', ''))
        elif sev == 'WARNING':
            status, action, conf = 'WARNING', 'KEEP', 0.75
            summary = str(r.get('message', ''))
        elif sev == 'MANUAL_REVIEW':
            status, action, conf = 'MANUAL_REVIEW', 'KEEP', 0.65
            summary = str(r.get('message', ''))
        else:
            status, action, conf = sev or 'WARNING', 'KEEP', 0.7
            summary = str(r.get('message', ''))
        rows.append({
            'rule_id': rid, 'row_index': str(r.get('row_index', '')),
            'element': str(r.get('element', '')), 'attribute': str(r.get('attribute', '')),
            'final_status': status, 'action': action, 'confidence': conf,
            'reasoning_summary': summary, 'crs_comment': _comment(r, status),
        })
    return pd.DataFrame(rows)

# ── OpenAI review ─────────────────────────────────────────────────────────────


def openai_review(flags: pd.DataFrame, api_key: str, model: str = 'gpt-4.1-mini',
                  max_items: int = 120, rag=None) -> pd.DataFrame:
    if flags is None or flags.empty:
        return pd.DataFrame()
    base = heuristic_review(flags)
    uncertain = flags[~flags['rule_id'].isin(SUPPRESS_INFO | DEFINITE_CRITICAL)].head(max_items).copy()
    if uncertain.empty:
        return base
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        use_rag = rag is not None and rag.ready
        if use_rag:
            payload = _enrich_with_rag(uncertain, rag)
            user_prompt = (
                'Review the BIM audit flags. Each includes REFERENCE_CONTEXT from official '
                'Rail Baltica BIM documents. Cite the source in every crs_comment.\n'
                + json.dumps(payload, ensure_ascii=False)
            )
            system = RAG_SYSTEM_PROMPT
        else:
            user_prompt = ('Review these BIM audit flags and return final decisions.\n'
                           + json.dumps(_plain_payload(uncertain, max_items), ensure_ascii=False))
            system = SYSTEM_PROMPT
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system},
                       {"role": "user", "content": user_prompt}],
                text={"format": {"type": "json_schema", "name": "bim_audit_reviews",
                                 "strict": True, "schema": SCHEMA}},
            )
            data = json.loads(resp.output_text)
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                           {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        return _merge_ai_into_base(base, pd.DataFrame(data.get('reviews', [])))
    except Exception:
        return base

# ── Google Gemini review (free tier) ─────────────────────────────────────────


def gemini_review(flags: pd.DataFrame, api_key: str, model: str = 'gemini-2.0-flash',
                  max_items: int = 120, rag=None) -> pd.DataFrame:
    if flags is None or flags.empty:
        return pd.DataFrame()
    base = heuristic_review(flags)
    uncertain = flags[~flags['rule_id'].isin(SUPPRESS_INFO | DEFINITE_CRITICAL)].head(max_items).copy()
    if uncertain.empty:
        return base
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        use_rag = rag is not None and rag.ready
        if use_rag:
            payload = _enrich_with_rag(uncertain, rag)
            user_prompt = (
                'Review the BIM audit flags. Each includes REFERENCE_CONTEXT from official '
                'Rail Baltica BIM documents. Cite the source in every crs_comment.\n'
                + _SCHEMA_HINT + '\n'
                + json.dumps(payload, ensure_ascii=False)
            )
            system = RAG_SYSTEM_PROMPT
        else:
            user_prompt = (
                'Review these BIM audit flags and return final decisions.\n'
                + _SCHEMA_HINT + '\n'
                + json.dumps(_plain_payload(uncertain, max_items), ensure_ascii=False)
            )
            system = SYSTEM_PROMPT
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(response_mime_type='application/json'),
        )
        response = gemini_model.generate_content(user_prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        data = json.loads(raw)
        return _merge_ai_into_base(base, pd.DataFrame(data.get('reviews', [])))
    except Exception:
        return base

# ── Dispatcher ────────────────────────────────────────────────────────────────


def review_findings(flags: pd.DataFrame, use_ai: bool, api_key: str | None,
                    model: str, rag=None, provider: str = 'openai') -> pd.DataFrame:
    if use_ai and api_key:
        if provider.lower() in {'google', 'gemini', 'google gemini'}:
            return gemini_review(flags, api_key, model=model, rag=rag)
        return openai_review(flags, api_key, model=model, rag=rag)
    return heuristic_review(flags)
