from __future__ import annotations
import os, uuid, shutil, warnings, re, hashlib
from pathlib import Path
import pandas as pd
import streamlit as st

from bim_audit_qc.detector import detect_files
from bim_audit_qc.references import ReferenceLibrary
from bim_audit_qc.audit_engine import run_audit
from bim_audit_qc.ai_review import review_findings
from bim_audit_qc.report import merge_final, write_report, final_decision
from bim_audit_qc.user_rules import UserRules, create_templates
from bim_audit_qc.rag import ReferenceRAG

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


# ---------------------------------------------------------------------------
# Access gate
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = '') -> str:
    try:
        return st.secrets.get(key, default) or default
    except Exception:
        return os.environ.get(key, default)


def _check_access(email: str, code: str) -> bool:
    raw_codes = _get_secret('ACCESS_CODES', '')
    if not raw_codes.strip():
        return False  # no codes configured — deny everyone until secrets are set
    allowed_codes = {c.strip() for c in raw_codes.split(',') if c.strip()}
    raw_emails = _get_secret('ALLOWED_EMAILS', '')
    allowed_emails = {e.strip().lower() for e in raw_emails.split(',') if e.strip()}
    code_ok = code.strip() in allowed_codes  # case-sensitive
    email_ok = (not allowed_emails) or (email.strip().lower() in allowed_emails)
    return code_ok and email_ok


def _access_gate():
    if st.session_state.get('authenticated'):
        return
    st.set_page_config(page_title='Rail Baltica BIM Audit — Login', layout='centered')
    st.title('Rail Baltica BIM Forensic Audit')
    st.markdown('Enter your email address and the access code provided by your administrator.')
    with st.form('login_form'):
        email = st.text_input('Email address')
        code = st.text_input('Access code', type='password')
        submitted = st.form_submit_button('Sign in', type='primary')
    if submitted:
        if _check_access(email, code):
            st.session_state['authenticated'] = True
            st.session_state['user_email'] = email.strip().lower()
            st.rerun()
        else:
            st.error('Incorrect access code. Contact your administrator.')
    st.stop()


_access_gate()

BASE = Path(__file__).parent
REF_DIR = BASE / 'references'
TEMP_DIR = BASE / 'temp_uploads'
REPORT_DIR = BASE / 'reports'
RULES_DIR = BASE / 'project_rules'
TEMP_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
RULES_DIR.mkdir(exist_ok=True)
create_templates(RULES_DIR)
RAG_CACHE_DIR = BASE / '.rag_cache'


@st.cache_resource(show_spinner=False)
def _load_rag(ref_dirs_key: tuple, api_key: str, provider: str) -> ReferenceRAG:
    rag = ReferenceRAG()
    if api_key:
        rag.build(list(ref_dirs_key), api_key, provider=provider, cache_dir=RAG_CACHE_DIR)
    return rag


def safe_display(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit/Arrow can be strict with mixed numeric/text columns. Display as strings."""
    if df is None:
        return pd.DataFrame()
    try:
        return df.astype(str)
    except Exception:
        return pd.DataFrame(df)


def extract_project_code_from_names(paths: list[Path | str]) -> str:
    """Derive project/package code for report naming from uploaded file names.

    Examples:
    RBDTD-EE-DS1-DPS2_TRE_BR1120-ZZ_0004_BQ_BR-TS_DTD_000003_BoQ (1).xlsx
    -> BR1120

    The detection intentionally ignores DS1/DS2/DS3, DPS2, DTD and purely numeric revision codes.
    """
    names = ' '.join(Path(str(p)).stem for p in paths if p)
    if not names:
        return 'UNKNOWN_PROJECT'

    # Strong RB naming convention pattern: _TRE_BR1120-ZZ or _XXX_CU140001-ZZ
    strong_patterns = [
        r'[_-](?:TRE|TLL|PRN|RB|RBE|DPS\d*)[_-]([A-Z]{2}\d{4,6})(?=[-_])',
        r'[_-]([A-Z]{2}\d{4,6})-ZZ(?:_|-)',
        r'[_-]([A-Z]{2}\d{4,6})(?=[-_][A-Z0-9]{2,})',
    ]
    for pat in strong_patterns:
        m = re.search(pat, names, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fallback: any 2 letters + 4-6 digits, but exclude known non-project tokens.
    blacklist = {'DS1','DS2','DS3','DPS1','DPS2','DPS3','DTD'}
    candidates = re.findall(r'\b([A-Z]{2}\d{4,6})\b', names.upper())
    candidates = [c for c in candidates if c not in blacklist]
    if candidates:
        return candidates[0]
    return 'UNKNOWN_PROJECT'


def unique_report_path(report_dir: Path, project_code: str) -> Path:
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', project_code or 'UNKNOWN_PROJECT').strip('_') or 'UNKNOWN_PROJECT'
    base = report_dir / f'Audit_report_{safe}.xlsx'
    if not base.exists():
        return base
    i = 2
    while True:
        p = report_dir / f'Audit_report_{safe}_v{i}.xlsx'
        if not p.exists():
            return p
        i += 1

st.set_page_config(page_title='Rail Baltica BIM Forensic AI QA/QC v9', layout='wide')
st.title('Rail Baltica BIM Forensic AI-assisted QA/QC — v9')
st.caption('V9 forensic logic: structural validation + element-level qty checks + column mapping docs + Dataset_Sanity sheet + controlled feedback learning')

_google_key  = _get_secret('GOOGLE_API_KEY',  '') or os.environ.get('GOOGLE_API_KEY',  '')
_openai_key  = _get_secret('OPENAI_API_KEY',  '') or os.environ.get('OPENAI_API_KEY',  '')

with st.sidebar:
    st.header('Audit settings')
    if st.session_state.get('user_email'):
        st.caption(f'Signed in as {st.session_state["user_email"]}')
        if st.button('Sign out', use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.markdown('---')
    discipline = st.selectbox(
        'Scope handling',
        ['AUTO_MIXED','STR','AR','DR','RO','BR','GE','EL','ME','TP','IA','IW','ED','FI','CO','LA','UD'],
        index=0,
        help='AUTO_MIXED is recommended when one model export contains several disciplines. The app reads the discipline per row from RBR-Object_ID.'
    )
    st.markdown('---')
    st.subheader('AI reasoning')
    _has_any_key = bool(_google_key or _openai_key)
    use_ai = st.checkbox('Use AI reasoning review', value=_has_any_key)

    if use_ai:
        # Build provider list — pre-configured ones come first
        _provider_opts = []
        if _google_key:
            _provider_opts.append('Google Gemini (free)')
        if _openai_key:
            _provider_opts.append('OpenAI')
        if not _provider_opts:
            _provider_opts = ['Google Gemini (free)', 'OpenAI']
        ai_provider_label = st.selectbox('Provider', _provider_opts)

        if 'Gemini' in ai_provider_label:
            ai_provider = 'google'
            model = 'gemini-2.0-flash'
            if _google_key:
                api_key = _google_key
                st.caption('✅ Google Gemini free tier — pre-configured.')
            else:
                api_key = st.text_input('Google API key', type='password',
                                        help='Free at aistudio.google.com/app/apikey')
                st.caption('Free tier: 1 M tokens / day · 15 req / min · no credit card needed.')
        else:
            ai_provider = 'openai'
            if _openai_key:
                api_key = _openai_key
                model = st.selectbox('Model', ['gpt-4.1-mini', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4o'], index=0)
                st.caption('✅ OpenAI — pre-configured.')
            else:
                api_key = st.text_input('OpenAI API key', type='password')
                model = st.selectbox('Model', ['gpt-4.1-mini', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4o'], index=0)
    else:
        api_key, model, ai_provider = '', 'gemini-2.0-flash', 'google'
        st.caption('Running local heuristic review — no API needed.')
    st.markdown('---')
    st.subheader('Controlled learning')
    use_user_rules = st.checkbox('Apply user-confirmed project rules', value=True)
    st.caption('Rules are read from project_rules/ on every audit. The app does not rewrite code automatically.')

st.info('Upload files with their real names. The app reloads every file in references/ and project_rules/ on each audit run. QEX/QTO templates can have title rows; the app scans for the real header row.')
col1, col2 = st.columns(2)
with col1:
    model_uploads = st.file_uploader('Model extract files', type=['xlsx','xlsm','xls','csv','txt'], accept_multiple_files=True)
with col2:
    qty_uploads = st.file_uploader('QEX / QTO / quantity files', type=['xlsx','xlsm','xls','csv','txt'], accept_multiple_files=True)

extra_ref_uploads = st.file_uploader(
    'Optional additional reference files / project-specific control tables',
    type=['xlsx','xlsm','xls','csv','txt'],
    accept_multiple_files=True,
    help='Use this for project-specific PayItem/PBS/product-code tables, extra Uniclass matrices, naming rules, EIR extracts or other control tables. They will be loaded in addition to the built-in references for this run.'
)

with st.expander('Permanent references and controlled learning rules', expanded=False):
    st.markdown(f'''
**Permanent references folder:** `{REF_DIR}`  
Add new `.xlsx`, `.xlsm`, `.csv` or `.txt` control tables into this folder at any time. They will be scanned again on the next audit run.

**Controlled learning folder:** `{RULES_DIR}`  
Add confirmed override / false-positive rules here. Templates are created automatically. The app applies them only if enabled in the sidebar.
''')
    if st.button('Open project folder in Finder instructions'):
        st.code('cd ~/Downloads/RB_BIM_Audit_AI_WebApp && open .', language='zsh')


with st.expander('What this report will contain', expanded=False):
    st.markdown('''
The downloaded workbook contains five sheets, aligned with the V9 forensic audit prompt:
1. **Model_Original** — raw model dataset as read after header detection.
2. **Model_With_Errors** — same structure, with red/yellow highlighted erroneous source cells and comments.
3. **Errors_Filterable** — filterable issue table with Source_Row, ObjectID, Attribute, Requirement, Severity, Explanation and Cross-check.
4. **Quantity_Control** — Model/QEX/QTO aggregation comparison by Pr_Code + Type_number + Unit.
5. **Dataset_Sanity** — file inventory (detected roles), column mapping documentation (which RBR attribute was found under which column), reference loading stats, and dataset limitations.
''')

run = st.button('Run forensic audit', type='primary')

if run:
    if not model_uploads and not qty_uploads:
        st.error('Please upload at least one model extract or quantity file.')
        st.stop()

    job = uuid.uuid4().hex[:8]
    job_dir = TEMP_DIR / job
    model_dir = job_dir / 'model_extracts'
    qty_dir = job_dir / 'quantity_files'
    extra_ref_dir = job_dir / 'additional_references'
    model_dir.mkdir(parents=True, exist_ok=True)
    qty_dir.mkdir(parents=True, exist_ok=True)
    extra_ref_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in model_uploads or []:
        p = model_dir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)
    for f in qty_uploads or []:
        p = qty_dir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(p)
    for f in extra_ref_uploads or []:
        p = extra_ref_dir / f.name
        p.write_bytes(f.getbuffer())

    all_uploaded_for_naming = [Path(p) for p in paths]
    project_code = extract_project_code_from_names(all_uploaded_for_naming)

    with st.spinner('Loading Rail Baltica references and detecting uploaded files...'):
        ref_dirs = [REF_DIR]
        if extra_ref_uploads:
            ref_dirs.append(extra_ref_dir)
        refs = ReferenceLibrary(ref_dirs)
        user_rules = UserRules(RULES_DIR) if use_user_rules else None
        selected, detected = detect_files(paths)

    rag = None
    if use_ai and api_key:
        with st.spinner('Loading RAG index from reference documents (cached after first run)...'):
            rag = _load_rag(tuple(str(d) for d in ref_dirs), api_key.strip(), ai_provider)
        if rag.ready:
            s = rag.stats()
            st.success(f"RAG index ready — {s['chunks_indexed']:,} chunks from {s['reference_files_indexed']} reference files.")
        if len(model_uploads or []) == 1:
            selected['model'] = str(model_dir / model_uploads[0].name)
        if qty_uploads:
            selected['quantities'] = [str(qty_dir / f.name) for f in qty_uploads]
            # Respect QEX/QTO filenames even when content score is weak.
            for p in selected['quantities']:
                lname = Path(p).name.lower()
                if 'qex' in lname and not selected.get('qex'):
                    selected['qex'] = p
                if ('qto' in lname or 'boq' in lname) and not selected.get('qto'):
                    selected['qto'] = p

    st.subheader('Reference loading summary')
    st.json(refs.stats())
    st.caption(f'Extra reference files added in this run: {len(extra_ref_uploads or [])}')

    st.subheader('Controlled learning summary')
    if use_user_rules and user_rules is not None:
        st.json(user_rules.stats())
    else:
        st.caption('User-confirmed project rules are disabled for this run.')

    st.subheader('Project/report naming')
    st.write(f'Detected project code: **{project_code}**')
    st.caption(f'Report will be named like: Audit_report_{project_code}.xlsx')

    st.subheader('Detected / ignored files')
    st.dataframe(safe_display(detected), width='stretch')
    c1, c2, c3 = st.columns(3)
    c1.write('**Selected model:**')
    c1.caption(selected.get('model') or 'NOT DETECTED')
    c2.write('**Selected QEX:**')
    c2.caption(selected.get('qex') or 'NOT DETECTED')
    c3.write('**Selected QTO:**')
    c3.caption(selected.get('qto') or 'NOT DETECTED')

    if not selected.get('model'):
        st.error('No model extract could be detected. Upload the model export under Model extract files, preferably Excel/CSV with RBR columns.')
        st.stop()

    with st.spinner('Running deterministic BIM rule engine...'):
        flags, limitations, qty_control, model_df, model_cols, model_meta, col_mappings = run_audit(
            selected['model'], refs, discipline=discipline,
            qex_path=selected.get('qex'), qto_path=selected.get('qto'), quantity_paths=selected.get('quantities')
        )

    st.subheader('Model parsing summary')
    st.json({'model_file': Path(selected['model']).name, 'detected_sheet': model_meta.get('sheet'), 'detected_header_row': model_meta.get('header_row'), 'rows': len(model_df), 'columns': len(model_df.columns)})
    if col_mappings is not None and not col_mappings.empty:
        missing_cols = col_mappings[col_mappings['Status'] == 'MISSING']['Standard_Key'].tolist()
        if missing_cols:
            st.warning(f'Core columns not detected: {", ".join(missing_cols)}. Checks requiring these attributes were skipped.')
        with st.expander('Column mapping (V9 attribute resolution)', expanded=False):
            st.dataframe(safe_display(col_mappings), use_container_width=True)

    st.subheader('Raw rule flags')
    st.write(f'{len(flags)} raw flags before reasoning review')
    st.dataframe(safe_display(flags.head(300)), width='stretch')

    with st.spinner('Running AI reasoning review with RAG reference context...' if (use_ai and rag and rag.ready) else 'Running local reasoning review...'):
        reviews = review_findings(flags, use_ai=use_ai, api_key=api_key.strip() or None, model=model, rag=rag, provider=ai_provider)
        final = merge_final(flags, reviews)
        if use_user_rules and user_rules is not None:
            final = user_rules.apply(final)

    final_kept = final[final['action']!='SUPPRESS'] if final is not None and not final.empty else pd.DataFrame()
    suppressed_count = 0 if final is None or final.empty else int((final['action']=='SUPPRESS').sum())
    decision = final_decision(final_kept, limitations, qty_control)

    st.subheader('Final findings after reasoning review')
    st.write(f'{len(final_kept)} final findings kept; {suppressed_count} informational/false-positive items suppressed.')
    st.info(f'Final decision: **{decision}**')
    st.dataframe(safe_display(final_kept.head(300)), width='stretch')

    if not final_kept.empty:
        learning_export = final_kept[['rule_id','element','attribute','model_value','final_status','crs_comment']].copy()
        learning_export.insert(0, 'enabled', 'FALSE')
        learning_export.insert(1, 'rule_name', 'Write a clear rule name here')
        learning_export['objectid_pattern'] = learning_export['element']
        learning_export['prefix_pattern'] = learning_export['element'].astype(str).apply(lambda x: '-'.join(x.split('-')[:3]) if '-' in x else '')
        learning_export['model_value_pattern'] = learning_export['model_value']
        learning_export['action'] = 'SUPPRESS or MODIFY or KEEP'
        learning_export['reasoning_summary'] = 'Explain why this correction is valid'
        learning_export['cross_check'] = 'Add exact reference / project decision / BEP note'
        learning_export = learning_export[['enabled','rule_name','rule_id','attribute','objectid_pattern','prefix_pattern','model_value_pattern','action','final_status','reasoning_summary','crs_comment','cross_check']]
        st.download_button(
            'Download correction-rule draft from current findings',
            data=learning_export.to_csv(index=False).encode('utf-8-sig'),
            file_name=f'correction_rules_draft_{project_code}.csv',
            mime='text/csv'
        )

    if qty_control is not None and not qty_control.empty:
        st.subheader('Quantity Control preview')
        st.dataframe(safe_display(qty_control.head(300)), width='stretch')

    out_path = unique_report_path(REPORT_DIR, project_code)
    write_report(out_path, flags, reviews, final, limitations, detected, refs.stats(), qty_control, model_df=model_df, model_cols=model_cols, model_meta=model_meta, col_mappings=col_mappings)
    with open(out_path, 'rb') as fh:
        st.download_button('Download 5-sheet forensic Excel audit report', data=fh.read(), file_name=out_path.name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.success(f'Report generated: {out_path.name}')

    try:
        shutil.rmtree(job_dir)
    except Exception:
        pass
