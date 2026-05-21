from __future__ import annotations
from pathlib import Path
import pandas as pd
from .io_utils import read_table_with_meta, TABLE_EXTS, find_col

MODEL_COL_CANDIDATES = {
    'object_id': ['RBR-Object_ID','Object_ID','ObjectID','Object Id','RBR_Object_ID','Element ID','Asset ID'],
    'type_number': ['RBR-Type_number','Type_number','Type-number','TypeNumber','Type Nr','Type No','RBR_Type_number'],
    'pr_code': ['RBR-Pr_Code','Pr_Code','Pr-Code','Product Code','PayItem','Uniclass','Classification','RBR_Pr_Code'],
    'occ': ['RBR-OCC','OCC','Occupation','Object classification','RBR_OCC'],
    'qty': ['RBR-Quantity','Quantity','Qty','Amount','Total','RBR_Qty'],
    'unit': ['RBR-Units','RBR-Unit','Unit','Units','UoM','Measurement unit','RBR_Units'],
}
QTY_COL_CANDIDATES = {
    'object_id': ['RBR-Object_ID','Object_ID','ObjectID','Object Id','Element ID','Asset ID'],
    'occ': ['RBR-OCC','OCC','Occupation','Object classification'],
    'pr_code': ['RBR-Pr_Code','Pr_Code','Pr-Code','Product Code','PayItem','Uniclass','Classification','Item code'],
    'type_number': ['RBR-Type_number','Type_number','Type-number','TypeNumber','Type Nr','Type No'],
    'qty': ['Qty','Quantity','RBR-Quantity','Amount','Total','Total quantity','Sum'],
    'unit': ['Unit','Units','UoM','Measurement unit','RBR-Units','RBR-Unit'],
}


_REF_ROLE_PATTERNS = [
    ('kontrolltabel',   ['kontrolltabel', 'occ_pr', 'occ-pr']),
    ('loi_matrix',      ['_loi', '-loi', 'loibuilding', 'loiinfra', 'loithird']),
    ('log_matrix',      ['_log', '-log', 'logbuilding', 'loginfra', 'logthird']),
    ('objectid_matrix', ['objectidtypenr', 'objectid_type', '_typenr', '-typenr']),
    ('uniclass',        ['uniclass']),
    ('pbs',             ['_pbs', '-pbs', 'payitem', 'pbs_']),
    ('naming_rules',    ['naming_conv', 'namingconv', 'file_naming', 'filenaming']),
    ('design_req',      ['ds1design', 'ds2design', 'ds3design', '_ds1_', '_ds2_', '_ds3_']),
]


def _detect_ref_role(name: str) -> str:
    n = name.lower()
    for role, patterns in _REF_ROLE_PATTERNS:
        if any(p in n for p in patterns):
            return role
    return ''


def _has(df, group):
    return bool(find_col(df, group))


def score_file(path: Path) -> dict:
    path = Path(path)
    name = path.name.lower()
    folder = str(path.parent).lower()
    if path.suffix.lower() not in TABLE_EXTS or path.name.startswith('~$'):
        return {'path': str(path), 'file': path.name, 'role': 'ignored', 'quantity_kind': '', 'score': 0, 'columns_found': '', 'reason': 'Unsupported or temporary file', 'header_row': '', 'sheet': ''}
    df, meta = read_table_with_meta(path, nrows=800)
    if df is None or df.empty or len(df.columns) == 0:
        return {'path': str(path), 'file': path.name, 'role': 'ignored', 'quantity_kind': '', 'score': 0, 'columns_found': '', 'reason': 'Empty or unreadable table', 'header_row': meta.get('header_row',''), 'sheet': meta.get('sheet','')}

    model_hits = sum(1 for cands in MODEL_COL_CANDIDATES.values() if find_col(df, cands))
    qty_hits = sum(1 for cands in QTY_COL_CANDIDATES.values() if find_col(df, cands))
    cols = ', '.join(map(str, list(df.columns)[:25]))
    text = ' '.join([name, folder, cols.lower()])

    name_qex = 'qex' in text
    name_qto = 'qto' in text or 'boq' in text or '_bq_' in text
    name_qty = any(k in text for k in ['mah', 'quant', 'quantity', 'qex', 'qto', 'boq', '_bq_'])
    name_model = any(k in text for k in ['mudeli', 'model', 'extract', 'väljav', 'valjav', 'default'])

    model_score = model_hits * 12 + (8 if name_model else 0)
    qty_score = qty_hits * 8 + (18 if name_qex or name_qto else 0) + (8 if name_qty else 0)

    if _has(df, MODEL_COL_CANDIDATES['object_id']):
        model_score += 30
    if _has(df, QTY_COL_CANDIDATES['qty']) and _has(df, QTY_COL_CANDIDATES['unit']):
        qty_score += 18
    if name_qex or name_qto:
        qty_score += 15
        # QEX/QTO templates sometimes have weak headers but filename is reliable enough for quantity role.

    if name_qex:
        qty_kind = 'qex'
    elif name_qto:
        qty_kind = 'qto'
    else:
        qty_kind = 'quantity_unknown'

    ref_role = _detect_ref_role(name) or _detect_ref_role(folder)

    if model_score >= qty_score and model_score >= 25:
        role, reason, score = 'model', 'Model-like BIM columns detected', model_score
    elif qty_score >= 22 or name_qex or name_qto:
        role, reason, score = 'quantity', 'Quantity/QEX/QTO-like table detected', qty_score
    elif ref_role:
        role, reason, score = ref_role, f'Reference file detected ({ref_role})', max(model_score, qty_score)
    else:
        role, reason, score = 'ignored', 'No clear model or quantity table signature', max(model_score, qty_score)

    return {
        'path': str(path), 'file': path.name, 'role': role, 'quantity_kind': qty_kind if role == 'quantity' else '',
        'score': round(float(score),2), 'columns_found': cols, 'reason': reason,
        'rows_sampled': len(df), 'header_row': meta.get('header_row',''), 'sheet': meta.get('sheet','')
    }


def detect_files(paths: list[Path]) -> tuple[dict, pd.DataFrame]:
    candidates = [score_file(Path(p)) for p in paths]
    det = pd.DataFrame(candidates)
    selected = {'model': None, 'qex': None, 'qto': None, 'quantities': []}
    if det.empty:
        return selected, det
    models = det[det['role']=='model'].sort_values('score', ascending=False)
    if not models.empty:
        selected['model'] = models.iloc[0]['path']
    qtys = det[det['role']=='quantity'].sort_values('score', ascending=False)
    selected['quantities'] = qtys['path'].tolist()
    qex = qtys[qtys['quantity_kind']=='qex']
    qto = qtys[qtys['quantity_kind']=='qto']
    if not qex.empty:
        selected['qex'] = qex.iloc[0]['path']
    if not qto.empty:
        selected['qto'] = qto.iloc[0]['path']
    return selected, det
