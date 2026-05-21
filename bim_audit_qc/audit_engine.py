from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import re
import pandas as pd
from .io_utils import read_table_with_meta, find_col, to_num, as_text, is_empty

MODEL_COLS = {
    'object_id': ['RBR-Object_ID','Object_ID','ObjectID','Object Id','RBR_Object_ID','Element ID','Asset ID'],
    'object_parent': ['RBR-Object_IDparent','Object_IDparent','Parent Object_ID','Parent Object ID','Object Parent'],
    'discipline_code': ['RBR-Discipline_Code','Discipline_Code','Discipline code','Discipline'],
    'type_number': ['RBR-Type_number','Type_number','Type-number','TypeNumber','Type Nr','Type No','RBR_Type_number'],
    'pr_code': ['RBR-Pr_Code','Pr_Code','Pr-Code','Product Code','PayItem','Uniclass','Classification','RBR_Pr_Code'],
    'occ': ['RBR-OCC','OCC','Occupation','Object classification','RBR_OCC'],
    'qty': ['RBR-Quantity','Quantity','Qty','Amount','Total','RBR_Qty','RBR Quantity'],
    'unit': ['RBR-Units','RBR-Unit','Unit','Units','UoM','Measurement unit','RBR_Units'],
    'name': ['Name','Element name','Object name'],
    'phase_demolished': ['RBR-Phase_Demolished','Phase_Demolished'],
    'route_code': ['RBR-Route_code','RBR-Route_Code','Route_code','Route Code'],
    'payitem_qty': ['RBR-PayItem_quantity','PayItem_quantity','PayItem Qty'],
    'payitem_unit': ['RBR-PayItem_unit','PayItem_unit','PayItem Unit'],
    'design_life': ['RBR-Design_life','Design_life'],
}
QTY_COLS = {
    'object_id': ['RBR-Object_ID','Object_ID','ObjectID','Object Id','Element ID','Asset ID'],
    'occ': ['RBR-OCC','OCC','Occupation','Object classification'],
    'pr_code': ['RBR-Pr_Code','Pr_Code','Pr-Code','Product Code','PayItem','Uniclass','Classification','Item code'],
    'type_number': ['RBR-Type_number','Type_number','Type-number','TypeNumber','Type Nr','Type No'],
    'qty': ['Qty','Quantity','RBR-Quantity','Amount','Total','Total quantity','Sum'],
    'unit': ['Unit','Units','UoM','Measurement unit','RBR-Units','RBR-Unit'],
}
ALLOWED_UNITS = {'m','m2','m3','m²','m³','pcs','pc','tk','kg','t','l','m1','nr','m^2','m^3','each','ea','lump sum'}
_PLACEHOLDERS = {'', '--', '-', 'nan', 'none', 'null', 'notdefined', 'not defined', 'n/a'}


def issue(rule_id, tier, severity, row_index, element, attr, requirement, model_value, message, cross_check='', context=None):
    return {
        'rule_id': rule_id, 'tier': tier, 'severity': severity, 'row_index': row_index, 'element': element,
        'attribute': attr, 'requirement': requirement, 'model_value': model_value, 'message': message,
        'cross_check': cross_check, 'context': context or {}
    }


def resolve_cols(df, mapping):
    return {k: find_col(df, v) for k, v in mapping.items()}


_RBR_CANONICAL = {
    'object_id': 'RBR-Object_ID', 'object_parent': 'RBR-Object_IDparent',
    'discipline_code': 'RBR-Discipline_Code', 'type_number': 'RBR-Type_number',
    'pr_code': 'RBR-Pr_Code', 'occ': 'RBR-OCC', 'qty': 'RBR-Quantity',
    'unit': 'RBR-Units', 'name': 'Name', 'phase_demolished': 'RBR-Phase_Demolished',
    'route_code': 'RBR-Route_code', 'payitem_qty': 'RBR-PayItem_quantity',
    'payitem_unit': 'RBR-PayItem_unit', 'design_life': 'RBR-Design_life',
}


def _build_col_mapping_doc(cols: dict) -> 'pd.DataFrame':
    rows = []
    for std, detected in cols.items():
        rows.append({
            'Standard_Key': std,
            'RBR_Canonical': _RBR_CANONICAL.get(std, std),
            'Detected_Column': detected or 'NOT FOUND',
            'Status': 'OK' if detected else 'MISSING',
            'Note': '' if detected else 'Column not detected; checks requiring this attribute were skipped.',
        })
    return pd.DataFrame(rows)


def object_parts(value: str):
    val = as_text(value).upper()
    return val.split('-') if val and val.lower() not in _PLACEHOLDERS else []


def obj_prefix(value: str) -> str:
    parts = object_parts(value)
    return '-'.join(parts[:3]) if len(parts) >= 3 else ''


def row_disc(value: str) -> str:
    parts = object_parts(value)
    return parts[0] if parts else ''


def object_class(value: str) -> str:
    parts = object_parts(value)
    return '-'.join(parts[:2]) if len(parts) >= 2 else ''


def _series(df, col):
    return df[col] if col and col in df.columns else pd.Series([''] * len(df), index=df.index)


def _excel_row(idx):
    return int(idx) + 2


def _element(row, obj_col, rownum):
    if obj_col:
        val = as_text(row.get(obj_col,''))
        if not is_empty(val):
            return val
    return f'row {rownum}'


def _has_valid_object(row, obj_col) -> bool:
    return bool(obj_col and not is_empty(as_text(row.get(obj_col,''))))


def _looks_float_artifact(v: str) -> bool:
    s = as_text(v)
    if not s or '.' not in s:
        return False
    # exporter artefacts: 14.850000000000001, 3.7800000000000002, etc.
    m = re.match(r'^-?\d+\.(\d+)$', s)
    if not m:
        return False
    dec = m.group(1)
    return len(dec) >= 12 and (dec.endswith('0001') or dec.endswith('0002') or dec.endswith('9998') or dec.endswith('9999') or '0000000000' in dec or '9999999999' in dec)


def _structural_validation(model_df, cols, model_meta, findings, limitations):
    if model_df is None or model_df.empty:
        return
    col_names = list(model_df.columns)
    seen: dict = {}
    for c in col_names:
        cs = as_text(c).strip()
        seen[cs] = seen.get(cs, 0) + 1
    dupe_cols = [cs for cs, cnt in seen.items() if cnt > 1 and cs]
    if dupe_cols:
        findings.append(issue(
            'STRUCTURAL_DUPLICATE_COLUMNS', 'T0', 'ERROR', 'MODEL', 'MODEL', 'Column headers',
            'All column names must be unique in the model export',
            ', '.join(dupe_cols[:20]),
            f'Duplicate column header(s): {", ".join(dupe_cols[:20])}. This causes incorrect column mapping and silent data loss.',
            'Dataset structural sanity / export settings'
        ))
    suspicious = [repr(str(c))[:40] for c in col_names if str(c) != str(c).strip() or any(ord(ch) < 32 for ch in str(c))]
    if suspicious:
        findings.append(issue(
            'STRUCTURAL_HIDDEN_CHARS', 'T0', 'WARNING', 'MODEL', 'MODEL', 'Column headers',
            'Column names must not contain leading/trailing whitespace or control characters',
            ', '.join(suspicious[:10]),
            f'Column header(s) with hidden characters or whitespace: {", ".join(suspicious[:10])}. May cause attribute lookup failures.',
            'Dataset structural sanity / export settings'
        ))
    blank_rows = int(model_df.apply(lambda row: all(is_empty(as_text(v)) for v in row), axis=1).sum())
    if blank_rows > 0:
        findings.append(issue(
            'STRUCTURAL_BLANK_ROWS', 'T0', 'WARNING', 'MODEL', 'MODEL', 'Dataset rows',
            'Model export must not contain entirely blank rows',
            f'{blank_rows} blank rows',
            f'{blank_rows} completely blank row(s) in the model export. May indicate export issues or merged cell artifacts.',
            'Dataset structural sanity / export settings'
        ))
    if model_meta:
        hrow = int(model_meta.get('header_row', 0) or 0)
        if hrow > 3:
            limitations.append({
                'Severity': 'INFO',
                'Limitation': f'Header row detected at row {hrow}, not row 1. Title/merged rows were skipped. Verify the correct header was identified.',
                'Cross-check': 'Dataset_Sanity / column mapping'
            })


def _dataset_sanity(model_df, cols, findings, limitations):
    if model_df is None or model_df.empty:
        limitations.append({'Severity':'CRITICAL ERROR','Limitation':'Model extract could not be read or is empty.', 'Cross-check':'File accessibility / dataset sanity'})
        return
    obj_col = cols.get('object_id')
    for idx, row in model_df.iterrows():
        nonempty = [as_text(v) for v in row.tolist() if not is_empty(v)]
        rownum = _excel_row(idx)
        element = _element(row,obj_col,rownum)
        if len(nonempty) == 1:
            findings.append(issue('PARTIAL_ELEMENT_EXPORT','T0','ERROR',rownum,element,'WHOLE ROW','Each element row must contain full attribute data',nonempty[0],'Row contains only one non-empty value; this indicates a partial or shifted export.','Dataset sanity validation'))
        # Rows with placeholder Object_ID and many placeholder / blank fields are partial element exports, not dozens of separate core-code errors.
        if obj_col and is_empty(as_text(row.get(obj_col,''))):
            meaningful = [as_text(v) for v in row.tolist() if not is_empty(v)]
            if meaningful:
                findings.append(issue('PARTIAL_ELEMENT_EXPORT','T0','ERROR',rownum,element,'WHOLE ROW','Every model row must identify a complete BIM element', '(partial row)', 'Row has no valid Object_ID. Core attribute checks are suppressed for this row to avoid cascading false positives.','Dataset sanity validation'))


def _discipline_code_is_package_level(model_df, obj_col, disc_col) -> bool:
    if not obj_col or not disc_col:
        return False
    obj_prefixes = _series(model_df, obj_col).apply(row_disc)
    obj_prefixes = [p for p in obj_prefixes if p]
    dvals = [as_text(v).upper() for v in _series(model_df, disc_col).tolist() if not is_empty(v)]
    if not obj_prefixes or not dvals:
        return False
    distinct_obj_disc = set(obj_prefixes)
    cnt = Counter(dvals)
    dominant, dom_n = cnt.most_common(1)[0]
    # In bridge/package extracts RBR-Discipline_Code can be a package/model discipline (e.g. BR),
    # while Object_ID rows contain DR/ED/STR object-level disciplines. Treat as package metadata.
    return len(distinct_obj_disc) >= 2 and dom_n / max(1, len(dvals)) >= 0.80 and dominant not in distinct_obj_disc


def _object_id_checks(model_df, cols, refs, discipline, findings):
    obj_col = cols.get('object_id'); disc_col = cols.get('discipline_code')
    if not obj_col:
        findings.append(issue('OBJ_COLUMN_MISSING','T0','CRITICAL ERROR','MODEL','MODEL','RBR-Object_ID','Required core column must exist','COLUMN MISSING','Model extract does not contain a detectable Object_ID column.','BIM Manual / ObjectIDTypeNr matrix'))
        return
    package_level_disc = discipline == 'AUTO_MIXED' and _discipline_code_is_package_level(model_df, obj_col, disc_col)
    objs = _series(model_df, obj_col).apply(as_text)
    valid_mask = ~objs.apply(is_empty)
    dupes = objs[valid_mask & objs.duplicated(keep=False)]
    dupe_set = set(dupes)
    for idx, val in objs.items():
        row = model_df.loc[idx]
        rownum = _excel_row(idx)
        element = _element(row, obj_col, rownum)
        if is_empty(val):
            findings.append(issue('OBJ_MISSING','T1','CRITICAL ERROR',rownum,element,'RBR-Object_ID','Must be populated and unique',val,'Object_ID is missing or contains a placeholder value.','ObjectIDTypeNr matrix / BIM Manual codification requirements'))
            continue
        if val in dupe_set:
            findings.append(issue('OBJ_DUPLICATE','T1','CRITICAL ERROR',rownum,element,'RBR-Object_ID','Must be unique',val,'Object_ID is duplicated in the model extract.','ObjectID/Type_number meelespea: RBR-Object_ID must be unique'))
        parts = object_parts(val); prefix = obj_prefix(val)
        if len(parts) != 4:
            findings.append(issue('OBJ_FORMAT','T1','CRITICAL ERROR',rownum,element,'RBR-Object_ID','Expected DISCIPLINE-CLASS-SUBCLASS-NUMBER, e.g. STR-SCL-SCP-0039',val,'Object_ID does not split into four required segments.','ObjectIDTypeNr matrix / BIM Manual codification requirements'))
            continue
        if not parts[-1].isdigit() or not (1 <= int(parts[-1]) <= 9999):
            findings.append(issue('OBJ_NUMBER_RANGE','T1','CRITICAL ERROR',rownum,element,'RBR-Object_ID','Final segment must be numeric 0001-9999',val,'Object_ID final numeric part is outside 0001-9999 or is not numeric.','Object_ID and Type_number meelespea'))
        if refs.valid_obj_prefixes and prefix not in refs.valid_obj_prefixes:
            findings.append(issue('OBJ_PREFIX_NOT_IN_MATRIX','T2','MANUAL_REVIEW',rownum,element,'RBR-Object_ID','Object_ID first three segments should exist in ObjectIDTypeNr matrix',prefix,'Object_ID prefix was not found in the loaded ObjectID/TypeNr matrices. Verify whether this is missing reference scope or a true Object_ID coding issue.','ObjectIDTypeNr Building/Infra matrix', {'similar_prefixes': refs.closest_prefixes(prefix) if hasattr(refs,'closest_prefixes') else []}))
        if disc_col and not package_level_disc:
            disc = as_text(row.get(disc_col,''))
            if not is_empty(disc) and disc.upper() != parts[0]:
                findings.append(issue('DISCIPLINE_MISMATCH','T2','ERROR',rownum,element,'RBR-Discipline_Code',f'Must match Object_ID discipline prefix {parts[0]}',disc,'RBR-Discipline_Code does not match the Object_ID first prefix.','Object_ID discipline consistency check'))
        if discipline and discipline != 'AUTO_MIXED' and parts[0] != discipline:
            findings.append(issue('OBJ_DISCIPLINE_SCOPE_INFO','T1','INFO',rownum,element,'RBR-Object_ID',f'Selected audit scope is {discipline}',val,'Row belongs to a different Object_ID discipline prefix than the selected audit scope. In mixed extracts this is not a defect.','Scope handling'))


def _type_checks(model_df, cols, findings):
    obj_col, type_col = cols.get('object_id'), cols.get('type_number')
    if not type_col:
        findings.append(issue('TYPE_COLUMN_MISSING','T0','ERROR','MODEL','MODEL','RBR-Type_number','Required Type_number column should exist','COLUMN MISSING','Model extract does not contain a detectable Type_number column.','ObjectIDTypeNr matrix'))
        return
    for idx, row in model_df.iterrows():
        if not _has_valid_object(row, obj_col):
            continue
        rownum = _excel_row(idx); element = _element(row, obj_col, rownum)
        typ = as_text(row.get(type_col,''))
        if is_empty(typ):
            findings.append(issue('TYPE_MISSING','T1','ERROR',rownum,element,'RBR-Type_number','Must be populated for typed objects',typ,'Type_number is missing or placeholder.','ObjectIDTypeNr matrix'))
            continue
        if '-' not in typ:
            findings.append(issue('TYPE_FORMAT','T1','ERROR',rownum,element,'RBR-Type_number','Expected CHARACTER-NUMBER, e.g. SCP-0039',typ,'Type_number does not contain the expected separator.','ObjectID/Type_number meelespea'))
        if obj_col:
            obj = as_text(row.get(obj_col,'')); oparts = object_parts(obj); tprefix = typ.split('-')[0] if '-' in typ else typ
            if len(oparts) == 4 and tprefix and oparts[1] and tprefix[0].upper() != oparts[1][0].upper():
                findings.append(issue('OBJ_TYPE_RELATION','T2','WARNING',rownum,element,'RBR-Type_number','First character of Type_number should align with Object_ID character code',typ,f'Type_number prefix {tprefix} does not share the first character with Object_ID character segment {oparts[1]}.','Object_ID and Type_number meelespea'))


def _classification_checks(model_df, cols, refs, findings):
    obj_col, pr_col, occ_col, type_col = cols.get('object_id'), cols.get('pr_code'), cols.get('occ'), cols.get('type_number')
    if pr_col:
        for idx, row in model_df.iterrows():
            if not _has_valid_object(row, obj_col):
                continue
            rownum = _excel_row(idx); element = _element(row, obj_col, rownum); pr = as_text(row.get(pr_col,''))
            if is_empty(pr):
                findings.append(issue('PR_MISSING','T1','ERROR',rownum,element,'RBR-Pr_Code','Must be populated',pr,'Pr_Code is missing or placeholder.','Uniclass / product code tables'))
            elif refs.valid_pr_codes and pr not in refs.valid_pr_codes:
                findings.append(issue('PR_UNKNOWN','T2','ERROR',rownum,element,'RBR-Pr_Code','Must exist in loaded Uniclass/reference tables',pr,'Pr_Code was not found in the loaded reference code list.','Uniclass2015 / product code reference'))
    else:
        findings.append(issue('PR_COLUMN_MISSING','T0','ERROR','MODEL','MODEL','RBR-Pr_Code','Pr_Code column should exist','COLUMN MISSING','Model extract does not contain a detectable Pr_Code column.','Uniclass/Product code references'))

    if occ_col and pr_col and obj_col:
        for idx, row in model_df.iterrows():
            if not _has_valid_object(row, obj_col):
                continue
            rownum = _excel_row(idx); obj = as_text(row.get(obj_col,'')); prefix = obj_prefix(obj)
            pr = as_text(row.get(pr_col,'')); occ = as_text(row.get(occ_col,'')); typ = as_text(row.get(type_col,'')) if type_col else ''
            if is_empty(pr) or is_empty(occ) or is_empty(prefix):
                continue
            if refs.valid_occ_pr_prefix and (prefix, pr, occ) not in refs.valid_occ_pr_prefix:
                findings.append(issue('OCC_PR_PREFIX_COMBO','T2','ERROR',rownum,obj,'RBR-OCC/RBR-Pr_Code/Object_ID prefix','Combination should exist in OCC-Pr-Prefix reference matrix where applicable',f'{prefix} | {pr} | {occ}','Object_ID prefix / Pr_Code / OCC combination was not found exactly in the loaded reference matrix.','OBJ_ID-Pr_Code-OCC kontrolltabel', {'prefix': prefix, 'pr_code': pr, 'occ': occ, 'type_number': typ}))


def _hierarchy_checks(model_df, cols, findings):
    obj_col, parent_col = cols.get('object_id'), cols.get('object_parent')
    if not obj_col or not parent_col:
        return
    objs = set(_series(model_df, obj_col).apply(as_text).loc[lambda s: ~s.apply(is_empty)])
    parent_map = {}
    for idx, row in model_df.iterrows():
        if not _has_valid_object(row, obj_col):
            continue
        rownum = _excel_row(idx); obj = as_text(row.get(obj_col,'')); parent = as_text(row.get(parent_col,''))
        if is_empty(parent):
            continue
        parent_map[obj] = parent
        if parent == obj:
            findings.append(issue('HIERARCHY_SELF_REFERENCE','T2','CRITICAL ERROR',rownum,obj,'RBR-Object_IDparent','Parent must not equal child Object_ID',parent,'Element references itself as parent.','Hierarchy validation'))
        elif parent not in objs:
            findings.append(issue('PARENT_MISSING','T2','ERROR',rownum,obj,'RBR-Object_IDparent','Referenced parent must exist in model dataset',parent,'Parent Object_ID is referenced but does not exist in the model extract.','Hierarchy validation'))
        elif row_disc(parent) and row_disc(obj) and row_disc(parent) != row_disc(obj):
            findings.append(issue('PARENT_DISCIPLINE_MISMATCH','T2','ERROR',rownum,obj,'RBR-Object_IDparent','Child and parent should use consistent discipline prefix',parent,'Parent and child Object_ID discipline prefixes differ.','Hierarchy validation'))
    for obj in list(parent_map)[:50000]:
        seen = set(); cur = obj
        while cur in parent_map:
            if cur in seen:
                findings.append(issue('CIRCULAR_HIERARCHY','T2','CRITICAL ERROR','MODEL',obj,'RBR-Object_IDparent','Hierarchy must not contain loops',cur,'Circular parent-child hierarchy detected.','Hierarchy validation'))
                break
            seen.add(cur); cur = parent_map[cur]


def _quantity_value_checks(model_df, cols, findings):
    obj_col, qty_col, unit_col = cols.get('object_id'), cols.get('qty'), cols.get('unit')
    if qty_col:
        nums = to_num(_series(model_df, qty_col))
        for idx, v in _series(model_df, qty_col).items():
            row = model_df.loc[idx]
            if not _has_valid_object(row, obj_col):
                continue
            rownum = _excel_row(idx); element = _element(row, obj_col, rownum)
            if is_empty(v):
                findings.append(issue('QTY_MISSING','T1','ERROR',rownum,element,'RBR-Quantity','Must be populated for quantity extraction',v,'Quantity is missing or placeholder.','QEX/QTO / 5D quantity requirements'))
            elif pd.isna(nums.loc[idx]):
                findings.append(issue('QTY_NOT_NUMERIC','T1','ERROR',rownum,element,'RBR-Quantity','Must be numeric',v,'Quantity is not numeric.','QEX/QTO / 5D quantity requirements'))
            elif nums.loc[idx] < 0:
                findings.append(issue('QTY_NEGATIVE','T1','ERROR',rownum,element,'RBR-Quantity','Quantity must not be negative unless explicitly justified',v,'Quantity is negative.','Quantity sanity validation'))
            elif nums.loc[idx] == 0:
                findings.append(issue('QTY_ZERO','T1','WARNING',rownum,element,'RBR-Quantity','Quantity should normally be greater than zero unless justified',v,'Quantity is zero; verify whether this is intended.','Quantity sanity validation'))
            elif _looks_float_artifact(v):
                findings.append(issue('QTY_NUMERIC_PRECISION','T1','WARNING',rownum,element,'RBR-Quantity','Quantity values should be rounded to stable, readable precision',v,'Quantity value appears to contain floating-point export artefacts and should be rounded consistently.','General BIM data-quality / numerical hygiene'))
    else:
        findings.append(issue('QTY_COLUMN_MISSING','T0','ERROR','MODEL','MODEL','RBR-Quantity','Quantity column should exist','COLUMN MISSING','Model extract does not contain a detectable quantity column.','QEX/QTO / 5D quantity requirements'))
    if unit_col:
        for idx, unit in _series(model_df, unit_col).apply(as_text).items():
            row = model_df.loc[idx]
            if not _has_valid_object(row, obj_col):
                continue
            rownum = _excel_row(idx); element = _element(row, obj_col, rownum)
            if is_empty(unit):
                findings.append(issue('UNIT_MISSING','T1','ERROR',rownum,element,'RBR-Units','Must be populated',unit,'Unit is missing or placeholder.','QEX/QTO unit consistency'))
            elif unit.lower() not in ALLOWED_UNITS:
                findings.append(issue('UNIT_UNUSUAL','T1','WARNING',rownum,element,'RBR-Units','Should use a consistent accepted unit value',unit,'Unit value is unusual; verify whether it is allowed for this project.','QEX/QTO unit consistency'))
    else:
        findings.append(issue('UNIT_COLUMN_MISSING','T0','ERROR','MODEL','MODEL','RBR-Units','Unit column should exist','COLUMN MISSING','Model extract does not contain a detectable unit column.','QEX/QTO unit consistency'))


def _infer_loi_object_type(disc: str, obj: str, row=None, cols=None) -> str:
    """Best-effort object-type inference for LOI applicability.

    It does not invent final requirements; it only maps known Object_ID classes to LOI object-type columns
    so type-specific LOI requirements (e.g. STR Main structural element H1/Q1/V1) are not applied to fills,
    excavations, waterproofing, etc.
    """
    disc = as_text(disc).upper()
    parts = object_parts(obj)
    if len(parts) < 2:
        return ''
    cls = parts[1].upper()
    subclass = parts[2].upper() if len(parts) > 2 else ''
    if disc == 'STR':
        if cls in {'FLL'}:
            return 'Fill'
        if cls in {'EXC'}:
            return 'Excavation'
        if cls in {'WTP'}:
            return 'Waterproofing'
        if cls in {'EXP'}:
            return 'Joint elements'
        if cls in {'SSTR'}:
            return 'Reinforcement'
        if cls in {'ST','STA','STC','SSB','SL'}:
            return 'Stair'
        if cls in {'FNC','BRR','NBW'}:
            return 'Fence, Barrier, Noise Wall'
        if cls in {'LOG'}:
            return 'Logo'
        if cls in {'FXT','FIX'}:
            return 'Fixture'
        # default structural objects: columns, deck, beams, slabs, piles, steel, etc.
        return 'Main structural element'
    if disc == 'DR':
        if cls == 'PIP': return 'Pipe'
        if cls == 'FLL': return 'Fill'
        if cls == 'GEX': return 'Excavation'
    return ''


def _row_list_from_indices(indices, limit=60):
    rows = [str(_excel_row(i)) for i in indices]
    if len(rows) > limit:
        return ','.join(rows[:limit]) + f',... (+{len(rows)-limit} more)'
    return ','.join(rows)


def _loi_attribute_checks(model_df, cols, refs, findings):
    """Validate LOI-required attributes without flooding the report with one row per missing column.

    Important distinction:
    - if the required attribute column exists but individual values are empty, record row-level errors;
    - if the required attribute column is absent from the whole export, record one SYSTEMIC ERROR per
      discipline/object-type/attribute group because there is no source cell to comment.
    """
    obj_col = cols.get('object_id')
    if not obj_col or not hasattr(refs, 'required_attrs_for_row'):
        return
    attr_col_cache = {}
    missing_column_groups = defaultdict(list)
    missing_column_context = {}

    for idx, row in model_df.iterrows():
        obj = as_text(row.get(obj_col,''))
        if is_empty(obj):
            continue
        d = row_disc(obj)
        if not d:
            continue
        loi_type = _infer_loi_object_type(d, obj, row, cols)
        required = refs.required_attrs_for_row(d, loi_type)
        # Do not fall back to all discipline attributes. That created false positives because
        # several LOI matrices contain many type-specific fields. If the object type cannot be
        # inferred, only common attributes are used by required_attrs_for_row().
        if not required:
            continue
        rownum = _excel_row(idx)
        element = _element(row, obj_col, rownum)
        for attr in sorted(required):
            if attr not in attr_col_cache:
                attr_col_cache[attr] = find_col(model_df, [attr])
            actual_col = attr_col_cache[attr]
            if not actual_col:
                key = (d, loi_type or 'COMMON/UNMAPPED', attr)
                missing_column_groups[key].append(idx)
                missing_column_context[key] = (d, loi_type, attr)
            else:
                v = as_text(row.get(actual_col,''))
                if is_empty(v):
                    findings.append(issue(
                        'LOI_REQUIRED_VALUE_EMPTY','T1','ERROR',rownum,element,attr,'Mandatory at LOI 400',v,
                        f'{attr} is mandatory at LOI 400 for {d}' + (f' {loi_type} elements' if loi_type else ' elements') + ', but the model value is empty/placeholder.',
                        'LOI matrix / BIM objects attributes matrix', {'discipline': d, 'loi_object_type': loi_type}
                    ))

    for (d, loi_type, attr), indices in sorted(missing_column_groups.items()):
        count = len(indices)
        rows = _row_list_from_indices(indices)
        findings.append(issue(
            'LOI_REQUIRED_COLUMN_MISSING_SYSTEMIC','T1','SYSTEMIC ERROR',
            rows, f'{d} {loi_type} ({count} elements)', attr, 'Mandatory at LOI 400',
            'COLUMN MISSING',
            f'{attr} is mandatory at LOI 400 for {d}' + (f' {loi_type} elements' if loi_type and loi_type != 'COMMON/UNMAPPED' else ' elements') +
            f', but the attribute column is missing from the model export. Affected valid elements: {count}. Because the source column does not exist, the issue is reported once as systemic instead of repeated on every row.',
            'LOI matrix / BIM objects attributes matrix', {'discipline': d, 'loi_object_type': loi_type, 'affected_count': count}
        ))


def _additional_attribute_sanity(model_df, cols, findings):
    # Checks not fully covered by LOI parsing but visible in current BR1720 dataset and V9 forensic prompt.
    obj_col = cols.get('object_id')
    for key, attr in [('design_life','RBR-Design_life'),('payitem_qty','RBR-PayItem_quantity'),('payitem_unit','RBR-PayItem_unit')]:
        col = cols.get(key)
        if not col:
            continue
        for idx, row in model_df.iterrows():
            rownum = _excel_row(idx); element = _element(row, obj_col, rownum)
            v = as_text(row.get(col,''))
            # include rows without Object_ID too because these fields help identify partial export rows
            if is_empty(v):
                findings.append(issue('ATTRIBUTE_EMPTY','T1','ERROR',rownum,element,attr,'Attribute should be populated when included in export',v,f'{attr} is empty or placeholder.','Model attribute completeness / dataset sanity'))


def _systemic_empty_key_attributes(model_df, cols, findings):
    """Project-wide checks for key RB attributes that are frequently required across mixed infra exports.

    This complements the LOI parser: if a key attribute column exists but is essentially empty for the whole
    valid model export, record it per affected element. It also avoids duplicate row/attribute findings already
    raised by the LOI pass.
    """
    obj_col = cols.get('object_id')
    if not obj_col:
        return
    valid_indices = [idx for idx, row in model_df.iterrows() if _has_valid_object(row, obj_col)]
    if not valid_indices:
        return
    existing = {(str(f.get('row_index')), str(f.get('attribute'))) for f in findings}
    for key, attr in [('route_code','RBR-Route_code'), ('phase_demolished','RBR-Phase_Demolished')]:
        col = cols.get(key)
        if not col:
            continue
        empty_indices = [idx for idx in valid_indices if is_empty(as_text(model_df.loc[idx].get(col,'')))]
        if len(empty_indices) / max(1, len(valid_indices)) < 0.80:
            continue
        for idx in empty_indices:
            row = model_df.loc[idx]
            rownum = _excel_row(idx)
            if (str(rownum), attr) in existing:
                continue
            element = _element(row, obj_col, rownum)
            findings.append(issue(
                'KEY_ATTRIBUTE_EMPTY_SYSTEMIC','T1','ERROR',rownum,element,attr,'Key Rail Baltica attribute should be populated where applicable',
                as_text(row.get(col,'')),
                f'{attr} is present in the model export but is empty/placeholder for the majority of valid elements. This indicates a systemic attribute population issue.',
                'LOI matrix / Rail Baltica BIM data completeness check', {'empty_count': len(empty_indices), 'valid_rows': len(valid_indices)}
            ))


def _prep_quantity_df(path):
    qdf, meta = read_table_with_meta(path)
    qcols = resolve_cols(qdf, QTY_COLS)
    return qdf, qcols, meta


def _aggregate(df, cols, keys=('pr_code','type_number','unit')):
    required = list(keys) + ['qty']
    if df is None or df.empty or any(not cols.get(k) for k in required):
        return pd.DataFrame(), [k for k in required if not cols.get(k)]
    d = pd.DataFrame()
    for k in keys:
        d[k] = _series(df, cols[k]).apply(as_text)
    d['qty_num'] = to_num(_series(df, cols['qty']))
    d = d[~d['qty_num'].isna()]
    g = d.groupby(list(keys), dropna=False)['qty_num'].sum().reset_index()
    return g, []


def quantity_control(model_df, cols, quantity_paths):
    rows = []
    if model_df is None or model_df.empty:
        return pd.DataFrame(rows)
    mgroup, missing = _aggregate(model_df, cols, keys=('pr_code','type_number','unit'))
    if missing:
        return pd.DataFrame([{'Pr_Code':'','Type_number':'','Unit':'','Model_Qty':'','QEX_Qty':'','QTO_Qty':'','Model-QEX_Diff':'','Model-QTO_Diff':'','Status':'LIMITATION','Comment':'Model does not contain all required quantity comparison columns: ' + ', '.join(missing)}])
    base = mgroup.rename(columns={'pr_code':'Pr_Code','type_number':'Type_number','unit':'Unit','qty_num':'Model_Qty'})
    qex_total = None; qto_total = None; comments = []
    for qpath in quantity_paths or []:
        qdf, qcols, meta = _prep_quantity_df(qpath)
        fname = Path(qpath).name
        lname = fname.lower()
        kind = 'QEX' if 'qex' in lname else ('QTO' if 'qto' in lname or 'boq' in lname else 'QTY')
        qgroup, qmissing = _aggregate(qdf, qcols, keys=('pr_code','type_number','unit'))
        if qmissing:
            comments.append(f'{fname}: LIMITATION - required columns not detected ({", ".join(qmissing)}); detected header row={meta.get("header_row")}, sheet={meta.get("sheet")}.')
            continue
        qgroup = qgroup.rename(columns={'pr_code':'Pr_Code','type_number':'Type_number','unit':'Unit','qty_num':kind + '_Qty'})
        if kind == 'QEX':
            qex_total = qgroup if qex_total is None else pd.concat([qex_total, qgroup], ignore_index=True).groupby(['Pr_Code','Type_number','Unit'], dropna=False)[kind + '_Qty'].sum().reset_index()
        elif kind == 'QTO':
            qto_total = qgroup if qto_total is None else pd.concat([qto_total, qgroup], ignore_index=True).groupby(['Pr_Code','Type_number','Unit'], dropna=False)[kind + '_Qty'].sum().reset_index()
        else:
            qgroup = qgroup.rename(columns={'QTY_Qty':'QEX_Qty'})
            qex_total = qgroup if qex_total is None else pd.concat([qex_total, qgroup], ignore_index=True).groupby(['Pr_Code','Type_number','Unit'], dropna=False)['QEX_Qty'].sum().reset_index()
            comments.append(f'{fname}: treated as quantity extraction because QEX/QTO kind could not be determined from filename.')
    result = base.copy()
    if qex_total is not None:
        result = result.merge(qex_total, on=['Pr_Code','Type_number','Unit'], how='outer')
    else:
        result['QEX_Qty'] = pd.NA
    if qto_total is not None:
        result = result.merge(qto_total, on=['Pr_Code','Type_number','Unit'], how='outer')
    else:
        result['QTO_Qty'] = pd.NA
    for c in ['Model_Qty','QEX_Qty','QTO_Qty']:
        if c not in result.columns:
            result[c] = pd.NA
        result[c] = pd.to_numeric(result[c], errors='coerce')
    result['Model-QEX_Diff'] = result['Model_Qty'] - result['QEX_Qty']
    result['Model-QTO_Diff'] = result['Model_Qty'] - result['QTO_Qty']
    def status(row):
        status = 'OK'; notes = []
        tol_abs = 0.01
        model_qty = row.get('Model_Qty')
        for label, diff_col, ref_col in [('QEX','Model-QEX_Diff','QEX_Qty'),('QTO','Model-QTO_Diff','QTO_Qty')]:
            ref_qty = row.get(ref_col)
            if pd.isna(ref_qty):
                notes.append(f'{label} not available for this group')
                continue
            diff = abs(row.get(diff_col) or 0)
            denom = max(abs(model_qty or 0), abs(ref_qty or 0), 1.0)
            rel = diff / denom
            ratio_note = ''
            try:
                if model_qty and not pd.isna(model_qty) and abs(model_qty) > 0.000001:
                    ratio = float(ref_qty) / float(model_qty)
                    if abs(ratio - 2.0) <= 0.03 or abs(ratio - 0.5) <= 0.03:
                        ratio_note = f' Detected near {ratio:.3f}x ratio, which indicates a likely double-count/half-count systemic quantity issue.'
            except Exception:
                pass
            if diff > tol_abs and rel > 0.01:
                if ratio_note:
                    status = 'SYSTEMIC ERROR'
                    notes.append(f'Model vs {label} quantity mismatch exceeds tolerance ({rel:.2%}).{ratio_note}')
                else:
                    if status != 'SYSTEMIC ERROR':
                        status = 'ERROR'
                    notes.append(f'Model vs {label} quantity mismatch exceeds tolerance ({rel:.2%})')
            else:
                notes.append(f'Model vs {label} within tolerance ({rel:.2%})')
        return status, '; '.join(notes) if notes else 'Quantities match within tolerance or comparison dataset missing.'
    statuses = result.apply(status, axis=1, result_type='expand') if not result.empty else pd.DataFrame()
    if not result.empty:
        result['Status'] = statuses[0]
        result['Comment'] = statuses[1]
    if comments:
        lim_rows = [{'Pr_Code':'','Type_number':'','Unit':'','Model_Qty':'','QEX_Qty':'','QTO_Qty':'','Model-QEX_Diff':'','Model-QTO_Diff':'','Status':'LIMITATION','Comment':c} for c in comments]
        result = pd.concat([result, pd.DataFrame(lim_rows)], ignore_index=True)
    cols_out = ['Pr_Code','Type_number','Unit','Model_Qty','QEX_Qty','QTO_Qty','Model-QEX_Diff','Model-QTO_Diff','Status','Comment']
    for c in cols_out:
        if c not in result.columns:
            result[c] = ''
    return result[cols_out]


def quantity_findings_from_control(qc: pd.DataFrame, findings):
    if qc is None or qc.empty:
        return
    for _, r in qc.iterrows():
        st = as_text(r.get('Status',''))
        if st in {'ERROR','SYSTEMIC ERROR'}:
            findings.append(issue('QTY_AGGREGATE_MISMATCH','T2',st,'AGGREGATE',f"{r.get('Pr_Code','')}|{r.get('Type_number','')}",'RBR-Quantity','Model, merged QEX and QTO quantities should match by Pr_Code + Type_number + Unit',f"Model={r.get('Model_Qty')} QEX={r.get('QEX_Qty')} QTO={r.get('QTO_Qty')}",as_text(r.get('Comment','')),'QEX/QTO quantity comparison'))
        elif st == 'LIMITATION':
            findings.append(issue('QTY_COMPARISON_LIMITATION','T2','LIMITATION','AGGREGATE','Quantity_Control','Quantity_Control','Quantity comparison must be reproducible','',as_text(r.get('Comment','')),'QEX/QTO header and column detection'))


def _compress_row_level_findings(findings):
    """Condense repetitive forensic findings into systemic rows for a readable report.

    The raw logic remains strict, but the final report should not contain thousands of identical
    rows when one systemic root cause explains them. Row lists are retained in Source_Row for traceability.
    """
    if not findings:
        return findings
    group_rules = {
        'OBJ_PREFIX_NOT_IN_MATRIX': ('attribute','model_value','requirement','cross_check'),
        'OCC_PR_PREFIX_COMBO': ('attribute','model_value','requirement','cross_check'),
        'PARENT_MISSING': ('attribute','model_value','requirement','cross_check'),
        'KEY_ATTRIBUTE_EMPTY_SYSTEMIC': ('attribute','requirement','cross_check'),
        'LOI_REQUIRED_VALUE_EMPTY': ('attribute','requirement','cross_check','message'),
        'OBJ_DUPLICATE': ('attribute','element','requirement','cross_check'),
    }
    threshold = {
        'OBJ_PREFIX_NOT_IN_MATRIX': 5,
        'OCC_PR_PREFIX_COMBO': 5,
        'PARENT_MISSING': 5,
        'KEY_ATTRIBUTE_EMPTY_SYSTEMIC': 20,
        'LOI_REQUIRED_VALUE_EMPTY': 30,
        'OBJ_DUPLICATE': 3,
    }
    buckets = defaultdict(list)
    passthrough = []
    for f in findings:
        rid = f.get('rule_id')
        if rid in group_rules:
            key = (rid,) + tuple(as_text(f.get(k,'')) for k in group_rules[rid])
            buckets[key].append(f)
        else:
            passthrough.append(f)

    out = list(passthrough)
    for key, items in buckets.items():
        rid = key[0]
        if len(items) < threshold.get(rid, 999999):
            out.extend(items)
            continue
        first = dict(items[0])
        rows = []
        examples = []
        for it in items:
            ri = as_text(it.get('row_index',''))
            if ri:
                rows.extend([x.strip() for x in ri.split(',') if x.strip()])
            if len(examples) < 8:
                examples.append(as_text(it.get('element','')))
        # dedupe while preserving order
        seen = set(); rows2=[]
        for r in rows:
            if r not in seen:
                rows2.append(r); seen.add(r)
        row_text = ','.join(rows2[:60]) + (f',... (+{len(rows2)-60} more)' if len(rows2) > 60 else '')
        first['row_index'] = row_text or 'MULTIPLE'
        first['element'] = f'{len(items)} affected rows; examples: ' + ', '.join([e for e in examples if e][:8])
        # True duplicates remain critical. Repeated row-level errors become systemic for readability.
        if rid in {'LOI_REQUIRED_VALUE_EMPTY','KEY_ATTRIBUTE_EMPTY_SYSTEMIC','PARENT_MISSING','OCC_PR_PREFIX_COMBO'}:
            first['severity'] = 'SYSTEMIC ERROR' if rid != 'LOI_REQUIRED_VALUE_EMPTY' else 'SYSTEMIC ERROR'
        if rid == 'OBJ_PREFIX_NOT_IN_MATRIX':
            first['severity'] = 'MANUAL_REVIEW'
        first['message'] = as_text(first.get('message','')) + f' Affected rows: {len(items)}. This repetitive finding was condensed into one systemic report row for readability; see Source_Row for sample/affected row references.'
        ctx = dict(first.get('context') or {})
        ctx['compressed_count'] = len(items)
        first['context'] = ctx
        out.append(first)
    # keep stable order: numeric rows first by first row number, aggregate/model rows later
    def sort_key(f):
        r = as_text(f.get('row_index',''))
        m = re.match(r'\d+', r)
        return (0, int(m.group(0))) if m else (1, r)
    return sorted(out, key=sort_key)


def _element_count_check(model_df, cols, qex_df, qex_cols_r, findings, limitations):
    if qex_df is None or qex_df.empty:
        return
    obj_col = cols.get('object_id')
    qex_obj_col = qex_cols_r.get('object_id')
    if not obj_col or not qex_obj_col:
        return
    model_ids = set(_series(model_df, obj_col).apply(as_text).pipe(lambda s: s[~s.apply(is_empty)]))
    qex_ids = set(_series(qex_df, qex_obj_col).apply(as_text).pipe(lambda s: s[~s.apply(is_empty)]))
    mc, qc_n = len(model_ids), len(qex_ids)
    if mc == 0 or qc_n == 0:
        return
    diff = abs(mc - qc_n)
    rel = diff / max(mc, qc_n)
    if rel > 0.05:
        sev = 'SYSTEMIC ERROR' if rel > 0.20 else 'ERROR'
        in_m = sorted(model_ids - qex_ids)[:5]
        in_q = sorted(qex_ids - model_ids)[:5]
        msg = f'Element count mismatch: model has {mc} unique Object_IDs, QEX has {qc_n} ({rel:.1%} difference). '
        if in_m: msg += f'In model but not QEX (examples): {", ".join(in_m)}. '
        if in_q: msg += f'In QEX but not model (examples): {", ".join(in_q)}.'
        findings.append(issue(
            'ELEMENT_COUNT_MISMATCH', 'T2', sev, 'AGGREGATE', 'Element_Count', 'RBR-Object_ID',
            'Model and QEX element counts should match within 5% tolerance',
            f'Model={mc} QEX={qc_n}', msg, 'QEX element count consistency'
        ))


def _element_level_quantity_check(model_df, cols, qex_df, qex_cols_r, findings, limitations):
    if qex_df is None or qex_df.empty:
        return
    obj_col = cols.get('object_id'); qty_col = cols.get('qty'); unit_col = cols.get('unit')
    qex_obj_col = qex_cols_r.get('object_id'); qex_qty_col = qex_cols_r.get('qty'); qex_unit_col = qex_cols_r.get('unit')
    if not obj_col or not qty_col or not qex_obj_col or not qex_qty_col:
        if not (obj_col and qty_col):
            limitations.append({'Severity': 'LIMITATION', 'Limitation': 'Element-level quantity comparison skipped: model Object_ID or Quantity column not detected.', 'Cross-check': 'QEX element-level comparison'})
        return
    qex_lookup: dict = {}
    for idx, row in qex_df.iterrows():
        oid = as_text(row.get(qex_obj_col, ''))
        if is_empty(oid):
            continue
        qq = to_num(pd.Series([row.get(qex_qty_col, '')])).iloc[0]
        qu = as_text(row.get(qex_unit_col, '')) if qex_unit_col else ''
        if not pd.isna(qq):
            qex_lookup[oid] = (qq, qu)
    if not qex_lookup:
        return
    mismatch_count = 0; reported = 0
    for idx, row in model_df.iterrows():
        oid = as_text(row.get(obj_col, ''))
        if is_empty(oid) or oid not in qex_lookup:
            continue
        mq = to_num(pd.Series([row.get(qty_col, '')])).iloc[0]
        if pd.isna(mq):
            continue
        qq, qu = qex_lookup[oid]
        diff = abs(mq - qq)
        denom = max(abs(float(mq)), abs(float(qq)), 1e-9)
        rel = diff / denom
        if diff <= 0.01 or rel <= 0.01:
            continue
        mismatch_count += 1
        if reported < 200:
            rownum = _excel_row(idx); element = _element(row, obj_col, rownum)
            unit_note = ''
            if unit_col:
                mu = as_text(row.get(unit_col, ''))
                if mu and qu and mu.lower() != qu.lower():
                    unit_note = f' Unit mismatch: model={mu}, QEX={qu}.'
            findings.append(issue(
                'ELEMENT_QTY_MISMATCH', 'T2', 'ERROR', rownum, element, 'RBR-Quantity',
                'Element quantity must match QEX within 1% tolerance',
                f'Model={mq:.4g} QEX={qq:.4g} ({rel:.1%} diff)',
                f'Quantity mismatch: model={mq:.4g} vs QEX={qq:.4g} ({rel:.1%} deviation).{unit_note}',
                'QEX element-level quantity comparison'
            ))
            reported += 1
    if mismatch_count > 200:
        findings.append(issue(
            'ELEMENT_QTY_MISMATCH', 'T2', 'SYSTEMIC ERROR', 'AGGREGATE',
            f'{mismatch_count} elements', 'RBR-Quantity',
            'All element quantities must match QEX within 1% tolerance',
            f'{mismatch_count} mismatches total',
            f'{mismatch_count} elements have quantity mismatches vs QEX. First 200 recorded individually; this systemic row covers the rest.',
            'QEX element-level quantity comparison', {'compressed_count': mismatch_count}
        ))


def _total_quantity_check_by_unit(model_df, cols, qex_df, qex_cols_r, findings):
    if qex_df is None or qex_df.empty or model_df is None or model_df.empty:
        return
    qty_col = cols.get('qty'); unit_col = cols.get('unit')
    qex_qty_col = qex_cols_r.get('qty'); qex_unit_col = qex_cols_r.get('unit')
    if not qty_col or not unit_col or not qex_qty_col or not qex_unit_col:
        return
    def _totals_by_unit(df, qc, uc):
        u = _series(df, uc).apply(as_text).str.strip()
        q = to_num(_series(df, qc))
        valid = ~u.apply(is_empty) & ~q.isna()
        if not valid.any():
            return {}
        return pd.DataFrame({'u': u[valid], 'q': q[valid]}).groupby('u')['q'].sum().to_dict()
    model_totals = _totals_by_unit(model_df, qty_col, unit_col)
    qex_totals = _totals_by_unit(qex_df, qex_qty_col, qex_unit_col)
    for unit in sorted(set(model_totals) | set(qex_totals)):
        mt = float(model_totals.get(unit, 0.0))
        qt = float(qex_totals.get(unit, 0.0))
        diff = abs(mt - qt)
        denom = max(abs(mt), abs(qt), 1.0)
        rel = diff / denom
        if diff > 0.01 and rel > 0.01:
            findings.append(issue(
                'TOTAL_QTY_BY_UNIT_MISMATCH', 'T2', 'ERROR', 'AGGREGATE', f'Total[{unit}]', 'RBR-Quantity',
                f'Total quantity for unit "{unit}" must match between model and QEX',
                f'Model={mt:.4g} QEX={qt:.4g} ({rel:.1%} diff)',
                f'Total quantity for unit "{unit}" differs: model={mt:.4g} vs QEX={qt:.4g} ({rel:.1%} deviation).',
                'QEX total quantity by unit comparison'
            ))


def run_audit(model_path, refs, discipline='AUTO_MIXED', qex_path=None, qto_path=None, quantity_paths=None):
    model_df, model_meta = read_table_with_meta(model_path)
    findings = []
    limitations = []
    col_mappings = pd.DataFrame()
    if model_df is None or model_df.empty:
        limitations.append({'Severity': 'CRITICAL ERROR', 'Limitation': 'Model extract could not be read or is empty.', 'File': str(model_path)})
        return pd.DataFrame(findings), pd.DataFrame(limitations), pd.DataFrame(), model_df, {}, model_meta, col_mappings
    cols = resolve_cols(model_df, MODEL_COLS)
    col_mappings = _build_col_mapping_doc(cols)
    missing_core = [k for k in ['object_id', 'type_number', 'pr_code', 'occ', 'qty', 'unit'] if not cols.get(k)]
    if missing_core:
        limitations.append({'Severity': 'LIMITATION', 'Limitation': 'Some core model columns were not detected; related checks were skipped or downgraded.', 'File': str(model_path), 'Missing': ', '.join(missing_core)})

    _structural_validation(model_df, cols, model_meta, findings, limitations)
    _dataset_sanity(model_df, cols, findings, limitations)
    _object_id_checks(model_df, cols, refs, discipline, findings)
    _type_checks(model_df, cols, findings)
    _classification_checks(model_df, cols, refs, findings)
    _hierarchy_checks(model_df, cols, findings)
    _quantity_value_checks(model_df, cols, findings)
    _loi_attribute_checks(model_df, cols, refs, findings)
    _additional_attribute_sanity(model_df, cols, findings)
    _systemic_empty_key_attributes(model_df, cols, findings)

    # Resolve best QEX path for element-level checks
    qex_path_el = qex_path
    if not qex_path_el:
        for p in quantity_paths or []:
            if p and 'qex' in Path(p).name.lower():
                qex_path_el = p
                break
    qex_df = None; qex_cols_r: dict = {}
    if qex_path_el:
        qex_df, _ = read_table_with_meta(qex_path_el)
        if qex_df is not None and not qex_df.empty:
            qex_cols_r = resolve_cols(qex_df, QTY_COLS)

    if qex_df is not None and not qex_df.empty:
        _element_count_check(model_df, cols, qex_df, qex_cols_r, findings, limitations)
        _element_level_quantity_check(model_df, cols, qex_df, qex_cols_r, findings, limitations)
        _total_quantity_check_by_unit(model_df, cols, qex_df, qex_cols_r, findings)

    qpaths = []
    if qex_path: qpaths.append(qex_path)
    if qto_path: qpaths.append(qto_path)
    for p in quantity_paths or []:
        if p and p not in qpaths:
            qpaths.append(p)
    qc = quantity_control(model_df, cols, qpaths)
    quantity_findings_from_control(qc, findings)
    findings = _compress_row_level_findings(findings)

    return pd.DataFrame(findings), pd.DataFrame(limitations), qc, model_df, cols, model_meta, col_mappings
