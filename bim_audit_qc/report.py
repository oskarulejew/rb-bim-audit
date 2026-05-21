from __future__ import annotations
from pathlib import Path
import pandas as pd
from .io_utils import as_text, is_empty, find_col

ERROR_STATUSES = {'ERROR','CRITICAL ERROR','SYSTEMIC ERROR'}
WARN_STATUSES = {'WARNING','MANUAL_REVIEW','LIMITATION'}


def merge_final(flags: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if flags is None or flags.empty:
        return pd.DataFrame()
    f = flags.copy()
    if reviews is None or reviews.empty:
        f['final_status'] = f['severity']
        f['action'] = 'KEEP'
        f['confidence'] = 0.7
        f['crs_comment'] = f['message']
        f['reasoning_summary'] = ''
        return f
    key_cols = ['rule_id','row_index','element','attribute']
    for c in key_cols:
        f[c] = f[c].astype(str); reviews[c] = reviews[c].astype(str)
    m = f.merge(reviews, on=key_cols, how='left', suffixes=('','_ai'))
    m['final_status'] = m['final_status'].fillna(m['severity'])
    m['action'] = m['action'].fillna('KEEP')
    m['crs_comment'] = m['crs_comment'].fillna(m['message'])
    m['reasoning_summary'] = m['reasoning_summary'].fillna('')
    m['confidence'] = m['confidence'].fillna(0.7)
    return m


def final_decision(final_kept: pd.DataFrame, limitations: pd.DataFrame | None = None, qty_control: pd.DataFrame | None = None) -> str:
    if final_kept is None or final_kept.empty:
        return 'Model suitable for submission'
    statuses = set(final_kept.get('final_status', pd.Series(dtype=str)).astype(str))
    critical_count = int(final_kept['final_status'].astype(str).isin(['CRITICAL ERROR','SYSTEMIC ERROR']).sum()) if 'final_status' in final_kept else 0
    error_count = int(final_kept['final_status'].astype(str).eq('ERROR').sum()) if 'final_status' in final_kept else 0
    limitation_count = int(final_kept['final_status'].astype(str).eq('LIMITATION').sum()) if 'final_status' in final_kept else 0
    qty_err = 0
    if qty_control is not None and not qty_control.empty and 'Status' in qty_control:
        qty_err = int(qty_control['Status'].astype(str).isin(['ERROR','SYSTEMIC ERROR','CRITICAL ERROR']).sum())
    if critical_count or error_count or qty_err:
        return 'Model not suitable for submission'
    if limitation_count or any(s in statuses for s in ['WARNING','MANUAL_REVIEW']):
        return 'Model conditionally suitable for submission'
    return 'Model suitable for submission'


def _errors_filterable(final_kept: pd.DataFrame, decision: str) -> pd.DataFrame:
    cols = ['Source_Row','ObjectID','Element_Class','Attribute','Model_Value','Requirement','Severity','Explanation','Cross-check']
    rows = []
    rows.append({
        'Source_Row': 'FINAL_DECISION', 'ObjectID': '', 'Element_Class': '', 'Attribute': 'FINAL_DECISION',
        'Model_Value': decision, 'Requirement': 'Final audit decision must be one of the allowed statements',
        'Severity': 'INFO', 'Explanation': decision, 'Cross-check': 'Overall audit result based on kept findings and quantity control'
    })
    if final_kept is not None and not final_kept.empty:
        for _, r in final_kept.iterrows():
            rows.append({
                'Source_Row': r.get('row_index',''),
                'ObjectID': r.get('element',''),
                'Element_Class': r.get('tier',''),
                'Attribute': r.get('attribute',''),
                'Model_Value': r.get('model_value',''),
                'Requirement': r.get('requirement',''),
                'Severity': r.get('final_status', r.get('severity','')),
                'Explanation': ((r.get('reasoning_summary') or r.get('message','')) + (f" | User rule applied: {r.get('learning_rule_applied')}" if r.get('learning_rule_applied') else '')),
                'Cross-check': r.get('cross_check','') or 'Rail Baltica BIM references loaded in the audit tool',
            })
    return pd.DataFrame(rows, columns=cols)


def _find_attribute_col(model_df: pd.DataFrame, attr: str):
    if model_df is None or model_df.empty:
        return None
    attr = as_text(attr)
    # combined or virtual attributes cannot be commented in one source cell
    if not attr or '/' in attr or attr in {'Dataset row','Quantity_Control','FINAL_DECISION'}:
        return None
    return find_col(model_df, [attr])


def _apply_model_comments(ws, workbook, model_df: pd.DataFrame, final_kept: pd.DataFrame):
    if model_df is None or model_df.empty or final_kept is None or final_kept.empty:
        return
    red_fmt = workbook.add_format({'bg_color': '#F4CCCC'})
    yellow_fmt = workbook.add_format({'bg_color': '#FFF2CC'})
    # write comment to erroneous source cell only when row and attribute resolve
    for _, r in final_kept.iterrows():
        row_raw = as_text(r.get('row_index',''))
        # row_index may be a single row ("42") or a compressed list ("2,3,4,...").
        row_nums = []
        for token in row_raw.split(','):
            token = token.strip()
            if token.isdigit():
                row_nums.append(int(token))
        if not row_nums:
            continue
        attr = as_text(r.get('attribute',''))
        colname = _find_attribute_col(model_df, attr)
        if not colname or colname not in model_df.columns:
            continue
        xcol = list(model_df.columns).index(colname)
        status = as_text(r.get('final_status',''))
        fmt = red_fmt if status in ERROR_STATUSES else yellow_fmt
        comment = as_text(r.get('crs_comment','')) or as_text(r.get('message',''))
        # Avoid creating thousands of comments for one systemic compressed issue.
        for excel_row_1based in row_nums[:250]:
            xrow = excel_row_1based - 1
            try:
                ws.write(xrow, xcol, model_df.iloc[excel_row_1based-2, xcol], fmt)
                if comment:
                    ws.write_comment(xrow, xcol, comment[:32000], {'visible': False, 'x_scale': 1.8, 'y_scale': 1.6})
            except Exception:
                continue


def _write_dataset_sanity_sheet(writer, detected, col_mappings, ref_stats, limitations):
    wb = writer.book
    section_fmt = wb.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': '#FFFFFF', 'border': 1})
    hdr_fmt = wb.add_format({'bold': True, 'bg_color': '#D9EAF7', 'border': 1})
    ok_fmt = wb.add_format({'bg_color': '#D9EAD3'})
    err_fmt = wb.add_format({'bg_color': '#F4CCCC'})

    ws = wb.add_worksheet('Dataset_Sanity')
    writer.sheets['Dataset_Sanity'] = ws
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 24)
    ws.set_column(2, 2, 32)
    ws.set_column(3, 3, 16)
    ws.set_column(4, 4, 56)
    ws.set_column(5, 5, 14)
    ws.set_column(6, 6, 18)

    row = 0
    ws.write(row, 0, 'FILE INVENTORY', section_fmt)
    row += 1
    for ci, h in enumerate(['File', 'Role', 'Score', 'Reason', 'Columns_Found', 'Header_Row', 'Sheet']):
        ws.write(row, ci, h, hdr_fmt)
    row += 1
    if detected is not None and not (isinstance(detected, pd.DataFrame) and detected.empty):
        det_df = detected if isinstance(detected, pd.DataFrame) else pd.DataFrame(detected)
        for _, dr in det_df.iterrows():
            ws.write(row, 0, str(dr.get('file', '')))
            ws.write(row, 1, str(dr.get('role', '')))
            ws.write(row, 2, str(dr.get('score', '')))
            ws.write(row, 3, str(dr.get('reason', '')))
            ws.write(row, 4, str(dr.get('columns_found', ''))[:120])
            ws.write(row, 5, str(dr.get('header_row', '')))
            ws.write(row, 6, str(dr.get('sheet', '')))
            row += 1
    row += 1

    ws.write(row, 0, 'COLUMN MAPPING', section_fmt)
    row += 1
    for ci, h in enumerate(['Standard_Key', 'RBR_Canonical', 'Detected_Column', 'Status', 'Note']):
        ws.write(row, ci, h, hdr_fmt)
    row += 1
    if col_mappings is not None and isinstance(col_mappings, pd.DataFrame) and not col_mappings.empty:
        for _, cm in col_mappings.iterrows():
            st = str(cm.get('Status', ''))
            fmt = ok_fmt if st == 'OK' else (err_fmt if st == 'MISSING' else None)
            ws.write(row, 0, str(cm.get('Standard_Key', '')), fmt)
            ws.write(row, 1, str(cm.get('RBR_Canonical', '')), fmt)
            ws.write(row, 2, str(cm.get('Detected_Column', '')), fmt)
            ws.write(row, 3, st, fmt)
            ws.write(row, 4, str(cm.get('Note', '')), fmt)
            row += 1
    row += 1

    ws.write(row, 0, 'REFERENCE LOADING STATS', section_fmt)
    row += 1
    ws.write(row, 0, 'Key', hdr_fmt); ws.write(row, 1, 'Value', hdr_fmt)
    row += 1
    for k, v in (ref_stats or {}).items():
        ws.write(row, 0, str(k)); ws.write(row, 1, str(v))
        row += 1
    row += 1

    ws.write(row, 0, 'DATASET LIMITATIONS', section_fmt)
    row += 1
    ws.write(row, 0, 'Severity', hdr_fmt); ws.write(row, 1, 'Limitation', hdr_fmt); ws.write(row, 2, 'Cross-check', hdr_fmt)
    row += 1
    if limitations is not None:
        lim_df = limitations if isinstance(limitations, pd.DataFrame) else pd.DataFrame(limitations if limitations else [])
        if not lim_df.empty:
            for _, lr in lim_df.iterrows():
                ws.write(row, 0, str(lr.get('Severity', '')))
                ws.write(row, 1, str(lr.get('Limitation', '')))
                ws.write(row, 2, str(lr.get('Cross-check', lr.get('File', ''))))
                row += 1


def write_report(out_path, flags, reviews, final, limitations, detected, ref_stats, qty_control, model_df=None, model_cols=None, model_meta=None, col_mappings=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_kept = final[final['action']!='SUPPRESS'].copy() if final is not None and not final.empty else pd.DataFrame()
    decision = final_decision(final_kept, limitations, qty_control)
    errors = _errors_filterable(final_kept, decision)
    # Ensure exact columns for Quantity_Control
    qty_cols = ['Pr_Code','Type_number','Unit','Model_Qty','QEX_Qty','QTO_Qty','Model-QEX_Diff','Model-QTO_Diff','Status','Comment']
    if qty_control is None or qty_control.empty:
        qty_control = pd.DataFrame(columns=qty_cols)
    for c in qty_cols:
        if c not in qty_control.columns:
            qty_control[c] = ''
    qty_control = qty_control[qty_cols]
    if model_df is None:
        model_df = pd.DataFrame()

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        model_df.to_excel(writer, sheet_name='Model_Original', index=False)
        model_df.to_excel(writer, sheet_name='Model_With_Errors', index=False)
        errors.to_excel(writer, sheet_name='Errors_Filterable', index=False)
        qty_control.to_excel(writer, sheet_name='Quantity_Control', index=False)

        wb = writer.book
        header_fmt = wb.add_format({'bold': True, 'bg_color': '#D9EAF7', 'border': 1, 'text_wrap': True})
        red_fmt = wb.add_format({'bg_color': '#F4CCCC'})
        yellow_fmt = wb.add_format({'bg_color': '#FFF2CC'})
        green_fmt = wb.add_format({'bg_color': '#D9EAD3'})
        grey_fmt = wb.add_format({'bg_color': '#E7E6E6'})
        text_fmt = wb.add_format({'text_wrap': True, 'valign': 'top'})
        num_fmt = wb.add_format({'num_format': '0.000', 'valign': 'top'})

        for sname, ws in writer.sheets.items():
            ws.freeze_panes(1, 0)
            max_col = 25 if sname.startswith('Model') else 12
            try:
                ws.autofilter(0, 0, max(1, len(model_df) if sname.startswith('Model') else 1), max_col)
            except Exception:
                pass
            ws.set_row(0, 28, header_fmt)
            if sname.startswith('Model'):
                ws.set_column(0, min(max_col, max(0, len(model_df.columns)-1)), 18, text_fmt)
            elif sname == 'Errors_Filterable':
                ws.set_column(0, 0, 14, text_fmt)
                ws.set_column(1, 1, 24, text_fmt)
                ws.set_column(2, 2, 14, text_fmt)
                ws.set_column(3, 5, 24, text_fmt)
                ws.set_column(6, 6, 18, text_fmt)
                ws.set_column(7, 8, 58, text_fmt)
            elif sname == 'Quantity_Control':
                ws.set_column(0, 2, 22, text_fmt)
                ws.set_column(3, 7, 16, num_fmt)
                ws.set_column(8, 8, 16, text_fmt)
                ws.set_column(9, 9, 58, text_fmt)

        # Conditional formatting
        err_ws = writer.sheets['Errors_Filterable']
        err_ws.conditional_format(1, 0, max(1, len(errors)), 8, {'type':'text','criteria':'containing','value':'CRITICAL ERROR','format': red_fmt})
        err_ws.conditional_format(1, 0, max(1, len(errors)), 8, {'type':'text','criteria':'containing','value':'SYSTEMIC ERROR','format': red_fmt})
        err_ws.conditional_format(1, 0, max(1, len(errors)), 8, {'type':'text','criteria':'containing','value':'ERROR','format': red_fmt})
        err_ws.conditional_format(1, 0, max(1, len(errors)), 8, {'type':'text','criteria':'containing','value':'WARNING','format': yellow_fmt})
        err_ws.conditional_format(1, 0, max(1, len(errors)), 8, {'type':'text','criteria':'containing','value':'LIMITATION','format': yellow_fmt})
        err_ws.conditional_format(1, 0, max(1, len(errors)), 8, {'type':'text','criteria':'containing','value':'INFO','format': grey_fmt})

        qty_ws = writer.sheets['Quantity_Control']
        qty_ws.conditional_format(1, 0, max(1, len(qty_control)), 9, {'type':'text','criteria':'containing','value':'ERROR','format': red_fmt})
        qty_ws.conditional_format(1, 0, max(1, len(qty_control)), 9, {'type':'text','criteria':'containing','value':'LIMITATION','format': yellow_fmt})
        qty_ws.conditional_format(1, 0, max(1, len(qty_control)), 9, {'type':'text','criteria':'containing','value':'OK','format': green_fmt})

        _apply_model_comments(writer.sheets['Model_With_Errors'], wb, model_df, final_kept)
        _write_dataset_sanity_sheet(writer, detected, col_mappings, ref_stats, limitations)

    return out_path
