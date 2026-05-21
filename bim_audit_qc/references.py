from __future__ import annotations
from pathlib import Path
import re, warnings
from collections import defaultdict
import pandas as pd
from .io_utils import as_text, is_empty

UNICLASS_RE = re.compile(r'\b(?:Pr|Ss|En|EF|SL|TE|PM|Zz)_[0-9A-Z]{2}(?:_[0-9A-Z]{2}){1,6}\b')
OBJ_PREFIX_RE = re.compile(r'\b[A-Z]{2,4}-[A-Z0-9]{2,5}-[A-Z0-9]{2,5}\b')


def _is_x(v: str) -> bool:
    return as_text(v).strip().upper() in {'X','X*','YES','Y','1','TRUE','REQUIRED','MANDATORY'}


class ReferenceLibrary:
    """Loads permanent and temporary reference/control spreadsheets on every audit run.

    v7 additions:
    - parses LOI matrices by discipline *and object-type column*, not only by discipline;
    - keeps common required attributes separately from type-specific required attributes;
    - scans all readable reference files, including project-specific tables dropped into references/.
    """
    def __init__(self, ref_dir):
        if isinstance(ref_dir, (list, tuple, set)):
            self.ref_dirs = [Path(p) for p in ref_dir]
            self.ref_dir = self.ref_dirs[0] if self.ref_dirs else Path('.')
        else:
            self.ref_dirs = [Path(ref_dir)]
            self.ref_dir = Path(ref_dir)
        self.files = []
        self.unreadable_files = []
        self.valid_pr_codes = set()
        self.valid_obj_prefixes = set()
        self.valid_disciplines = set()
        self.valid_occ_pr_prefix = set()
        # Legacy/simple API
        self.loi_required_attrs = defaultdict(set)
        # v7 granular LOI API
        self.loi_common_required = defaultdict(set)                 # disc -> attrs required for all object types
        self.loi_required_by_type = defaultdict(lambda: defaultdict(set))  # disc -> object type -> attrs
        self.loi_object_types = defaultdict(list)                   # disc -> object types found in sheet
        self.load()

    def _add_obj_prefix(self, value: str):
        value = as_text(value).upper()
        for p in OBJ_PREFIX_RE.findall(value):
            self.valid_obj_prefixes.add(p)
            self.valid_disciplines.add(p.split('-')[0])

    def _scan_values(self, df: pd.DataFrame, max_cells: int = 100000):
        if df is None or df.empty:
            return
        vals = df.astype(str).fillna('').values.ravel()[:max_cells]
        for val in vals:
            val = as_text(val)
            if not val:
                continue
            for code in UNICLASS_RE.findall(val):
                self.valid_pr_codes.add(code)
            self._add_obj_prefix(val)

    def _parse_occ_control_raw(self, raw: pd.DataFrame):
        if raw is None or raw.empty:
            return
        header_i = None
        for i in range(min(len(raw), 30)):
            rowtxt = ' '.join(as_text(v).lower() for v in raw.iloc[i].tolist())
            if 'obj-id prefix' in rowtxt and 'pr_code' in rowtxt and 'rbr-occ' in rowtxt:
                header_i = i
                break
        if header_i is None:
            return
        headers = [as_text(v).lower() for v in raw.iloc[header_i].tolist()]
        def find_header(term):
            for j,h in enumerate(headers):
                if term in h:
                    return j
            return None
        obj_j = find_header('obj-id prefix')
        pr_j = find_header('pr_code')
        occ_j = find_header('rbr-occ')
        if obj_j is None or pr_j is None or occ_j is None:
            return
        for i in range(header_i+1, len(raw)):
            prefix = as_text(raw.iat[i, obj_j]).upper()
            pr = as_text(raw.iat[i, pr_j])
            occ = as_text(raw.iat[i, occ_j])
            if prefix and pr and occ and not is_empty(prefix) and not is_empty(pr) and not is_empty(occ):
                self._add_obj_prefix(prefix)
                self.valid_pr_codes.add(pr)
                self.valid_occ_pr_prefix.add((prefix, pr, occ))

    def _extract_loi_from_raw_sheet(self, sheet_name: str, raw: pd.DataFrame):
        """Parse RB LOI sheets with object-type applicability columns.

        Typical structure:
        row 3: object type headers ... Attribute ... LOI
        row 4: Type ... 200 300 400 500
        rows below: X/o applicability, attribute name, LOI marks.
        """
        disc = sheet_name.split('_')[0].strip().upper()
        if not disc or len(disc) > 6 or raw is None or raw.empty:
            return
        rows, cols = raw.shape
        attr_col = None
        loi400_col = None
        header_row = None
        # detect Attribute column and LOI 400 column independently
        for r in range(min(rows, 25)):
            for c in range(cols):
                val = as_text(raw.iat[r, c]).strip().lower()
                if val == 'attribute':
                    attr_col = c
                    header_row = r
                if val in {'400','400.0','loi 400','loi400'}:
                    loi400_col = c
        if attr_col is None or loi400_col is None:
            return

        # In several RB LOI matrices, the visible 'Attribute' label sits one column left of
        # the actual attribute values (because of merged header cells). Detect and shift.
        attr_label_col = attr_col
        def rbr_count(col):
            if col is None or col >= cols:
                return 0
            return sum(1 for rr in range(header_row + 1 if header_row is not None else 0, rows) if as_text(raw.iat[rr, col]).startswith('RBR-'))
        if rbr_count(attr_col) == 0 and rbr_count(attr_col + 1) > 0:
            attr_col = attr_col + 1

        # object type columns are left of Attribute label column and have a non-empty type name in/near header row
        type_cols = []
        # Prefer the same header row where Attribute was found; fall back to first 5 rows.
        for c in range(0, attr_label_col):
            name = ''
            for rr in [header_row, header_row-1 if header_row else None, header_row+1 if header_row is not None and header_row+1 < rows else None, 2, 3]:
                if rr is None or rr < 0 or rr >= rows:
                    continue
                candidate = as_text(raw.iat[rr, c]).strip()
                if candidate and candidate.lower() not in {'type', 'should be included', 'loi'}:
                    name = candidate
                    break
            if name:
                type_cols.append((c, name))
        if type_cols:
            self.loi_object_types[disc] = [n for _, n in type_cols]

        for r in range(rows):
            attr = as_text(raw.iat[r, attr_col])
            if not attr.startswith('RBR-'):
                continue
            mark400 = as_text(raw.iat[r, loi400_col])
            if not _is_x(mark400):
                continue
            self.loi_required_attrs[disc].add(attr)
            # granular applicability
            applied_types = []
            for c, typ in type_cols:
                mark = as_text(raw.iat[r, c])
                if _is_x(mark):
                    self.loi_required_by_type[disc][typ].add(attr)
                    applied_types.append(typ)
            if type_cols and len(applied_types) == len(type_cols):
                self.loi_common_required[disc].add(attr)
            elif not type_cols:
                self.loi_common_required[disc].add(attr)

    def load(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            all_paths = []
            for ref_dir in self.ref_dirs:
                if ref_dir.exists():
                    all_paths.extend(sorted(ref_dir.glob('*')))
            for path in all_paths:
                if path.name.startswith('~$') or path.suffix.lower() not in {'.xlsx','.xlsm','.xls','.csv','.txt'}:
                    continue
                lname = path.name.lower()
                self.files.append(path.name)
                try:
                    if path.suffix.lower() in {'.csv','.txt'}:
                        try:
                            df = pd.read_csv(path, dtype=str, nrows=10000, encoding='utf-8-sig')
                        except Exception:
                            df = pd.DataFrame()
                        self._scan_values(df)
                        continue
                    xls = pd.ExcelFile(path)
                    # v7 performance: do not exhaustively scan DS/file-naming workbooks. They are useful as traceable references,
                    # but their many artificial filename examples can create thousands of false Object_ID-like prefixes and slow loading.
                    is_naming = ('file_naming' in lname or 'naming_conventions' in lname or 'ds1design' in lname or 'ds2design' in lname or 'ds3design' in lname)
                    is_loi = 'loi' in lname
                    is_obj = ('objectid' in lname or 'typenr' in lname or 'object_id' in lname or 'obj_id' in lname)
                    is_code = ('uniclass' in lname or 'pbs' in lname or 'payitem' in lname or 'pr_code' in lname or 'kontrolltabel' in lname or 'occ' in lname)
                    if is_naming and not (is_loi or is_obj or is_code):
                        continue
                    sheet_limit = 30 if (is_loi or is_obj) else 8
                    row_limit = 700 if (is_loi or is_obj) else 5000
                    for sheet in xls.sheet_names[:sheet_limit]:
                        try:
                            raw = pd.read_excel(path, sheet_name=sheet, dtype=str, header=None, nrows=row_limit)
                        except Exception:
                            continue
                        raw = raw.iloc[:, :100]
                        if is_code or is_obj:
                            self._scan_values(raw, max_cells=60000)
                        if 'obj_id-pr_code-occ' in lname or 'kontrolltabel' in lname:
                            self._parse_occ_control_raw(raw)
                        if is_loi:
                            self._extract_loi_from_raw_sheet(sheet, raw)
                except Exception:
                    self.unreadable_files.append(path.name)

    def closest_prefixes(self, prefix: str, limit: int = 5) -> list[str]:
        prefix = as_text(prefix).upper()
        if not prefix or not self.valid_obj_prefixes:
            return []
        try:
            from rapidfuzz import process, fuzz
            return [m[0] for m in process.extract(prefix, sorted(self.valid_obj_prefixes), scorer=fuzz.WRatio, limit=limit)]
        except Exception:
            return sorted([p for p in self.valid_obj_prefixes if p.startswith(prefix[:3])])[:limit]

    def required_attrs_for_disc(self, disc: str) -> set[str]:
        return self.loi_required_attrs.get(as_text(disc).upper(), set())

    def required_attrs_for_row(self, disc: str, object_type: str | None = None) -> set[str]:
        disc = as_text(disc).upper()
        required = set(self.loi_common_required.get(disc, set()))
        if object_type:
            required |= set(self.loi_required_by_type.get(disc, {}).get(object_type, set()))
        return required

    def stats(self):
        typed = {d: len(types) for d, types in self.loi_object_types.items()}
        return {
            'reference_files_loaded': len(self.files),
            'unreadable_reference_files': len(self.unreadable_files),
            'valid_pr_codes': len(self.valid_pr_codes),
            'valid_object_id_prefixes': len(self.valid_obj_prefixes),
            'valid_disciplines_detected': ', '.join(sorted(self.valid_disciplines))[:250],
            'valid_occ_pr_prefix_combos': len(self.valid_occ_pr_prefix),
            'loi_disciplines_loaded': ', '.join(sorted(self.loi_required_attrs.keys()))[:250],
            'loi_object_type_counts': str(typed)[:500],
        }
