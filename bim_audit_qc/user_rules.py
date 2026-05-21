from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

from .io_utils import as_text


def _truthy(v: str) -> bool:
    return as_text(v).strip().lower() in {'1','true','yes','y','x'}


def _pattern_match(pattern: str, value: str) -> bool:
    pattern = as_text(pattern).strip()
    value = as_text(value)
    if not pattern:
        return True
    # allow plain wildcard patterns and regex patterns prefixed with re:
    try:
        if pattern.lower().startswith('re:'):
            return re.search(pattern[3:], value, flags=re.IGNORECASE) is not None
        # wildcard: STR-WTP-* or *WTP*
        rx = '^' + re.escape(pattern).replace('\\*', '.*') + '$'
        return re.search(rx, value, flags=re.IGNORECASE) is not None
    except Exception:
        return pattern.lower() in value.lower()


class UserRules:
    """Controlled learning layer.

    The app never rewrites code. Instead, user-confirmed corrections are stored as
    spreadsheet rules in project_rules/. These rules are loaded on every audit and
    can suppress false positives, downgrade uncertain flags, or enforce project-specific
    decisions in a traceable way.
    """
    DEFAULT_COLUMNS = [
        'enabled', 'rule_name', 'scope', 'rule_id', 'attribute', 'objectid_pattern',
        'prefix_pattern', 'model_value_pattern', 'action', 'final_status',
        'reasoning_summary', 'crs_comment', 'cross_check'
    ]

    def __init__(self, rules_dir: str | Path):
        self.rules_dir = Path(rules_dir)
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.rules = pd.DataFrame(columns=self.DEFAULT_COLUMNS)
        self.files = []
        self.unreadable_files = []
        self.load()

    def load(self):
        frames = []
        for path in sorted(self.rules_dir.glob('*')):
            if path.name.startswith('~$') or path.suffix.lower() not in {'.xlsx','.xlsm','.xls','.csv'}:
                continue
            try:
                if path.suffix.lower() == '.csv':
                    df = pd.read_csv(path, dtype=str).fillna('')
                else:
                    df = pd.read_excel(path, dtype=str).fillna('')
                if df.empty:
                    continue
                df.columns = [as_text(c).strip() for c in df.columns]
                for c in self.DEFAULT_COLUMNS:
                    if c not in df.columns:
                        df[c] = ''
                df = df[self.DEFAULT_COLUMNS]
                df['_source_rule_file'] = path.name
                frames.append(df)
                self.files.append(path.name)
            except Exception:
                self.unreadable_files.append(path.name)
        if frames:
            self.rules = pd.concat(frames, ignore_index=True).fillna('')
            if 'enabled' in self.rules.columns:
                self.rules = self.rules[self.rules['enabled'].apply(lambda v: _truthy(v) or as_text(v).strip() == '')].copy()

    def stats(self):
        return {
            'rule_files_loaded': len(self.files),
            'rules_loaded': int(len(self.rules)),
            'unreadable_rule_files': len(self.unreadable_files),
            'rule_files': ', '.join(self.files)[:500],
        }

    def _rule_matches(self, rule: pd.Series, row: pd.Series) -> bool:
        # Empty fields mean wildcard / no condition.
        if as_text(rule.get('rule_id')) and as_text(rule.get('rule_id')) != as_text(row.get('rule_id')):
            return False
        if as_text(rule.get('attribute')) and as_text(rule.get('attribute')) != as_text(row.get('attribute')):
            return False
        if not _pattern_match(rule.get('objectid_pattern',''), row.get('element','')):
            return False
        # Prefix is first three Object_ID segments when available.
        element = as_text(row.get('element',''))
        prefix = '-'.join(element.split('-')[:3]) if '-' in element else ''
        if not _pattern_match(rule.get('prefix_pattern',''), prefix):
            return False
        if not _pattern_match(rule.get('model_value_pattern',''), row.get('model_value','')):
            return False
        return True

    def apply(self, final: pd.DataFrame) -> pd.DataFrame:
        if final is None or final.empty or self.rules.empty:
            return final
        out = final.copy()
        if 'learning_rule_applied' not in out.columns:
            out['learning_rule_applied'] = ''
        for idx, row in out.iterrows():
            for _, rule in self.rules.iterrows():
                if not self._rule_matches(rule, row):
                    continue
                action = as_text(rule.get('action')).upper() or as_text(row.get('action')) or 'KEEP'
                status = as_text(rule.get('final_status')).upper() or as_text(row.get('final_status'))
                if action in {'KEEP','MODIFY','SUPPRESS'}:
                    out.at[idx, 'action'] = action
                if status in {'OK','WARNING','ERROR','SYSTEMIC ERROR','CRITICAL ERROR','LIMITATION','MANUAL_REVIEW','INFO'}:
                    out.at[idx, 'final_status'] = status
                if as_text(rule.get('reasoning_summary')):
                    out.at[idx, 'reasoning_summary'] = as_text(rule.get('reasoning_summary'))
                if as_text(rule.get('crs_comment')):
                    out.at[idx, 'crs_comment'] = as_text(rule.get('crs_comment'))
                out.at[idx, 'learning_rule_applied'] = as_text(rule.get('rule_name')) or as_text(rule.get('_source_rule_file'))
                break
        return out


def create_templates(rules_dir: str | Path):
    rules_dir = Path(rules_dir)
    rules_dir.mkdir(parents=True, exist_ok=True)
    template = rules_dir / 'false_positive_rules_TEMPLATE.xlsx'
    if not template.exists():
        df = pd.DataFrame([
            {
                'enabled': 'FALSE',
                'rule_name': 'Example: suppress accepted WTP prefix warning',
                'scope': 'GLOBAL or project code',
                'rule_id': 'OBJ_PREFIX_NOT_IN_MATRIX',
                'attribute': 'RBR-Object_ID',
                'objectid_pattern': 'STR-WTP-000-*',
                'prefix_pattern': 'STR-WTP-000',
                'model_value_pattern': '',
                'action': 'SUPPRESS',
                'final_status': 'INFO',
                'reasoning_summary': 'User-confirmed project-specific exception. This pattern is accepted for this project.',
                'crs_comment': 'Suppressed by user-confirmed rule.',
                'cross_check': 'project_rules/false_positive_rules_TEMPLATE.xlsx'
            }
        ])
        df.to_excel(template, index=False)
    accepted = rules_dir / 'accepted_patterns_TEMPLATE.xlsx'
    if not accepted.exists():
        df = pd.DataFrame([
            {
                'enabled': 'FALSE',
                'rule_name': 'Example: downgrade accepted zero quantity for specific code',
                'scope': 'GLOBAL or project code',
                'rule_id': 'QTY_ZERO',
                'attribute': 'RBR-Quantity',
                'objectid_pattern': '',
                'prefix_pattern': '',
                'model_value_pattern': '0',
                'action': 'MODIFY',
                'final_status': 'WARNING',
                'reasoning_summary': 'Zero value is not automatically an error; verify if justified in project context.',
                'crs_comment': 'Mistake: Quantity is zero. Explanation: Verify whether the zero quantity is intentional and justified. Cross-check: project-specific accepted pattern.',
                'cross_check': 'project_rules/accepted_patterns_TEMPLATE.xlsx'
            }
        ])
        df.to_excel(accepted, index=False)
