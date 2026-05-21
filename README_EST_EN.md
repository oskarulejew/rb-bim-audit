# RB BIM Audit AI WebApp v7 — Forensic Logic Upgrade

This version improves the audit logic based on comparison between the generated WebApp report and the AI-only forensic report.

Main changes:
- AUTO_MIXED discipline logic: package-level `RBR-Discipline_Code` is no longer treated as a row-level Object_ID prefix mismatch when the model contains mixed object disciplines.
- Stronger LOI parsing: LOI matrices are parsed by discipline and object-type columns, not only by discipline.
- Row-level LOI findings: missing mandatory attributes are written per affected element/attribute so comments can be placed in the exact cells.
- Type-specific STR logic: `RBR-H1`, `RBR-Q1`, `RBR-V1` are applied to inferred STR main structural elements, not blindly to all STR rows.
- Key attribute systemic check: empty `RBR-Route_code` and `RBR-Phase_Demolished` are detected as systemic completeness issues when mostly empty.
- Numeric hygiene check: floating-point export artifacts like `14.850000000000001` are flagged as warnings.
- Merge bug fixed: AI/review decisions are matched by `rule_id + row_index + element + attribute`, preventing duplicated report rows.
- Reference loading optimized: DS naming workbooks are kept as traceable references but no longer exhaustively scanned as Object_ID prefix sources.

Run:
```zsh
cd ~/Downloads/RB_BIM_Audit_AI_WebApp
chmod +x *.command
./install.command
./start.command
```

Open: http://localhost:8501


## v8 logic upgrade
- Compresses repeated row-level findings into systemic findings with affected row references.
- Does not repeat missing LOI columns for every element; reports one systemic issue per attribute/context.
- Detects near 2x/0.5x quantity patterns as systemic quantity errors.
- Keeps AUTO_MIXED discipline handling for extracts with multiple Object_ID prefixes.
