"""Result-processing stage: turn stored ``Results/`` into figures and tables.

Modules
-------
record_io        Results/ path schema + CSV / manifest read & write
ingest_results   ingest pre-validated per-task CSVs (Autorater) into the schema
render           shared forest-plot renderer (one style for the whole paper)
plot_forest      CLI: Results/ -> Plots/ (full + preview; --repro reproduces the paper figs)
summarize_tables CLI: Results/ -> coverage / width / decomposition tables
"""
