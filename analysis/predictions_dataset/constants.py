"""Shared constants for prediction-dataset construction and analysis."""

MIN_PRED_AREA_FRACTION = 0.0003

# Version of the predictions-dataset schema (column names, order, and field
# definitions). Bump whenever a column or definition changes.
# v2: fp counts credit every criterion-matching component, so fragmentation of
# a detected mass is no longer counted as a false positive.
SCHEMA_VERSION = 2
