# Data Preparation

This document describes the data preparation steps performed after the data
integration stage. The script implementing these steps is
`scripts/data_preparation_final.py`. Running that script reads the integrated
CSV at `data/processed/integrated_tv_shows.csv`, performs cleaning and
transformations, performs duplicate removal (by title + release year), and
writes two artifacts:

- `data/processed/cleaned_tv_shows.csv`  (cleaned dataset)
- `data/processed/data_preparation_report.txt`  (plain-text report of actions)
- PostgreSQL table: `integrated_tv_shows_cleaned`


Overview of the main processing performed
----------------------------------------

1) Initial inspection and null reporting
	 - A null-count summary is produced for all columns.
	 - Example rows (up to 3 per important column) where nulls appear are
		 included in the report to show real cases.

2) Transformations applied
	 - title_normalized: created by lowercasing and removing common country
		 markers such as `(u.s.)`, `(uk)`, `(us)` and normalizing whitespace.
	 - release_year: converted to nullable integer (`Int64`) when possible.
	 - premiere_date: parsed to `datetime` (invalid parses become `NaT`).
	 - rating_avg: numeric extraction from the `rating` column (which may be a
		 string representation like `{'average': 7.5}`); non-parsable values become
		 `NaN`.
	 - genres_parsed: parsed into a Python list when possible (handles
		 literal list strings like `['Comedy', 'Music']`, and falls back to
		 comma-splitting).

3) Null handling / imputation
	 - runtime_minutes: imputed using the column median. The report contains the
		 count of nulls before and after imputation and the median value used.
	 - rating_avg: not aggressively imputed by default; the report records the
		 number of `NaN` values and the global median. (If desired, a follow-up
		 step could fill `NaN` with the median.)
	 - language: filled with `'Unknown'` where missing.

4) Duplicate detection and removal
	 - Duplicates are detected using the composite key: `title_normalized +
		 release_year`.
	 - All duplicates are reported (count) and up to 5 example groups are
		 printed in the report with their rows. Duplicates are removed by
		 keeping the first occurrence (stable), and the number removed is logged.

5) Outlier detection
	 - IQR (interquartile range) rule is used to flag outliers for numeric
		 columns. The script currently flags and reports outliers for:
		 - `runtime_minutes`
		 - `rating_avg`
	 - For each flagged column the report shows the count of outliers and up to
		 5 example rows. The script does not automatically cap or remove
		 outliers — it only flags them (safer, non-destructive). A capping or
		 removal policy can be added later if desired.

Examples (illustrative)
------------------------

Below are example cases taken from the report output (the actual report file
contains dataset-specific examples):

- Null handling example (runtime missing):
	- Before: title='Casa Grande', release_year=2023, runtime_minutes=None
	- After : runtime_minutes filled with median (e.g. 30.0)

- Rating parsing examples:
	- rating column value "{'average': 7.5}" -> rating_avg = 7.5
	- rating column value "{'average': None}" -> rating_avg = NaN
	- rating column unstructured text "7.0/10" -> regex extracts 7.0

- Duplicate example (group):
	- key: "the carol burnett show||1967"
		- row 1: title='The Carol Burnett Show', release_year=1967, premiere_date=1967-09-11
		- row 2: title='The Carol Burnett Show', release_year=1967, premiere_date=1967-09-11
	- Action: duplicate group detected, kept first row, removed the second.

- Outlier example (runtime):
	- title='Some Very Long Documentary', runtime_minutes=720 -> flagged as outlier
	- title='Short Clip', runtime_minutes=2 -> flagged as outlier

Notes and rationale
-------------------

- Duplicate policy: Title + Year is a sensible deduplication key for
	integrated TV show data because many entries have the same title but are
	different years (remains distinct), while exact duplicates (same show,
	same year) are typically data errors introduced during integration.

- Rating treatment: Ratings are often sparse and can be stored in many
	shapes. We extract an `rating_avg` numeric column for analytics but avoid
	filling missing values automatically to prevent introducing bias. The
	report contains counts and the median if you want to apply median
	imputation later.

- Outliers: Flagging only avoids inadvertent data loss. After review, you
	may decide to cap extreme runtime values (e.g., > 4 hours) or to remove
	obviously erroneous rows.

How to run
----------

From the repository root run:

```bash
python scripts/data_preparation_final.py
```

This will write `cleaned_tv_shows.csv` and `data_preparation_report.txt` to
`data/processed/`.

Artifacts produced
------------------

- `data/processed/cleaned_tv_shows.csv` : cleaned dataset with the added
	columns `title_normalized`, `rating_avg`, and `genres_parsed` and with
	duplicates removed.
- `data/processed/data_preparation_report.txt` : plain-text report describing
	null counts, example rows, duplicate counts and examples, outlier counts
	and examples, and the transformations applied.

