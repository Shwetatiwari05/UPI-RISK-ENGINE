# Parquet and Chunked Pipeline

The batch pipeline now converts the mapped transaction data to compressed Apache
Parquet before preprocessing. It processes row groups in bounded chunks, so the
7.2M-row dataset is not concatenated into one in-memory dataframe.

## Run the pipeline

From the project root:

### macOS / Linux
```bash
source .venv/bin/activate
python main.py --all
```

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
python main.py --all
```

The stages are:

1. Raw CSV/Parquet files are read in chunks and mapped to the common schema, then
   written to `data/merged/mapped_common_schema.parquet` with Snappy compression.
2. A bounded sample is used to fit preprocessing state once.
3. Point-in-time feature engineering runs inside the streamed preprocessing pass.
4. Supervised models train once on a bounded fraud-preserving sample from the
   training period; isotonic calibrators are then fitted on a held-out,
   natural-prevalence frame disjoint from that sample.
5. Anomaly models train once on a uniform, label-independent sample from the
   training period.
6. Fusion thresholds for FRAUD_LIKELY and the ambiguous review band are tuned on
   out-of-time rows and persisted to `models/fusion_thresholds.json`.

The transformed output is written to:

```text
data/processed/processed_features.parquet
```

## Adjust the chunk size

The default is `100000` rows per chunk. Change it from the command line:

```powershell
python main.py --all --chunk-size 50000
```

Use a smaller value when RAM is limited. Use a larger value when the machine has
more memory and Parquet conversion is I/O-bound. A practical starting range is
`50000` to `250000`.

The number of rows used to fit preprocessing state is independently configurable:

```powershell
python main.py --all --chunk-size 100000 --fit-rows 200000
```

`--fit-rows` does not discard rows from the transformed Parquet output. It only
controls the bounded sample used to learn imputer, scaler, encoder, and outlier
clipping state. Unknown categories are handled by the existing encoder behavior.

## Model training limits

The current algorithms are batch estimators:

- Random Forest does not implement `partial_fit`.
- XGBoost is trained with one `fit` call in the existing module.
- Isolation Forest is trained with one `fit` call.
- LOF with `novelty=True` is trained with one `fit` call.

Therefore the pipeline does not call `fit` once per chunk. That would overwrite
previous learning or create inconsistent models. Instead, Parquet is scanned in
chunks and a reproducible bounded sample is passed to each existing batch model.
Adjust those sample limits when needed:

```powershell
python main.py --all --supervised-rows 700000 --legitimate-ratio 3 --anomaly-rows 200000
```

Supervised sampling preserves every fraud row when the row budget allows it, then
selects legitimate rows at the configured ratio. Legitimate rows are sampled
proportionally across available hour/day strata. This fraud-preserving
undersampling is the only class-imbalance correction applied — neither model
uses `class_weight` or `scale_pos_weight` re-weighting on top of it. Raw model
scores are mapped to real-world fraud probabilities by isotonic calibration on
a held-out, natural-prevalence calibration frame disjoint from training.
The default legitimate ratio is 3:1 and can be changed with
`--legitimate-ratio 5`.

Anomaly sampling is uniform and label-independent because fraud labels should not
rebalance an unsupervised detector.

## Parquet behavior

- `pyarrow.parquet.ParquetFile.iter_batches` is used for Parquet reads.
- CSV inputs use `pandas.read_csv(..., chunksize=...)` and only columns needed for
  schema mapping are read.
- Parquet writes use Snappy compression and one row group per processing chunk.
- The API analytics endpoint reads only the columns required for its reports.
- Generated transaction IDs include the source, original identifier context, and
  deterministic row offset, so chunk boundaries and cross-dataset overlaps do not
  create duplicate IDs.

## Correctness and memory checks

The conversion logs each mapped chunk and cumulative row count. The preprocessing
logs each tenth processed chunk and cumulative row count. The final Parquet metadata
can be checked with:

```powershell
python -c "import pyarrow.parquet as pq; p=pq.ParquetFile('data/merged/mapped_common_schema.parquet'); print(p.metadata.num_rows, p.num_row_groups, p.schema.names)"
```

The sum of row-group counts should equal the mapped row count in
`reports/schema_reports.json`. The processed Parquet file should have the same row
count as the mapped Parquet file. Duplicate IDs now stop the pipeline instead of
being silently removed. Row accounting is written to
`reports/pipeline_row_counts.json` with `rows_read`, `rows_removed`, and
`rows_written` for mapped and processed stages.

For a quick read test:

```powershell
python -c "import pyarrow.parquet as pq; t=pq.read_table('data/merged/mapped_common_schema.parquet', columns=['transaction_id','amount','fraud_label']); print(t.num_rows, t.column_names)"
```

## Important limitation

Feature engineering is invoked once for each bounded chunk, while the fitted
preprocessor is shared across all chunks. This keeps the existing feature formulas
and avoids fitting scalers/encoders independently. The non-incremental model
algorithms remain batch-trained by design; chunking is used for ingestion,
Parquet conversion, preprocessing, analytics, and bounded model sampling.
