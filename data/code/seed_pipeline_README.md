# Seed Pipeline (Discovery + Enhancement Only)

This pipeline builds and iteratively expands a domain seed corpus.
It intentionally does **not** generate QA pairs.

## Script

- `data/code/seed_pipeline.py`
- Config: `data/code/seed_pipeline_config.yaml`

## Domains Included

- `delhi_food`
- `poetry_style`
- `mahakumbh`
- `isro`

## Pipeline Stages

1. `init`: create domain workspace + load configured seed URLs.
2. `discover`: crawl from seed URLs and collect discoverable URLs.
3. `ingest`: fetch discovered pages and chunk into candidate corpus.
4. `bootstrap-labels`: weak-label candidate chunks (positive/negative) from keyword rules.
5. `train`: train a lightweight Naive Bayes discriminator.
6. `score`: classify candidate chunks as `accept` / `review` / `reject`.
7. `promote`: move accepted chunks into seed corpus and add their URLs as seeds.
8. `iterate`: run the whole loop in one command.

## Example Commands

```powershell
python data\code\seed_pipeline.py --domain delhi_food init
python data\code\seed_pipeline.py --domain delhi_food discover --max-depth 1 --max-pages 60
python data\code\seed_pipeline.py --domain delhi_food ingest --max-urls 60
python data\code\seed_pipeline.py --domain delhi_food bootstrap-labels --max-pos 300 --max-neg 300
python data\code\seed_pipeline.py --domain delhi_food train --backend fasttext
python data\code\seed_pipeline.py --domain delhi_food score --backend fasttext
python data\code\seed_pipeline.py --domain delhi_food promote --min-score 0.85
```

Full iteration:

```powershell
python data\code\seed_pipeline.py --domain delhi_food iterate --backend fasttext --max-pages 80 --max-urls 80
```

## Output Layout

- `data/seed_pipeline/domains/<domain>/seed_urls.jsonl`
- `data/seed_pipeline/domains/<domain>/discovered_urls.jsonl`
- `data/seed_pipeline/domains/<domain>/candidate_chunks.jsonl`
- `data/seed_pipeline/domains/<domain>/labels.jsonl`
- `data/seed_pipeline/domains/<domain>/discriminator_model.json`
- `data/seed_pipeline/domains/<domain>/scored_chunks.jsonl`
- `data/seed_pipeline/domains/<domain>/seed_chunks.jsonl`
- `data/seed_pipeline/domains/<domain>/run_log.jsonl`

## Notes

- Discovery quality depends heavily on `allowed_domains` and keywords in config.
- `bootstrap-labels` is only a starter for the first iteration; you should later add manual labels for higher precision.
- For stricter filtering, raise `--accept-threshold` and `--min-score`.
- The default training backend is `fasttext` (Meta FAIR fastText style classifier).
