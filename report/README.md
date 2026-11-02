# Report pipeline

One pipeline produces everything the README and the website report show: numbers, imagery and the GIF. Its output,
the **bundle** in `report/bundle/`, is committed. The website copies the bundle (pinned to a commit of this repository)
and draws the report's figures and interactive components from it.

```
report/
  scripts/   build steps (Python), run from the repository root
    metrics.py       every number, from the run outputs
    imagery.py       the timeline fire: frames, GIF, filmstrip (fire chosen by a fixed rule)
    views.py         model walk-through (inputs, encoder features) and the threshold-explainer day
    build.py         runs all of the above, writes site.json and manifest.json
  figures/   render.mjs: captures the README figures in assets/ from the website report
  sources/   external checks (timeline-fire.json: the timeline fire matched to its official perimeter)
  bundle/    generated output, committed
    site.json        all numbers the website report reads (schema_version, draft flag)
    manifest.json    every file's SHA-256 and size, the source commit, build time, schema version
    images/          timeline frames, model views, explainer data, GIF, filmstrip
```

## Rebuilding

```bash
uv run --group notebook python report/scripts/build.py           # needs the test-year evaluation
uv run --group notebook python report/scripts/build.py --draft   # validation-only draft
node report/figures/render.mjs --url <website report URL>         # README figures in assets/ (local Chrome)
```

The imagery steps also need the converted fires (`datasets/processed/wsts/`), the supervised checkpoint
(`outputs/modal/unet-sup-02/checkpoints/best.pt`) and the training-year normalisation statistics
(`outputs/modal/baselines-01/stats-2018-2019.json`); they run on CPU.

## Inputs (local, not in Git)

`metrics.py` reads downloaded run outputs under `outputs/modal/`:

| Section | Reads |
| --- | --- |
| dataset | `ingest-wsts-01/validation.json` |
| validation results | `baselines-01/persistence-2020.json`, `unet-sup-02/{config.json,metrics.jsonl}`, `far-fire-01/{rings,far-fire}-2020.json` |
| JEPA | `jepa-01`, `jepa-02a`, `jepa-02b` (pretraining gates), `jepa-02b-probe-r2`, `probe-random-03` (probe curves) |
| leak | `leak-wsts-01/composite_leak.json` |
| wind features | `unet-wind-01/{metrics.jsonl,rings-2020.json}` |
| test year (2021) | `test-2021-01/{rings,far-fire}-2021.json`, the single test-year evaluation |

No number in the report is typed by hand: every value comes from these files.

## Using the bundle elsewhere

- `site.json` has a `schema_version`; consumers should check it and fail on a version they do not know.
- A bundle with `"draft": true` lacks the test-year results and must not be published.
- Imagery is derived from WildfireSpreadTS (CC BY 4.0); keep the attribution with any copy.
