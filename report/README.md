# Report pipeline

One pipeline produces everything the README and the website report show: numbers, imagery and the GIF. Its output,
the **bundle** in `report/bundle/`, is committed. The website copies the bundle (pinned to a commit of this repository)
and draws the report's figures and interactive components from it.

```
report/
  scripts/   build steps (Python), run from the repository root
    metrics.py       every number, from the run outputs
    imagery.py       the timeline fire: frames, GIF, filmstrip (fire chosen by a fixed rule), and the card cover
    satellite.py     NASA GIBS true-colour images for each timeline day, resampled onto the model's grid
    views.py         model walk-through (inputs, encoder features) and the threshold-explainer day
    samples.py       fixed input-coverage rule for the model-view day
    wind.py          direction-only arrow geometry (east/north -> north-up image coordinates)
    gallery.py       all-input gallery metadata, physical-unit conversion and range summaries
    architecture.py  model-derived shapes/counts and removable hooks on the real fusion/decoder blocks
    build.py         runs all of the above, writes site.json and manifest.json
  figures/   render.mjs: captures the README figures in assets/ from the website report
  sources/   external checks (timeline-fire.json: the timeline fire matched to its official perimeter)
  bundle/    generated output, committed
    site.json        all numbers the website report reads (schema_version, draft flag)
    manifest.json    every file's SHA-256 and size, the source commit, build time, schema version
    images/          timeline frames, satellite-<t>.jpg per day, cover.jpg, model views, explainer data, GIF, filmstrip
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

## Satellite images and the cover

The timeline's swipe-compare and the card cover use true-colour VIIRS images from NASA GIBS
(`VIIRS_SNPP_CorrectedReflectance_TrueColor`), one per timeline day, for the day the forecast is about (`next_date`).
`satellite.py` resamples each onto the timeline crop's 375 m grid, so a pixel lines up with the maps; the unit tests check
the geometry without network access. **The first build fetches about 66 small images over the network** (raw responses
are cached in `/tmp/fireline-cache/satellite/`, so a rebuild does not fetch twice). Each image is a daily mosaic, not the
overpass of the fire detections, and smoke or cloud can hide the ground and the fire front. Display only: nothing in the
model or metrics uses them. Credit and terms: `LICENSES/NASA-GIBS.md`; provenance is in `timeline.satellite`.

The cover (`images/cover.jpg`) is the swipe view for one day, chosen by eye (`COVER_NEXT_DATE` in `imagery.py`, recorded
in `timeline.cover_day`): terrain map left, satellite right, the fire layers over both, divider at 50%.

The model walk-through uses the timeline day nearest in calendar days to the filmstrip's median-F1 **typical** day,
with ties going to the earlier date, subject to a fixed input-coverage gate: each of the five input days must have
at least 50% finite pixels in each of NDVI, EVI2 and the three VIIRS reflectance channels within the shared timeline
crop. Today's ERC, maximum temperature, wind speed/direction and elevation must each meet the same gate. The build
fails if no day qualifies; it never falls back to a missing-input sample. Finite coverage does not guarantee
cloud-free reflectance. Selection uses input availability, not the candidate's forecast quality. The chosen date
and rule are recorded in `timeline.model_view`; the timeline, filmstrip and threshold explainer retain their
original days. For Dixie this selects **27 August 2021 → 28 August**, with inputs from 23–27 August; the threshold
explainer remains on **18 August → 19 August**. This only re-renders the already-selected fire, not the test-year
evaluation.

The wind-speed thumbnails include sparse arrows from the unnormalised daily-mean GRIDMET east/north components,
sampled every 32 pixels (12 km, coarser than the ~4.6 km source). They point **to** the flow direction: the model's
meteorological FROM conversion is `u = -speed*sin(direction)`, `v = -speed*cos(direction)`; north-up image coordinates
use `(u, -v)`. Equal arrow lengths encode direction only; the underlying colours show speed. Calm or missing vectors
are omitted. The selected raster is north-up (positive x pixel size, negative y pixel size, no rotation).

`timeline.model_view.gallery` contains every physical input for that same day and crop, plus a diagnostic
M11/I2/I1 false-colour view and raw aspect. Native source resolutions follow WSTS Table 1; all tiles remain on the
375 m WSTS grid. VIIRS reflectance and vegetation indices are divided by 10,000, GRIDMET temperatures converted
from K to °C, while GFS temperatures are already °C. Continuous colours stretch to the finite 2nd–98th percentiles;
tile metadata records the actual finite min/max separately, constant fields explicitly, missing fractions and
RGB per-band ranges. MODIS IGBP land cover uses fixed discrete class colours and a legend of present classes,
representing the model's 17 one-hot channels (absent classes are zero). Forecasts show this input day's stored GFS
00 UTC fields, not target-day observed weather; precipitation is the archive total, not a 24-hour mean rate.

The architecture view additionally records `timeline.model_view.architecture`: shapes for one 128 × 128 training
crop (B=1 omitted), counts from the instantiated checkpoint model, and links to the whole-block encoder, fused and
decoder maps. It adds 1/2 and 1/4 encoder PCA views for today, fusion PCA at all four scales, and three decoder PCA
views. Original full-resolution and 1/8 five-day PNGs and the fused 1/8 PNG retain their filenames and content.
Encoder PCA uses a shared basis across all five days at each scale, even where only today is displayed. Fused and
decoder maps each use their own PCA basis; colours cannot be compared between stages. Inference still runs on full
images, reflection-padded to a multiple of 32; diagram shapes illustrate training crops, not the displayed map crop.
The real `SpreadModel.decode` is observed with temporary forward hooks rather than reimplemented, and tests check
both exact output equivalence and hook cleanup. The input stage shows all 18 daily channels and separates static
views (21 model channels after categorical one-hot expansion) and the two diagnostic views.

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

Measured values come from the saved runs above. The additional validation-only seed check reads
`v2-seed-1/rings-2020.json` (baseline reference and seed 1) and `v2-seed-2/rings-2020.json` (seed 2), and computes the
sample standard deviation across three baseline seeds; it does not estimate JEPA's seed noise. Published model
sizes and the attention comparison are cited reference data in `report/sources/model-size.json` (original WSTS
Table 5 and arXiv:2502.12003v3 Table 2). `metrics.model_context` retains the two papers' differing Res18 counts and
derives the attention gain range on **original WSTS**, not on the WSTS+ evaluation. Size tables never display AP.
The website's `ModelSize.astro` and the generated README `assets/model-size.png` use the same bundle context.
Each `rings` summary also includes `constant_prevalence_reference`, derived as p(1-p) from the saved evaluation
prevalence. This is explicitly a hindsight reference, not a deployable forecast. `Calibration.astro` displays the
saved Brier scores for validation and test; no reliability calculation, calibration fitting or test rescoring is
performed by the report. Planned calibration distinguishes strictly increasing maps (rank/AP-preserving) from
isotonic regression, which can introduce ties.
All input tiles and their metadata/land-cover legend are now always visible inside Architecture's input stage;
there is no separate collapsed All Inputs section. Training-loss prose/formulas are authored once in the website
MDX using its existing remark-math/KaTeX support, wrapped by `TrainingLoss.astro`, and captured as
`assets/training-loss.png` for the README. `tests/test_report_loss.py` checks the stated masked BCE + batch-global
smoothed Dice formula, true-negative/false-alarm behaviour, empty-fire smoothing and gradients through all five days.

## Using the bundle elsewhere

- `site.json` has a `schema_version`; consumers should check it and fail on a version they do not know.
- A bundle with `"draft": true` lacks the test-year results and must not be published.
- Imagery is derived from WildfireSpreadTS (CC BY 4.0); keep the attribution with any copy.
