# Model card: `coffee-price-per-kg`

Predicts what a kilogram of roasted coffee costs on a Mexico City shelf, from what a
buyer can know before paying: the shop, the bag's size, and what the shop's sheet says
about the coffee.

| | |
|---|---|
| Version described | v5 (`champion`) |
| Type | LightGBM regressor inside a scikit-learn pipeline |
| Items | One offer: a coffee in one size. 510 offers of 157 coffees, 4 roasters |
| Read on | 2026-09-21, one read of the shops' catalogues |
| Evaluation | Out of fold over every coffee, grouped by coffee (see below) |
| Tracking | MLflow experiment `coffee-offer`, tagged with the git commit that produced it |

## What it is for

Estimating the going price of a bag from its shop, its size and its origin: sanity
checking a price, seeing what altitude or a Gesha is worth on a shelf, and answering
"what would this cost?" through the agent's prediction tool.

It is a **portfolio and teaching artifact**. It is not a fair-value estimate for a
producer, not a recommendation of what to charge, and not a forecast: it was fitted on a
single read of four catalogues and knows nothing about how prices move over time.

## Inputs

| Feature | Where it comes from |
|---|---|
| `shop` | The roaster selling it |
| `bag_grams` | The size, read from the offer's titles - never the platform's own weight, which contradicts them in 77 offers |
| `country`, `state` | Its coffee's sheet, as PSD and SIAP name them; the value its origins agree on, `multiple` when they differ, empty when the sheet does not say |
| `processing_method`, `variety` | The same, in the CQI's vocabulary |
| `altitude_m` | Mean of the midpoints of the altitude ranges its origins state |
| `origins_n`, `varieties_n` | How many origins and varieties the sheet names; more than one origin is a blend |
| `variety_<name>` (16 columns) | One per variety that appears in at least four coffees, because a blend's summarised variety hides them. Chosen by how often each appears, never by price |
| `producer` | As the sheet writes it |

Nothing is imputed. A missing altitude reaches LightGBM as a missing value and the model
learns which way to send it; a missing category is encoded as missing, exactly as an
unseen category is at serving time. With state missing in 46% of offers, imputing a mode
would invent a state - and a shop that does not say where a coffee grew is telling us
something about that coffee.

Excluded on purpose: `price_mxn` (the target times the size), the 14 offers with no size
to divide by (kits, samplers) and the 3 whose price was flagged as copied from another
size.

## Performance

Every coffee is scored by a model fitted without it (grouped 5-fold over all 510 offers).
A single 25% draw of coffees was tried first and discarded: it decides a small model's
fate on 39 coffees and threw away three quarters of the evidence.

| Out of fold | MAE (MXN/kg) |
|---|---|
| **Model** | **229.9** |
| Baseline: every bag at its shop's mean | 252.7 |
| Baseline: every bag at its shop's mean **for that size** | 239.8 |

| Paired comparison, resampling coffees | Difference | Certainty |
|---|---|---|
| vs the shop's mean | -22.8 (95% CI -46.9 to +2.9) | **96%** - promoted |
| vs the shop's mean for that size | -9.9 | 78% - short of the bar |

Bags of one coffee are not independent evidence: the bootstrap resamples whole coffees,
not offers, or four sizes of one mistake would count as four.

### Where the edge is, and where there is none

| | Model | Shop mean |
|---|---|---|
| The 435 offers whose coffee states an origin | 237.4 | 264.2 |
| The 75 offers whose coffee has no sheet | 186.4 | 186.1 |

The model earns its place on coffees that say where they grew, and matches the baseline
exactly where there is nothing to know. Altitude is what it leans on most, then the shop
and the size; shuffling `processing_method` or the summarised `variety` makes it slightly
*better*, which is why the per-variety columns exist.

## Limitations and bias

- **It is mostly one shop.** 128 of the 157 coffees are Almanegra's, so "what a kilo
  costs" is largely what Almanegra charges.
- **Part of what it knows is the size.** Against a baseline that also knows the size the
  edge is 9.9 MXN/kg at 78% certainty: real, but not established at this project's bar.
- **One snapshot.** Prices were read on a single day; nothing here says how they move.
- **Specialty only.** Four roasters' online catalogues are not the city's coffee: no
  supermarket coffee, no cafés' menus.

## Alternatives tried

Measured against the same baseline, on the same split, in
`experiments/price_estimators.py`: a ridge on one-hot columns (328.5 MAE, worse than the
baseline), a random forest (311.9), LightGBM on a log target (323.0) and the tuned
LightGBM (319.0) - single-draw numbers, before the evaluation moved out of fold. No
estimator was the answer.

What was: how the hyperparameters were chosen. Five folds over 118 coffees is a noisy
objective, and the tuner picked a learning rate of 0.011 over 174 trees - a model so slow
it barely left the mean - with a `min_frequency` of 27, which grouped nearly every
variety and state into "infrequent". Drawing the folds four times over (`cv_repeats`) and
narrowing the search to what a few hundred rows support produced a small model (110
trees, 4 leaves) that generalises.

## Promotion and monitoring

Promoted by the same gate as every model: it must beat the strongest baseline in at least
95% of paired bootstrap resamples, and the current champion too. `min_probability_better`
was never lowered for it.

Next version: more roasters, and the re-reads of stage 4, which turn "what do these
coffees cost" into "how do these prices move" - a question with far more statistical
power than comparing different coffees to each other.

## Reproducing

```bash
make data                       # extract, validate, clean
make ml MODEL=offer             # features, tuned training, batch scores
uv run python experiments/price_estimators.py   # the estimators that were not the answer
```
