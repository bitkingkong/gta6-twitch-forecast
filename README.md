# GTA VI launch-day Twitch forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour (2–24) of its release day, 2026-11-19 UTC, as five quantiles per hour.

## Method

The house baseline says every hour stays at hour 1's print. Real launch days
have a shape, so this entry learns the shape from the case's comparables:
for every past big launch, take the log ratio of each hour's viewers to that
launch's hour-1 viewers, then forecast

    q_tau(h) = p1 * exp(m_h + z_tau * s_h)

where `p1` is this case's hour-1 print, `m_h` is the weighted median of the
comparables' log ratios at hour `h`, `s_h` is a shrunk weighted spread of
those ratios (hours where launches disagree get honestly wide intervals),
and `z_tau` is the normal quantile at each scored level.

Comparables are weighted by similarity to a console-only single-player
launch: console-only counts more, rolling-midnight unlocks (like GTA VI's)
count more than simultaneous global unlocks, recent launches count a little
more, and launches whose hour-1 audience is near this case's count more
(Gaussian kernel in log space). Early-access-contaminated and weak analogues
count less. GTA VI's own pre-release category history is not a launch, so
it never enters the shape.

The spread is calibrated on the practice cases (raw cross-launch spread
covered 96–100% of hours; an honest 5%–95% band should cover ~90%, so
intervals are scaled by 0.85), and hour-1 viewers-per-channel plus the
top-20 live streams (final only) widen the intervals slightly when the
audience is concentrated in a few channels.

Any missing field degrades gracefully to the persistence baseline, so the
program never crashes and never emits an invalid case.

## Validation (local, practice cases p00–p03)

- Format: 23 rows, quantiles non-decreasing, in (0, 1e9] — all valid.
- Determinism: two runs byte-identical.
- Runtime: ~0.3 s for all four cases (limit is 8 s per case).
- Pinball loss vs the real SullyGnome hourly answers: **0.214 mean**
  (p00 0.394, p01 0.163, p02 0.126, p03 0.172) vs **0.412** for the
  starter baseline on the same answers (house quotes ~0.387).
- Leave-one-out over 37 held-out comparable launches: **0.439** vs
  **0.930** for the baseline.

## Run

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Python 3.13, standard library only, no network, deterministic.
