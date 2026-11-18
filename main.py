"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Method: shape-aware empirical forecast from the case's comparables.
For every past big launch in the case, look at its release day (UTC) and record,
for each hour h = 2..24, the log ratio log(viewers_h / viewers_hour_1).  The
forecast for the target launch is then

    q_tau(h) = p1 * exp(m_h + z_tau * s_h)

where p1 is the target's hour-1 print, m_h is the *weighted median* of the
comparables' log ratios at hour h, s_h is a shrunk weighted spread of those
log ratios (so hours where launches disagree get honestly wide intervals),
and z_tau is the standard normal quantile at each scored level.

Comparables are weighted by similarity to a console-only single-player launch
like GTA VI: console-only launches count more, early-access-contaminated and
weak analogues count less, recent launches count a little more, and launches
whose hour-1 audience is close in size to this case's count more (a Gaussian
kernel in log space).  The target's own pre-release category history is not
a launch, so it never enters the shape; it is only a fallback anchor if the
case somehow has no hour-1 print.

Hour-1 channels (and, in the final, the top-20 live streams) nudge the
spread: a category concentrated in a few channels is more volatile, so its
intervals widen a little.  Everything degrades gracefully: any missing or
empty field falls back to the persistence baseline (every hour at hour 1's
print, spread 0.15*sqrt(h-1) in log), and the program never crashes and
never emits an invalid case.

Python 3.13, standard library only, no network, deterministic, fast.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input/), writes
/output/<id>.out ($MAKEFUN_OUTPUT_DIR else /output/) for each.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Fallback spread (the house baseline's): log-sigma per sqrt hour.
BASE_SIGMA = 0.15
# Spread calibration: on the four practice cases the raw cross-launch spread
# covered 96-100% of hours (an honest 5%-95% band should cover ~90%), so
# intervals are scaled by 0.85. Calibrated on practice, not per-case tuned.
SIGMA_SCALE = 0.85


def parse_ts(text):
    """ISO 8601 UTC timestamp -> aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z = [norm_ppf(tau) for tau in LEVELS]


def round2(x):
    """Round to 2 decimals, half up (matches the starter's rounding)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def clamp_q(x):
    """A quantile must be a finite number in (0, 1e9]."""
    if x is None or not math.isfinite(x):
        return 1.0
    if x <= 0:
        return 1.0
    if x > 1_000_000_000:
        return 1_000_000_000
    return x


def weighted_median(pairs):
    """Weighted median of (value, weight) pairs; pairs must be non-empty."""
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2.0:
            return value
    return ordered[-1][0]


def weighted_std(pairs, center):
    """Weighted population std of values around center; returns (std, n_eff)."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0, 0.0
    var = sum(w * (v - center) ** 2 for v, w in pairs) / total
    n_eff = total * total / sum(w * w for _, w in pairs) if pairs else 0.0
    return math.sqrt(max(var, 0.0)), n_eff


def comp_tier(comp):
    """Tier tag from the unlock text ('Tier: comparable' etc.)."""
    unlock = comp.get("unlock") or ""
    m = re.search(r"Tier:\s*([a-z\-]+)", unlock)
    return m.group(1) if m else "comparable"


def comp_weight(comp, p1, year_now=2026):
    """Similarity weight of one comparable launch for this case."""
    tier = comp_tier(comp)
    if tier == "target-history":
        return 0.0
    base = {"comparable": 1.0, "early-access-contaminated": 0.35,
            "weak": 0.30}.get(tier, 0.6)
    platforms = comp.get("platforms") or ""
    # Console-only launches are the closest analogue to GTA VI (PS5 + Xbox).
    if "PC" not in platforms:
        base *= 1.25
    # GTA VI unlocks at midnight local, rolling country by country; launches
    # with the same rolling-midnight pattern share its day shape, while
    # single simultaneous global unlocks spike all at once and differ.
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" in unlock or "rolling" in unlock:
        base *= 1.3
    elif "single global" in unlock or "simultaneous" in unlock:
        base *= 0.7
    # Recent launches count a little more.
    try:
        year = int((comp.get("day_start") or "")[:4])
    except ValueError:
        year = 2020
    base *= 0.75 + 0.25 * max(0.0, min(1.0, (year - 2020) / 6.0))
    # Launches whose hour-1 audience is near this case's count more
    # (Gaussian kernel in log space, bandwidth 0.8).
    hourly = comp.get("hourly") or []
    p1c = None
    day = parse_ts(comp.get("day_start"))
    if day is not None:
        for pt in hourly:
            at = parse_ts(pt.get("from"))
            if at is not None and at == day and pt.get("viewers"):
                p1c = float(pt["viewers"])
                break
    if p1c and p1c > 0 and p1 > 0:
        base *= math.exp(-0.5 * ((math.log(p1c) - math.log(p1)) / 0.8) ** 2)
    return base


def release_day_ratios(comp):
    """{hour: viewers} for the comparable's UTC release day, hours 1..24."""
    day = parse_ts(comp.get("day_start"))
    if day is None:
        return {}
    out = {}
    for pt in comp.get("hourly") or []:
        at = parse_ts(pt.get("from"))
        if at is None:
            continue
        delta = (at - day).total_seconds()
        if 0 <= delta < 24 * 3600 and delta % 3600 == 0:
            viewers = pt.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                out[int(delta // 3600) + 1] = float(viewers)
    return out


def hour_stats(case, p1):
    """Per-hour (median log ratio, sigma) from the comparables.

    Returns a dict hour -> (m, s). Hours with too few comparables fall
    back to the neighbouring hours' stats, then to the flat baseline.
    """
    comps = []
    for comp in case.get("comparables") or []:
        w = comp_weight(comp, p1)
        if w <= 0:
            continue
        ratios = release_day_ratios(comp)
        if 1 in ratios:
            comps.append((w, ratios))
    raw = {}
    for h in range(2, 25):
        pairs = []
        for w, ratios in comps:
            if h in ratios:
                pairs.append((math.log(ratios[h] / ratios[1]), w))
        if len(pairs) >= 6:
            m = weighted_median(pairs)
            s, n_eff = weighted_std(pairs, m)
            raw[h] = (m, s, n_eff)
    if not raw:
        return {}
    pooled = sorted(s for _, s, _ in raw.values())
    sigma_pool = pooled[len(pooled) // 2] if pooled else 0.35
    sigma_pool = min(max(sigma_pool, 0.15), 0.60)
    stats = {}
    for h in range(2, 25):
        if h in raw:
            m, s, n_eff = raw[h]
            # Shrink the spread toward the pooled spread (small samples lie).
            s = math.sqrt((n_eff * s * s + 4.0 * sigma_pool * sigma_pool) / (n_eff + 4.0))
            stats[h] = (m, min(max(s, 0.15), 0.90))
        else:
            # Borrow the nearest hour that has stats.
            near = min(raw, key=lambda hh: (abs(hh - h), hh))
            stats[h] = raw[near][:2]
            stats[h] = (stats[h][0], min(max(stats[h][1], 0.15), 0.90))
    # Light 3-point smoothing of the median shape (keeps genuine unlock
    # jumps, damps hour-to-hour noise from changing comparable subsets).
    medians = [stats[h][0] for h in range(2, 25)]
    smooth = []
    for i, m in enumerate(medians):
        lo = medians[i - 1] if i > 0 else m
        hi = medians[i + 1] if i < len(medians) - 1 else m
        smooth.append(0.25 * lo + 0.5 * m + 0.25 * hi)
    for i, h in enumerate(range(2, 25)):
        stats[h] = (smooth[i], stats[h][1])
    return stats


def spread_multiplier(case, p1):
    """Widen intervals a little when the audience is concentrated.

    Uses hour-1 viewers-per-channel vs the comparables' median, and (final
    only) the top stream's share of the hour-1 print. Bounded, small.
    """
    mult = 1.0
    try:
        known1 = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
        ch = known1.get("channels") if known1 else None
        if ch and ch > 0 and p1 > 0:
            vpc = p1 / float(ch)
            vpcs = []
            for comp in case.get("comparables") or []:
                ratios = release_day_ratios(comp)
                if 1 in ratios:
                    day = parse_ts(comp.get("day_start"))
                    for pt in comp.get("hourly") or []:
                        at = parse_ts(pt.get("from"))
                        if at is not None and at == day and pt.get("channels"):
                            vpcs.append(ratios[1] / float(pt["channels"]))
                            break
            if vpcs:
                vpcs.sort()
                med = vpcs[len(vpcs) // 2]
                if med > 0:
                    mult *= math.exp(0.08 * max(-1.5, min(1.5, math.log(vpc / med))))
        streams = case.get("streams")
        if streams:
            top = max((s.get("viewers") or 0) for s in streams if isinstance(s, dict))
            if top and p1 > 0:
                share = top / p1
                if share > 0.5:
                    mult *= 1.0 + 0.15 * min(share - 0.5, 0.5)
    except Exception:
        return 1.0
    return min(max(mult, 0.9), 1.3)


def baseline_rows(p1, hours):
    """The persistence fallback: flat median, widening spread."""
    rows = []
    for h in hours:
        rows.append({"hour": h, "q": [clamp_q(round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)))) for z in Z]})
    return rows


def forecast_case(case, name):
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        try:
            p1 = max(1.0, float(first.get("viewers") or 1))
        except (TypeError, ValueError):
            p1 = 1.0
    if not p1 or not math.isfinite(p1):
        p1 = 1.0
    try:
        stats = hour_stats(case, p1)
    except Exception:
        stats = {}
    if not stats:
        return {"hours": baseline_rows(p1, hours)}
    mult = spread_multiplier(case, p1)
    rows = []
    for h in hours:
        m, s = stats.get(h, (0.0, BASE_SIGMA * math.sqrt(max(h - 1, 1))))
        s = min(max(s * mult * SIGMA_SCALE, 0.12), 1.0)
        qs = []
        prev = 0.0
        for z in Z:
            q = clamp_q(round2(p1 * math.exp(m + z * s)))
            q = max(q, prev)  # keep the row non-decreasing
            qs.append(q)
            prev = q
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Last-resort fallback: never emit an invalid case.
            try:
                first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
                p1 = max(1.0, float(first.get("viewers") or 1)) if first else 1.0
            except Exception:
                p1 = 1.0
            hours = [row["hour"] for row in case.get("forecast") or []] or list(range(2, 25))
            forecast = {"hours": baseline_rows(p1, hours)}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
