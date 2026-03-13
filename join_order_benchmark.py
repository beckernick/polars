"""
join_order_benchmark.py

Shows how join order affects performance when some dimensions are highly selective.

Setup
-----
  events   : 5 M rows
               type_id  in 0..999      (1 000 distinct values)
               user_id  in 0..999_999  (1 M distinct values)
               region_id in 0..49      (50  distinct values)
               channel_id in 0..199    (200 distinct values)
  users    : 1 M rows   (large;  all events match)
  regions  : 5  rows    (tiny;   only region_id 0-4  → ~10 % of events survive)
  types    : 10 rows    (tiny;   only type_id   0-9  → ~1  % of events survive)
  channels : 100 rows   (small;  all channel_ids 0-99 match)

Worst order : events JOIN users(1M) JOIN channels(100) JOIN regions(5) JOIN types(10)
              intermediate sizes: 5M → 5M → 5M → 500K → 50K
Best order  : events JOIN types(10) JOIN regions(5) JOIN channels(100) JOIN users(1M)
              intermediate sizes: 5M → 50K → 5K → 5K → 5K

The best order reduces each intermediate by joining the most selective tables first.
"""

import time
import polars as pl
import random
from polars.lazyframe.opt_flags import QueryOptFlags

opts = QueryOptFlags()  # all optimizations on by default
# opts.join_reorder = False       # disable only this one

random.seed(0)

N_EVENTS = 10_000_000
N_USERS = 1_000_000
N_TYPES_ALL = 1_000  # total type_ids in events
N_TYPES = 5  # only these exist in 'types' → ~0.5 % selectivity
N_REGIONS_ALL = 50  # total region_ids in events
N_REGIONS = 5  # only these exist in 'regions' → ~10 % selectivity
N_CHANNELS_ALL = 200  # total channel_ids in events
N_CHANNELS = 100  # all these exist in 'channels' → 50 % selectivity

print(f"polars    : {pl.__version__}")
print(f"events    : {N_EVENTS:>9,}")
print(f"users     : {N_USERS:>9,}  (all events match)")
print(f"channels  : {N_CHANNELS:>9,}  (~50 % of events match)")
print(f"regions   : {N_REGIONS:>9,}  (~10 % of events match)")
print(f"types     : {N_TYPES:>9,}  (~1  % of events match)")
print()

# ── build tables ──────────────────────────────────────────────────────────────
events = pl.DataFrame(
    {
        "event_id": pl.arange(N_EVENTS, eager=True),
        "user_id": pl.Series([random.randint(0, N_USERS - 1) for _ in range(N_EVENTS)]),
        "type_id": pl.Series(
            [random.randint(0, N_TYPES_ALL - 1) for _ in range(N_EVENTS)]
        ),
        "region_id": pl.Series(
            [random.randint(0, N_REGIONS_ALL - 1) for _ in range(N_EVENTS)]
        ),
        "channel_id": pl.Series(
            [random.randint(0, N_CHANNELS_ALL - 1) for _ in range(N_EVENTS)]
        ),
        "amount": pl.Series([random.random() for _ in range(N_EVENTS)]),
    }
)

users = pl.DataFrame(
    {
        "user_id": pl.arange(N_USERS, eager=True),
        "user_name": pl.Series([f"user_{i}" for i in range(N_USERS)]),
    }
)

types = pl.DataFrame(
    {
        "type_id": pl.arange(N_TYPES, eager=True),
        "type_name": pl.Series([f"type_{i}" for i in range(N_TYPES)]),
    }
)

regions = pl.DataFrame(
    {
        "region_id": pl.arange(N_REGIONS, eager=True),
        "region_name": pl.Series([f"region_{i}" for i in range(N_REGIONS)]),
    }
)

channels = pl.DataFrame(
    {
        "channel_id": pl.arange(N_CHANNELS, eager=True),
        "channel_name": pl.Series([f"channel_{i}" for i in range(N_CHANNELS)]),
    }
)

surviving = (
    events.filter(pl.col("type_id") < N_TYPES)
    .filter(pl.col("region_id") < N_REGIONS)
    .filter(pl.col("channel_id") < N_CHANNELS)
    .shape[0]
)
print(f"events surviving all joins: {surviving:,}  ({surviving / N_EVENTS:.2%})\n")

# ── lazy frames ───────────────────────────────────────────────────────────────
lf_events = events.lazy()
lf_users = users.lazy()
lf_types = types.lazy()
lf_regions = regions.lazy()
lf_channels = channels.lazy()

# ── benchmark helper ──────────────────────────────────────────────────────────
RUNS = 3


def bench(label: str, fn) -> float:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:<55s}  {best:.3f}s   rows={result.shape[0]:>8,}")
    return best


# ── queries ───────────────────────────────────────────────────────────────────
print("=" * 75)
print(f"STREAMING ENGINE  (best of {RUNS} runs)")
print("=" * 75)

t_worst_s = bench(
    "WORST: JOIN users → channels → regions → types",
    lambda: (
        lf_events.join(lf_users, on="user_id", how="inner")
        .join(lf_channels, on="channel_id", how="inner")
        .join(lf_regions, on="region_id", how="inner")
        .join(lf_types, on="type_id", how="inner")
        .collect(engine="streaming", optimizations=opts)
    ),
)

t_best_s = bench(
    "BEST:  JOIN types  → regions → channels → users",
    lambda: (
        lf_events.join(lf_types, on="type_id", how="inner")
        .join(lf_regions, on="region_id", how="inner")
        .join(lf_channels, on="channel_id", how="inner")
        .join(lf_users, on="user_id", how="inner")
        .collect(engine="streaming", optimizations=opts)
    ),
)

print(f"\n  Speedup from correct join order: {t_worst_s / t_best_s:.1f}×")
