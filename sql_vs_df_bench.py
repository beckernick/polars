"""
Benchmark: Polars SQL interface vs native LazyFrame API

Demonstrates that the SQL interface fails to push WHERE predicates through
joins, causing it to join full tables before filtering — while the LazyFrame
API filters each source before joining.

Query: join orders + items, filter on category and status, sum amount by order.
"""

import time
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

RNG = np.random.default_rng(42)

N_ORDERS = 2_000_000
N_ITEMS  = 10_000_000

# ── synthetic data ────────────────────────────────────────────────────────────
# ~10% of orders match the filter (status == "A")
orders = pl.LazyFrame({
    "order_id":  np.arange(N_ORDERS, dtype=np.int32),
    "status":    RNG.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                            size=N_ORDERS),
    "region":    RNG.integers(0, 10, size=N_ORDERS, dtype=np.int32),
})

# ~10% of items match the filter (category == 1)
items = pl.LazyFrame({
    "item_id":   np.arange(N_ITEMS, dtype=np.int32),
    "order_id":  RNG.integers(0, N_ORDERS, size=N_ITEMS, dtype=np.int32),
    "category":  RNG.integers(1, 11, size=N_ITEMS, dtype=np.int32),
    "amount":    RNG.uniform(1.0, 1000.0, size=N_ITEMS).astype(np.float64),
})

RUNS = 5

# ── 1. Native LazyFrame API ───────────────────────────────────────────────────
def query_lazyframe() -> pl.DataFrame:
    return (
        orders.filter(pl.col("status") == "A")
        .join(
            items.filter(pl.col("category") == 1),
            on="order_id",
        )
        .group_by("order_id", "region")
        .agg(pl.col("amount").sum())
        .collect()
    )


# ── 2. SQL interface ──────────────────────────────────────────────────────────
def query_sql() -> pl.DataFrame:
    ctx = pl.SQLContext(orders=orders, items=items)
    return ctx.execute("""
        SELECT o.order_id, o.region, SUM(i.amount) AS amount
        FROM orders o
        JOIN items i ON o.order_id = i.order_id
        WHERE o.status   = 'A'
          AND i.category = 1
        GROUP BY o.order_id, o.region
    """, eager=True)


# ── run benchmark ─────────────────────────────────────────────────────────────
def bench(fn, label):
    result = fn()  # warmup
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    best = min(times)
    print(f"{label:<20} mean={mean*1000:.1f}ms  best={best*1000:.1f}ms  runs={RUNS}")
    return result


print(f"Polars {pl.__version__}")
print(f"orders={N_ORDERS:,}  items={N_ITEMS:,}\n")
print(f"{'Interface':<20} Timing")
print("-" * 55)

r_lf  = bench(query_lazyframe, "LazyFrame API")
r_sql = bench(query_sql,       "SQL interface")

# ── verify results match ──────────────────────────────────────────────────────
print()
try:
    assert_frame_equal(
        r_lf.sort("order_id"),
        r_sql.sort("order_id"),
        check_column_order=False,
        abs_tol=1e-6,
    )
    print("Results match: YES")
except AssertionError as e:
    print(f"Results match: NO — {e}")

# ── compare optimized query plans ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTIMIZED QUERY PLANS")
print("=" * 60)

lf_plan = (
    orders.filter(pl.col("status") == "A")
    .join(items.filter(pl.col("category") == 1), on="order_id")
    .group_by("order_id", "region")
    .agg(pl.col("amount").sum())
)

ctx2 = pl.SQLContext(orders=orders, items=items)
sql_plan = ctx2.execute("""
    SELECT o.order_id, o.region, SUM(i.amount) AS amount
    FROM orders o
    JOIN items i ON o.order_id = i.order_id
    WHERE o.status   = 'A'
      AND i.category = 1
    GROUP BY o.order_id, o.region
""", eager=False)

print("\n--- LazyFrame API (optimized) ---")
print(lf_plan.explain(optimized=True))
print("\n--- SQL interface (optimized) ---")
print(sql_plan.explain(optimized=True))
