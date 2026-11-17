"""Step 2: Clean incomplete cart-session data and define the abandonment label.
Run in the same pyspark shell right after Step 1:
    exec(open("step2_clean_data.py").read())
"""
from pyspark.sql import functions as F

# Reuse `df` from Step 1 (load it if this is a fresh shell)
try:
    df
except NameError:
    exec(open("step1_load_data.py").read())

n_raw = df.count()

# ---------------------------------------------------------------
# 2.1 Remove duplicate sessions
# ---------------------------------------------------------------
df = df.dropDuplicates(["session_id"])
n_dedup = df.count()
print(f"2.1 Duplicates removed: {n_raw - n_dedup}")

# ---------------------------------------------------------------
# 2.2 Invalid timestamps: end before start -> treat end as missing
# ---------------------------------------------------------------
bad_ts = df.filter(F.col("session_end") < F.col("session_start")).count()
df = df.withColumn(
    "session_end",
    F.when(F.col("session_end") < F.col("session_start"), None).otherwise(F.col("session_end")),
)
print(f"2.2 Invalid end-times set to NULL: {bad_ts}")

# ---------------------------------------------------------------
# 2.3 Missing-action flags (missingness itself can be a signal)
# ---------------------------------------------------------------
for c in ["session_end", "pages_viewed", "items_added", "items_removed",
          "cart_value", "checkout_started"]:
    df = df.withColumn(f"{c}_missing", F.col(c).isNull().cast("int"))

# ---------------------------------------------------------------
# 2.4 Statistics for imputation (computed only from complete rows)
# ---------------------------------------------------------------
cart_rows = df.filter(F.col("items_added") > 0)
median_items = cart_rows.approxQuantile("items_added", [0.5], 0.01)[0]

unit_price_df = (df.filter((F.col("items_added") > 0) & F.col("cart_value").isNotNull())
                   .withColumn("unit_price", F.col("cart_value") / F.col("items_added")))
median_unit_price = unit_price_df.approxQuantile("unit_price", [0.5], 0.01)[0]

ratio_df = (df.filter(F.col("pages_viewed").isNotNull() & (F.col("pages_viewed") > 0))
              .withColumn("view_ratio", F.col("product_views") / F.col("pages_viewed")))
median_view_ratio = ratio_df.approxQuantile("view_ratio", [0.5], 0.01)[0]

print(f"2.4 median items (cart sessions)={median_items}, "
      f"median unit price={median_unit_price:.2f}, median product/page ratio={median_view_ratio:.2f}")

# ---------------------------------------------------------------
# 2.5 Drop sessions where we cannot tell if a cart even existed
# ---------------------------------------------------------------
undeterminable = (F.col("items_added").isNull()
                  & F.col("cart_value").isNull()
                  & F.col("checkout_started").isNull())
n_undet = df.filter(undeterminable).count()
df = df.filter(~undeterminable)
print(f"2.5 Dropped (no way to tell if cart existed): {n_undet}")

# ---------------------------------------------------------------
# 2.6 Impute items_added from the other cart signals
# ---------------------------------------------------------------
df = df.withColumn(
    "items_added",
    F.when(F.col("items_added").isNotNull(), F.col("items_added"))
     .when(F.col("cart_value") > 0, F.lit(median_items))
     .when(F.col("cart_value") == 0, F.lit(0.0))
     .when(F.col("checkout_started") == 1, F.lit(median_items))   # cart_value missing
     .otherwise(F.lit(0.0)),
)

# ---------------------------------------------------------------
# 2.7 Impute cart_value from items_added x typical unit price
# ---------------------------------------------------------------
df = df.withColumn(
    "cart_value",
    F.when(F.col("cart_value").isNotNull(), F.col("cart_value"))
     .when(F.col("items_added") == 0, F.lit(0.0))
     .otherwise(F.round(F.col("items_added") * F.lit(median_unit_price), 2)),
)

# ---------------------------------------------------------------
# 2.8 Remaining columns
# ---------------------------------------------------------------
df = (df
      # no removal observed -> assume 0
      .withColumn("items_removed", F.coalesce(F.col("items_removed"), F.lit(0.0)))
      # estimate page count from product views
      .withColumn("pages_viewed",
                  F.coalesce(F.col("pages_viewed"),
                             F.greatest(F.col("product_views").cast("double"),
                                        F.round(F.col("product_views") / F.lit(median_view_ratio)))))
      # checkout: 0 if no cart, otherwise -1 = unknown (tree can split on it;
      # filling 0 would wrongly label unknowns as "never started checkout")
      .withColumn("checkout_started",
                  F.when(F.col("checkout_started").isNotNull(), F.col("checkout_started"))
                   .when(F.col("items_added") == 0, F.lit(0.0))
                   .otherwise(F.lit(-1.0)))
      .fillna({"device": "unknown", "traffic_source": "unknown"}))

# ---------------------------------------------------------------
# 2.9 Keep cart sessions only and define the label
#     abandoned = 1 -> items in cart but no purchase
# ---------------------------------------------------------------
cart_sessions = (df.filter(F.col("items_added") > 0)
                   .withColumn("abandoned", (F.col("purchased") == 0).cast("int")))

print("\n--- Remaining nulls (session_end is handled in Step 3 with duration) ---")
cart_sessions.select([F.sum(F.col(c).isNull().cast("int")).alias(c)
                      for c in cart_sessions.columns
                      if not c.endswith("_missing")]).show(vertical=True)

total = cart_sessions.count()
print(f"Sessions after cleaning: {df.count()}  |  cart sessions: {total}")
cart_sessions.groupBy("abandoned").count().orderBy("abandoned").show()

cart_sessions.cache()
print("Step 2 complete. DataFrame `cart_sessions` is ready for Step 3.")
