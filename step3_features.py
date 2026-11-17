"""Step 3: Behavioral feature engineering.
Run in the same pyspark shell right after Step 2:
    exec(open("step3_features.py").read())
"""
from pyspark.sql import functions as F

try:
    cart_sessions
except NameError:
    exec(open("step2_clean_data.py").read())

# ---------------------------------------------------------------
# 3.1 Session duration (seconds / minutes)
# ---------------------------------------------------------------
feat = cart_sessions.withColumn(
    "session_duration_sec",
    (F.unix_timestamp("session_end") - F.unix_timestamp("session_start")).cast("double"),
)

# Impute missing durations: pages_viewed x typical seconds-per-page
sec_per_page_df = (feat.filter(F.col("session_duration_sec").isNotNull() & (F.col("pages_viewed") > 0))
                       .withColumn("spp", F.col("session_duration_sec") / F.col("pages_viewed")))
median_spp = sec_per_page_df.approxQuantile("spp", [0.5], 0.01)[0]
print(f"3.1 median seconds per page = {median_spp:.1f}")

feat = feat.withColumn(
    "session_duration_sec",
    F.coalesce(F.col("session_duration_sec"),
               F.round(F.col("pages_viewed") * F.lit(median_spp))),
)
feat = feat.withColumn("session_duration_min", F.round(F.col("session_duration_sec") / 60, 2))

# ---------------------------------------------------------------
# 3.2 Cart behaviour features
# ---------------------------------------------------------------
feat = (feat
    .withColumn("net_items_in_cart", F.col("items_added") - F.col("items_removed"))
    .withColumn("removal_rate", F.round(F.col("items_removed") / F.col("items_added"), 3))
    .withColumn("avg_item_price", F.round(F.col("cart_value") / F.col("items_added"), 2))
    .withColumn("cart_to_view_ratio",
                F.round(F.col("items_added") / F.greatest(F.col("product_views").cast("double"), F.lit(1.0)), 3))
    .withColumn("pages_per_minute",
                F.round(F.col("pages_viewed") / F.greatest(F.col("session_duration_min"), F.lit(0.1)), 2))
)

# ---------------------------------------------------------------
# 3.3 Time-of-visit features
# ---------------------------------------------------------------
feat = (feat
    .withColumn("hour_of_day", F.hour("session_start"))
    .withColumn("day_of_week", F.dayofweek("session_start"))          # 1=Sun ... 7=Sat
    .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
)

# ---------------------------------------------------------------
# 3.4 Keep only model-ready columns (no IDs, names, or the raw `purchased` outcome)
# ---------------------------------------------------------------
numeric_features = [
    "session_duration_sec", "pages_viewed", "product_views", "items_added",
    "items_removed", "net_items_in_cart", "removal_rate", "cart_value",
    "avg_item_price", "cart_to_view_ratio", "pages_per_minute",
    "checkout_started", "hour_of_day", "day_of_week", "is_weekend", "age",
    "session_end_missing", "pages_viewed_missing", "items_added_missing",
    "items_removed_missing", "cart_value_missing", "checkout_started_missing",
]
categorical_features = ["device", "traffic_source", "gender", "city"]

model_df = feat.select(["session_id", "session_duration_min", *numeric_features,
                        *categorical_features, "abandoned"])
model_df.cache()

print(f"\nModel-ready rows: {model_df.count()}  |  "
      f"{len(numeric_features)} numeric + {len(categorical_features)} categorical features")
model_df.select("session_duration_min", "items_added", "cart_value",
                "removal_rate", "pages_per_minute").describe().show()
print("Average of key features: abandoned (1) vs purchased (0)")
model_df.groupBy("abandoned").agg(
    F.round(F.avg("session_duration_min"), 2).alias("avg_duration_min"),
    F.round(F.avg("items_added"), 2).alias("avg_items_added"),
    F.round(F.avg("removal_rate"), 3).alias("avg_removal_rate"),
    F.round(F.avg("cart_value"), 2).alias("avg_cart_value"),
).orderBy("abandoned").show()

print("Step 3 complete. DataFrame `model_df` is ready for Step 4.")
