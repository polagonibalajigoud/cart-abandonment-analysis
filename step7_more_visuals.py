"""Step 7: More visualizations (behaviour dashboard + model diagnostics).
Needs viz_helpers.py in the same folder. Run in the pyspark shell after Steps 3-4:
    exec(open("step7_more_visuals.py").read())
"""
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import viz_helpers as vz

try:
    model_df
except NameError:
    exec(open("step3_features.py").read())


def rate_frame(sdf, label_col, order=None, sort_by_rate=False):
    rows = (sdf.groupBy(label_col)
               .agg(F.count("*").alias("sessions"), (F.avg("abandoned") * 100).alias("rate"))
               .collect())
    pdf = pd.DataFrame([r.asDict() for r in rows]).rename(columns={label_col: "label"})
    if order:
        pdf = pdf.set_index("label").reindex(order).dropna().reset_index()
    elif sort_by_rate:
        pdf = pdf.sort_values("rate", ascending=False)
    else:
        pdf = pdf.sort_values("label")
    return pdf


overall = model_df.agg(F.avg("abandoned")).first()[0] * 100

# ---------- 7.1 behaviour dashboard ----------
items = model_df.withColumn("g", F.when(F.col("items_added") >= 6, "6+")
                                   .otherwise(F.col("items_added").cast("int").cast("string")))
val = F.col("cart_value")
value = model_df.withColumn("g", F.when(val < 50, "<50").when(val < 100, "50-100")
                                   .when(val < 200, "100-200").when(val < 300, "200-300")
                                   .otherwise("300+"))
checkout = model_df.withColumn("g", F.when(F.col("checkout_started") == 1, "Started")
                                      .when(F.col("checkout_started") == 0, "Not started")
                                      .otherwise("Unknown"))
frames = {
    "device":   rate_frame(model_df, "device", sort_by_rate=True),
    "source":   rate_frame(model_df, "traffic_source", sort_by_rate=True),
    "items":    rate_frame(items, "g", order=["1", "2", "3", "4", "5", "6+"]),
    "value":    rate_frame(value, "g", order=["<50", "50-100", "100-200", "200-300", "300+"]),
    "hour":     rate_frame(model_df, "hour_of_day"),
    "checkout": rate_frame(checkout, "g", order=["Not started", "Started", "Unknown"]),
}
vz.behaviour_dashboard(frames, overall, "abandonment_dashboard.png")
print("Saved abandonment_dashboard.png")


# ---------- 7.2 model diagnostics (needs Steps 4 to have run in this shell) ----------
def confusion(preds):
    c = {(r["abandoned"], r["prediction"]): r["count"]
         for r in preds.groupBy("abandoned", "prediction").count().collect()}
    return [[c.get((0, 0.0), 0), c.get((0, 1.0), 0)],
            [c.get((1, 0.0), 0), c.get((1, 1.0), 0)]]

try:
    imp_a = (feature_names(preds_a), model_a.stages[-1].featureImportances.toArray())
    imp_b = (feature_names(preds_b), model_b.stages[-1].featureImportances.toArray())
    vz.model_panel(confusion(preds_a), confusion(preds_b), imp_a, imp_b, "model_diagnostics.png")
    print("Saved model_diagnostics.png")
except NameError:
    print("Skipped model diagnostics: run Step 4 first (preds_a / model_a not found).")

plt.show()
print("Step 7 complete.")
