"""Step 6: Visualize cart abandonment rate by time spent on the website.
Run in the same pyspark shell (needs `model_df` from Step 3):
    exec(open("step6_visualize.py").read())
If matplotlib is missing, first run in a normal Command Prompt:  python -m pip install matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

try:
    model_df
except NameError:
    exec(open("step3_features.py").read())

# ---------------------------------------------------------------
# 6.1 Bucket sessions by time spent (minutes), aggregate in Spark
# ---------------------------------------------------------------
d = F.col("session_duration_min")
bucketed = model_df.withColumn(
    "time_bucket",
    F.when(d < 1, "0-1")
     .when(d < 2, "1-2")
     .when(d < 5, "2-5")
     .when(d < 10, "5-10")
     .when(d < 20, "10-20")
     .otherwise("20+"),
)

summary = (bucketed.groupBy("time_bucket")
           .agg(F.count("*").alias("sessions"),
                F.sum("abandoned").alias("abandoned_sessions"),
                F.round(F.avg("abandoned") * 100, 1).alias("abandonment_rate_pct")))

# The result is tiny (6 rows), so collect and build a pandas frame by hand
order = ["0-1", "1-2", "2-5", "5-10", "10-20", "20+"]
rows = [r.asDict() for r in summary.collect()]
pdf = pd.DataFrame(rows).set_index("time_bucket").reindex(order).reset_index()
print(pdf.to_string(index=False))

overall = model_df.agg(F.avg("abandoned")).first()[0] * 100

# ---------------------------------------------------------------
# 6.2 Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(pdf["time_bucket"], pdf["abandonment_rate_pct"],
              color="#2a6f97", width=0.65)
ax.axhline(overall, color="#c1121f", linestyle="--", linewidth=1.5,
           label=f"Overall abandonment rate ({overall:.1f}%)")

for bar, rate, n in zip(bars, pdf["abandonment_rate_pct"], pdf["sessions"]):
    ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.8, f"{rate:.1f}%",
            ha="center", fontweight="bold")
    ax.text(bar.get_x() + bar.get_width() / 2, 2, f"n={int(n):,}",
            ha="center", color="white", fontsize=9)

ax.set_title("Cart abandonment rate by time spent on the website", fontsize=14, pad=12)
ax.set_xlabel("Time spent in session (minutes)")
ax.set_ylabel("Abandonment rate (%)")
ax.set_ylim(0, max(pdf["abandonment_rate_pct"].max() + 10, 70))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("abandonment_by_time_spent.png", dpi=150)
print("Saved chart to abandonment_by_time_spent.png")
plt.show()

print("\nStep 6 complete. Project finished!")
