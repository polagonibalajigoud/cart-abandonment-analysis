"""Step 4: Build Decision Tree classifiers to predict cart abandonment.
Run in the same pyspark shell right after Step 3:
    exec(open("step4_decision_tree.py").read())
"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

try:
    model_df
except NameError:
    exec(open("step3_features.py").read())

LABEL = "abandoned"
categorical_cols = ["device", "traffic_source", "gender", "city"]
all_numeric = [
    "session_duration_sec", "pages_viewed", "product_views", "items_added",
    "items_removed", "net_items_in_cart", "removal_rate", "cart_value",
    "avg_item_price", "cart_to_view_ratio", "pages_per_minute",
    "checkout_started", "hour_of_day", "day_of_week", "is_weekend", "age",
    "session_end_missing", "pages_viewed_missing", "items_added_missing",
    "items_removed_missing", "cart_value_missing", "checkout_started_missing",
]

# ---------------------------------------------------------------
# 4.1 Train / test split (80 / 20, reproducible)
# ---------------------------------------------------------------
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)
train_df.cache(); test_df.cache()
print(f"4.1 Train rows: {train_df.count()}  |  Test rows: {test_df.count()}")


def build_pipeline(numeric_cols):
    """Index + one-hot the categoricals, assemble features, add a Decision Tree."""
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in categorical_cols]
    encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in categorical_cols],
                            outputCols=[f"{c}_vec" for c in categorical_cols])
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_vec" for c in categorical_cols],
        outputCol="features", handleInvalid="keep")
    tree = DecisionTreeClassifier(
        labelCol=LABEL, featuresCol="features",
        maxDepth=5, minInstancesPerNode=20, seed=42)
    return Pipeline(stages=[*indexers, encoder, assembler, tree])


def feature_names(predictions_df):
    """Recover readable names for every slot of the assembled feature vector."""
    attrs = predictions_df.schema["features"].metadata["ml_attr"]["attrs"]
    flat = [a for group in attrs.values() for a in group]
    return [a["name"] for a in sorted(flat, key=lambda a: a["idx"])]


def show_importances(name, model, preds, top=8):
    tree = model.stages[-1]
    names = feature_names(preds)
    imps = sorted(zip(names, tree.featureImportances.toArray()), key=lambda x: -x[1])[:top]
    print(f"\n{name}: depth={tree.depth}, nodes={tree.numNodes}")
    print("Top features:")
    for n, v in imps:
        print(f"   {n:28s} {v:.3f}")


# ---------------------------------------------------------------
# 4.2 Model A: all features (includes checkout_started)
# ---------------------------------------------------------------
print("\n4.2 Training Model A (all features)...")
model_a = build_pipeline(all_numeric).fit(train_df)
preds_a = model_a.transform(test_df)
show_importances("Model A", model_a, preds_a)

# ---------------------------------------------------------------
# 4.3 Model B: browsing behaviour only (no checkout signal)
#     Answers: can we predict abandonment BEFORE the shopper reaches checkout?
# ---------------------------------------------------------------
browse_numeric = [c for c in all_numeric
                  if c not in ("checkout_started", "checkout_started_missing")]
print("\n4.3 Training Model B (browsing behaviour only)...")
model_b = build_pipeline(browse_numeric).fit(train_df)
preds_b = model_b.transform(test_df)
show_importances("Model B", model_b, preds_b)

# Print a readable version of Model A's rules (first lines)
print("\nModel A decision rules (top of the tree):")
print("\n".join(model_a.stages[-1].toDebugString.split("\n")[:15]))

print("\nStep 4 complete. `preds_a` and `preds_b` hold test-set predictions for Step 5.")
