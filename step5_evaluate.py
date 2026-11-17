"""Step 5: Evaluate the Decision Tree models (accuracy, recall, and more).
Run in the same pyspark shell right after Step 4:
    exec(open("step5_evaluate.py").read())
"""
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

try:
    preds_a, preds_b
except NameError:
    exec(open("step4_decision_tree.py").read())


def evaluate(name, preds):
    mc = lambda metric, **kw: MulticlassClassificationEvaluator(
        labelCol="abandoned", predictionCol="prediction", metricName=metric, **kw).evaluate(preds)

    accuracy  = mc("accuracy")
    recall    = mc("recallByLabel", metricLabel=1.0)       # of real abandoners, how many did we catch?
    precision = mc("precisionByLabel", metricLabel=1.0)    # of flagged shoppers, how many truly abandoned?
    f1        = mc("fMeasureByLabel", metricLabel=1.0)
    auc       = BinaryClassificationEvaluator(
        labelCol="abandoned", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds)

    print(f"\n==================  {name}  ==================")
    print(f"Accuracy                     : {accuracy:.3f}")
    print(f"Recall    (abandoned = 1)    : {recall:.3f}   <- share of real abandoners we caught")
    print(f"Precision (abandoned = 1)    : {precision:.3f}")
    print(f"F1 score  (abandoned = 1)    : {f1:.3f}")
    print(f"AUC (ROC)                    : {auc:.3f}")

    print("Confusion matrix (rows = actual, columns = predicted):")
    (preds.groupBy("abandoned").pivot("prediction", [0.0, 1.0]).count()
          .fillna(0).orderBy("abandoned")
          .withColumnRenamed("0.0", "pred_purchased")
          .withColumnRenamed("1.0", "pred_abandoned").show())
    return accuracy, recall


# Baseline: always predict the majority class in the test set
test_n = preds_a.count()
majority = preds_a.groupBy("abandoned").count().orderBy(F.desc("count")).first()
print(f"Baseline (always guess the majority class): accuracy = {majority['count'] / test_n:.3f}")

acc_a, rec_a = evaluate("Model A: all features", preds_a)
acc_b, rec_b = evaluate("Model B: browsing behaviour only", preds_b)

print("\nSummary")
print(f"{'Model':36s}{'Accuracy':>10s}{'Recall':>10s}")
print(f"{'A: all features':36s}{acc_a:>10.3f}{rec_a:>10.3f}")
print(f"{'B: browsing only (pre-checkout)':36s}{acc_b:>10.3f}{rec_b:>10.3f}")
print("\nStep 5 complete.")
