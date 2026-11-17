"""Step 1: Start Spark and load customers + sessions into DataFrames."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               DoubleType, TimestampType)

spark = (SparkSession.builder
         .appName("CartAbandonment")
         .master("local[*]")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# Explicit schema = faster and avoids silent type surprises on messy data
customer_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("customer_name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("city", StringType(), True),
])

session_schema = StructType([
    StructField("session_id", StringType(), False),
    StructField("customer_id", IntegerType(), True),
    StructField("session_start", TimestampType(), True),
    StructField("session_end", TimestampType(), True),
    StructField("pages_viewed", DoubleType(), True),
    StructField("product_views", IntegerType(), True),
    StructField("items_added", DoubleType(), True),
    StructField("items_removed", DoubleType(), True),
    StructField("cart_value", DoubleType(), True),
    StructField("checkout_started", DoubleType(), True),
    StructField("device", StringType(), True),
    StructField("traffic_source", StringType(), True),
    StructField("purchased", IntegerType(), True),
])

customers = spark.read.csv("customer_table.csv", header=True, schema=customer_schema)
sessions = spark.read.csv("sessions.csv", header=True, schema=session_schema)

# Join session data with customer demographics
df = sessions.join(customers, on="customer_id", how="left")

print("Customers:", customers.count())
print("Sessions :", sessions.count())
print("Joined   :", df.count())
df.printSchema()
df.show(5, truncate=False)

# Quick missing-value profile (this feeds Step 2)
df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).show(vertical=True)
print("Duplicate session_ids:", df.count() - df.dropDuplicates(["session_id"]).count())

# Saving via Spark's Parquet writer needs winutils on Windows, so we skip it.
# Later steps will simply continue from this DataFrame (or re-run this script).
print("Step 1 complete.")
