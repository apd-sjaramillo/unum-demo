import mlflow
from mlflow import MlflowClient
from pyspark.sql.functions import col

validation_dataset = "unum_prod_catalog_demo.churn_project.customer_churn"
model_name = "unum_prod_catalog_demo.churn_project.churn_model"
model_alias = "Challenger"

print(f"🧪 Validating Model: {model_name}@{model_alias}")
print(f"📊 Validation Dataset: {validation_dataset}")

client = MlflowClient()
mlflow.set_registry_uri('databricks-uc')

## Load Data and Model

# Read validation dataset
df = spark.read.table(validation_dataset)
print(f"✅ Loaded validation data: {df.count()} rows")
# Load model
model_details = client.get_model_version_by_alias(model_name, model_alias)
print(f"✅ Loaded model: {model_name} v{model_details.version}")

## Test 1: Performance Metric Check
# Get F1 score from training run
model_run_id = model_details.run_id
f1_score = mlflow.get_run(model_run_id).data.metrics['f1_score']
try:
    # Compare with Champion if exists
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics['f1_score']
    print(f'Champion F1: {champion_f1:.3f}, Challenger F1: {f1_score:.3f}')
    metric_f1_passed = f1_score >= champion_f1
except:
    print(f"No Champion found. Accept model as first one. F1: {f1_score:.3f}")
    metric_f1_passed = True

print(f'✅ F1 metric test: {"PASSED" if metric_f1_passed else "FAILED"}')
client.set_model_version_tag(name=model_name, version=model_details.version, key="metric_f1_passed", value=metric_f1_passed)

## Test 2: Description Check
# Check model has proper description
if not model_details.description:
    has_description = False
    print("❌ Please add model description")
elif len(model_details.description) <= 20:
    has_description = False
    print("❌ Please add detailed model description (20+ chars)")
else:
    has_description = True
    print(f"✅ Model has description: {len(model_details.description)} chars")

client.set_model_version_tag(name=model_name, version=model_details.version, key="has_description", value=has_description)

## Test 3: Validation Dataset Performance and checking compatibility with PySpark
# Run model on validation dataset
model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}@{model_alias}")

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# Feature engineering
df_processed = df.withColumn(
    'charges_per_month', 
    F.col('total_charges') / (F.col('tenure') + 1)
).withColumn(
    'is_new_customer',
    F.when(F.col('tenure') <= 6, 1).otherwise(0).cast(IntegerType())
)

# Calculate median for high_value_customer feature
monthly_charges_median = df_processed.approxQuantile('monthly_charges', [0.5], 0.25)[0]

df_processed = df_processed.withColumn(
    'high_value_customer',
    F.when(F.col('monthly_charges') > monthly_charges_median, 1).otherwise(0).cast(IntegerType())
)

# Manual label encoding using mapping
# Get distinct values
contract_types = [row.contract_type for row in df_processed.select('contract_type').distinct().collect()]
payment_methods = [row.payment_method for row in df_processed.select('payment_method').distinct().collect()]

# Create mapping expressions
contract_mapping = F.create_map([F.lit(x) for pair in zip(contract_types, range(len(contract_types))) for x in pair])
payment_mapping = F.create_map([F.lit(x) for pair in zip(payment_methods, range(len(payment_methods))) for x in pair])

df_processed = df_processed.withColumn('contract_type_encoded', contract_mapping[F.col('contract_type')].cast(IntegerType()))
df_processed = df_processed.withColumn('payment_method_encoded', payment_mapping[F.col('payment_method')].cast(IntegerType()))

# Select feature columns
feature_columns = [
    'tenure', 'monthly_charges', 'total_charges', 'charges_per_month',
    'is_new_customer', 'high_value_customer', 'contract_type_encoded', 'payment_method_encoded'
]

X = df_processed.select(*feature_columns)

preds_df = X.withColumn('predictions', model_udf(*model_udf.metadata.get_input_schema().input_names()))

# Calculate accuracy
matching_rows = preds_df.filter(col("churn") == col("predictions")).count()
total_rows = preds_df.count()
accuracy = matching_rows / total_rows if total_rows > 0 else 0

print(f'✅ Validation accuracy: {accuracy:.2%}')
validation_passed = accuracy >= 0.65  # 65% threshold

client.set_model_version_tag(name=model_name, version=model_details.version, key="validation_accuracy", value=round(accuracy, 2))
client.set_model_version_tag(name=model_name, version=model_details.version, key="validation_passed", value=validation_passed)

## Test 4: Output Data Type Check
# Check predictions are valid (0/1 or boolean)
invalid_rows = preds_df.filter(
    (col("predictions").isNull()) | 
    ~((col("predictions") == 0) | (col("predictions") == 1))
)

invalid_count = invalid_rows.count()
has_valid_values = invalid_count == 0

if has_valid_values:
    print("✅ All predictions are valid (0/1)")
else:
    print(f"❌ Found {invalid_count} invalid prediction values")

client.set_model_version_tag(name=model_name, version=model_details.version, key="has_valid_values", value=has_valid_values)

## Overall Validation Result
# Check all tests
all_tests = [
    ("F1 Performance", metric_f1_passed),
    ("Description", has_description), 
    ("Validation Accuracy", validation_passed),
    ("Valid Outputs", has_valid_values)
]

print("📋 VALIDATION SUMMARY")
print("=" * 30)
for test_name, result in all_tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{test_name}: {status}")

# Overall result
overall_passed = all(result for _, result in all_tests)
print(f"\n🎯 Overall Result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")

# Tag overall result
client.set_model_version_tag(name=model_name, version=model_details.version, key="validation_overall", value=overall_passed)

if not overall_passed:
    raise Exception("Model did not pass the tests")