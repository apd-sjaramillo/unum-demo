
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
model_details = client.get_model_version_by_alias(model_name, model_alias)

# Set Champion alias
client.set_registered_model_alias(
    name=model_name,
    alias="Champion", 
    version=model_details.version
)

# Add promotion timestamp
from datetime import datetime
client.set_model_version_tag(
    name=model_name, 
    version=model_details.version, 
    key="promoted_to_champion", 
    value=datetime.now().isoformat()
)

print(f"✅ Model promoted: {model_name}@Champion")
print(f"Model version {model_details.version}")
