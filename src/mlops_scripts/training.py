# Production Training Code - Approach 1: Code Promotion
# This code is promoted from development to production workspace

# !pip install --upgrade typing_extensions mlflow
# dbutils.library.restartPython()

import mlflow

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pandas as pd

def train_production_model():
    """Production model training function"""
    
    # Load data from production catalog
    df = spark.table("unum_prod_catalog_demo.churn_project.customer_churn").toPandas()
    mlflow.set_experiment("/Shared/mlops_demo/churn_prediction")
    mlflow.set_registry_uri('databricks-uc')
    # Feature engineering (same logic as dev)
    df_processed = df.copy()
    df_processed['charges_per_month'] = df_processed['total_charges'] / (df_processed['tenure'] + 1)
    df_processed['is_new_customer'] = (df_processed['tenure'] <= 6).astype(int)
    df_processed['high_value_customer'] = (df_processed['monthly_charges'] > df_processed['monthly_charges'].median()).astype(int)
    
    le_contract = LabelEncoder()
    le_payment = LabelEncoder()
    df_processed['contract_type_encoded'] = le_contract.fit_transform(df_processed['contract_type'])
    df_processed['payment_method_encoded'] = le_payment.fit_transform(df_processed['payment_method'])
    
    feature_columns = [
        'tenure', 'monthly_charges', 'total_charges', 'charges_per_month',
        'is_new_customer', 'high_value_customer', 'contract_type_encoded', 'payment_method_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['churn']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    mlflow.set_registry_uri('databricks-uc')
    with mlflow.start_run(run_name="churn_model_prod") as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Register model to Unity Catalog
        model_name = "churn_model"
        register_model_name = "unum_prod_catalog_demo.churn_project.churn_model"

        model_info=mlflow.sklearn.log_model(
            pipeline, 
            "model",
            registered_model_name=register_model_name,
            input_example=X_train.iloc[:5],
        )
        client = mlflow.MlflowClient()
        client.update_model_version(
        name=register_model_name,
        version=model_info.registered_model_version,
        description="Customer churn prediction model using logistic regression with feature engineering including charges per month, customer tenure, and contract type encoding."
        )
        client.set_registered_model_alias(
            name=register_model_name,
            alias="Challenger",  # or "Champion", "Staging", etc.
            version=model_info.registered_model_version,
        )
        
        return run.info.run_id, model_name

# Execute training
run_id, model_name = train_production_model()
