import os
import json
import logging
import boto3
from dotenv import load_dotenv
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
import yaml
import pandas as pd
from kafka import KafkaConsumer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import numpy as np
import mlflow
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, classification_report, f1_score, fbeta_score, make_scorer, precision_recall_curve, precision_score, recall_score, roc_auc_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FraudDetectionTraining:
    def __init__(self, config_path='/app/config.yaml'):
        # env hygiene
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        # load .env first so we can reference values
        load_dotenv(dotenv_path='/app/.env')

        self.config = self._load_config(config_path)

        # propagate AWS -> boto3 & MLflow s3 endpoint
        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID', ''),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow'].get('s3_endpoint_url', '')
        })

        self._validate_environment()

        # MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info('Configuration loaded successfully')
            # sanity defaults
            config.setdefault('kafka', {})
            config.setdefault('mlflow', {})
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _validate_environment(self):
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f'Missing required environment variables: {missing}')
        self._check_minio_connection()

    def _check_minio_connection(self):
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow'].get('s3_endpoint_url'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('MinIO connection verified. Buckets: %s', bucket_names)

            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')
            if mlflow_bucket and mlflow_bucket not in bucket_names:
                try:
                    s3.create_bucket(Bucket=mlflow_bucket)
                    logger.info('Created missing MLflow bucket: %s', mlflow_bucket)
                except Exception as e:
                    # ignore race if created by another service
                    logger.warning('Bucket create skipped (%s): %s', mlflow_bucket, e)
        except Exception as e:
            logger.error('MinIO connection failed: %s', str(e))

    def read_from_kafka(self) -> pd.DataFrame:
        """Read JSON messages from Kafka topic -> DataFrame"""
        consumer = None
        try:
            kcfg = self.config['kafka']
            topic = kcfg['topic']
            logger.info('Connecting to Kafka topic: %s', topic)

            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=kcfg.get('bootstrap_servers') or os.getenv('KAFKA_BOOTSTRAP_SERVERS', '').split(','),
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username=os.getenv('KAFKA_USERNAME'),
                sasl_plain_password=os.getenv('KAFKA_PASSWORD'),
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset=kcfg.get('auto_offset_reset', 'earliest'),
                enable_auto_commit=False,
                consumer_timeout_ms=int(kcfg.get('timeout', 10000))
            )

            messages = [msg.value for msg in consumer]
            if consumer is not None:
                consumer.close()

            if not messages:
                raise ValueError('No messages received from Kafka')

            df = pd.DataFrame(messages)
            # required columns sanity
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            else:
                raise ValueError('Missing "timestamp" in Kafka data')

            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label "is_fraud" is missing from Kafka data')

            fraud_rate = float(df['is_fraud'].mean() * 100.0)
            logger.info(f'Kafka data read successfully. Fraud rate: {fraud_rate:.2f}%')

            return df

        except Exception as e:
            logger.error('Failed to read data from Kafka: %s', str(e), exc_info=True)
            raise
        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for downstream model training"""
        # sorting
        df = df.sort_values(['user_id', 'timestamp']).copy()

        # ---- Temporal features ----
        df['transaction_hour'] = df['timestamp'].dt.hour
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] < 5)).astype(int)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['transaction_day'] = df['timestamp'].dt.day

        # ---- Behavior features (24h rolling count per user) ----
        df['user_activity_24h'] = (
            df.groupby('user_id', group_keys=False)
              .apply(lambda g: g.set_index('timestamp')['amount']
                                .rolling('24h', closed='left')
                                .count()
                                .reindex(g['timestamp'])
                                .reset_index(drop=True))
              .fillna(0)
              .astype(float)
        )

        # ---- Monetary velocity (amount / 7-step rolling mean) ----
        df['amount_to_avg_ratio'] = (
            df.groupby('user_id', group_keys=False)['amount']
              .apply(lambda s: (s / s.rolling(7, min_periods=1).mean()).fillna(1.0))
              .astype(float)
        )

        # ---- Merchant risk flag ----
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df['merchant_risk'] = df['merchant'].isin(high_risk_merchants).astype(int)

        # ---- Categorical encoding (simple label encode for merchant) ----
        if 'merchant' in df.columns:
            df['merchant'] = df['merchant'].astype('category').cat.codes

        feature_cols = [
            'amount', 'is_night', 'is_weekend', 'transaction_day',
            'user_activity_24h', 'amount_to_avg_ratio', 'merchant_risk', 'merchant'
        ]
        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column "is_fraud"')

        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            raise ValueError(f'Missing feature columns after FE: {missing_feats}')

        return df[feature_cols + ['is_fraud']]
    

    def train_model(self):
        try:
            logger.info('Starting the model training process')
            
            raw_df = self.read_from_kafka()
            data = self.create_features(raw_df)

            # Split features & labels
            X = data.drop(columns=['is_fraud'])
            y = data['is_fraud'].astype(int)

            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
            
            if y.sum() < 10:
                logger.warning('Low positive samples: %d. Consider additional data augmentation', y.sum())

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['model'].get('test_size', 0.2),
                stratify=y,
                random_state=self.config['model'].get('seed', 42)
            )

            with mlflow.start_run():
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })

                # Preprocessing: encode 'merchant' column
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('merchant_encoder', OrdinalEncoder(
                            handle_unknown='use_encoded_value', 
                            unknown_value=-1, 
                            dtype=np.float32
                        ), ['merchant'])
                    ],
                    remainder='passthrough'
                )

                # XGBoost model
                xgb = XGBClassifier(
                    eval_metric='aucpr',
                    random_state=self.config['model'].get('seed', 42),
                    reg_lambda=1.0,
                    n_estimators=self.config['model']['param']['n_estimators'],
                    n_jobs=-1,
                    tree_method=self.config['model'].get('tree_method', 'hist')
                )

                # Pipeline with SMOTE to handle imbalance
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=self.config['model'].get('seed', 42))),
                    ('classifier', xgb)
                ], memory='./cache')

                param_dist = {
                    'classifier__max_depth' : [3,5,7],
                    'classifier__learning_rate': [0.01,0.05,0.1],
                    'classifier__subsample': [0.6,0.8,1.0],
                    'classifier__colsample_bytree': [0.6,0.8,1.0],
                    'classifier__gamma': [0,0.1,0.3],
                    'classifier__reg_alpha': [0,0.1,0.5]
                }

                searcher = RandomizedSearchCV(
                    pipeline,
                    param_dist,
                    n_iter=20,
                    scoring=make_scorer(fbeta_score, beta=2, zer_division=0),
                    cv=StratifiedKFold(n_splits=3, shuffle=True),
                    n_jobs=-1,
                    refit=True,
                    error_score='raise',
                    random_state=self.config['model'].get('seed',42)
                )

                logger.info('Starting hyperparameter tuning')
                searcher.fit(X_train,y_train)
                best_model = searcher.best_estimator_
                best_params = searcher.best_params_
                logger.info('Best hyperparameters: %s', best_params)

                train_proba = best_model.predict_proba(X_train)[:,-1]
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)
                f1_scores = [2 * (p * r) / (p +r) if (p + r)> 0 else 0 for p, r in zip(precision_arr[:-1], recall_arr[:-1])]
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info('Optimal threshold determined: %.4f', best_threshold)

                X_test_processed = best_model.named_steps=['preprocessor'].transform(X_test)

                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:,1]

                y_pred  = (test_proba >= best_threshold).astype(int)

                metrics = {
                    'auc_pr': float(average_precision_score(y_test,test_proba)),
                    'precision': float(precision_score(y_test,y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test,y_pred,zero_division=0)),
                    'f1': float(f1_score(y_test,y_pred,zero_division=0)),
                    'threshold': float(best_threshold)
                }

                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)

        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise




