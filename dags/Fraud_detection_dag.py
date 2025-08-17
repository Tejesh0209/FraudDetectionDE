from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowException
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'Tejesh',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 14),
    'execution_timeout': timedelta(minutes=120),
    'max_active_runs': 1,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def _train_model(**context):
    """Airflow wrapper for training task"""
    import sys
    sys.path.append('/opt/airflow/dags')
    
    from fraud_detection_training import FraudDetectionTraining
    try:
        logger.info('Initializing fraud detection training')
        trainer = FraudDetectionTraining()
        model, precision = trainer.train_model()

        return {'status': 'success', 'precision': precision}
    except Exception as e:
        logger.error('Training failed: %s', str(e), exc_info=True)
        raise AirflowException(f'Model Training Failed: {str(e)}')

with DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Fraud detection model training pipeline',
    schedule_interval='0 3 * * *',
    catchup=False,
    tags=['Fraud', 'ML']
) as dag:

    validate_environment = BashOperator(
        task_id='validate_environment',
        bash_command='''
        echo "Validating environment..."
        test -f /app/config.yaml &&
        test -f /app/.env &&
        echo "Environment is valid"
        '''
    )

    training_task = PythonOperator(
        task_id='execute_training',
        python_callable=_train_model,
    )

    cleanup_task = BashOperator(
        task_id='cleanup_resources',
        bash_command='rm -f /app/tmp/*.pkl /app/*.png || true',
        trigger_rule='all_done'
    )

    validate_environment >> training_task >> cleanup_task

    dag.doc_md = """
    ## Fraud Detection Pipeline

    Daily Training of fraud detection model using:
    - Transaction data from Kafka
    - XGBoost classifier with precision optimization
    - MLflow for experiment tracking
    """