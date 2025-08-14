FROM apache/airflow:2.10.5

#create models directiry and set ownership
USER root
RUN mkdir -p /app/models

USER airflow
#copy requirements file
COPY requirements.txt /tmp/requirements.txt

#install python packages as root
RUN pip install -r /tmp/requirements.txt

#Switch back to airflow for safety
USER airflow

