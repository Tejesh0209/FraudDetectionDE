FROM python:3.10.16-slim-bookworm

WORKDIR /app
COPY requirements.txt /app/requirements.txt

#Install Dependencies
RUN pip install -r /app/requirements.txt

#copy the rest of the application
COPY . . 

#Set the command to run your application
EXPOSE 5500
