FROM python:3.6.10

# Create an application directory -> added
RUN mkdir -p /app

ENV FLASK_APP=/app/REST_AL.py
ENV IS_DOCKER Yes

WORKDIR /app

COPY requirements.txt .

RUN pip install -r /app/requirements.txt

# expose port -> added
EXPOSE 5000

COPY . /app

#ENTRYPOINT flask run --host=0.0.0.0 
#changed:
CMD flask run --host=0.0.0.0 

