FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY requirements.txt /app/.
RUN pip install numpy
RUN pip install -r requirements.txt
COPY . /app
