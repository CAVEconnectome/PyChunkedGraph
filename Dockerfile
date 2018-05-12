FROM python:3.4
ADD . /code
WORKDIR /code/src/master
RUN pip install -r requirements.txt
#RUN pip install -r /code/src/pychunkedgraph/requirements.txt
CMD ["python", "FlaskServer.py"]