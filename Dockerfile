FROM python:3.7
COPY docker_test.py /
COPY database.ini /
RUN pip install pymongo
RUN pip install configparser
RUN pip install dnspython

CMD ["python", "./docker_test.py"]