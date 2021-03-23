FROM python:3.7
EXPOSE 8080
COPY st_season_team_profile.py /
COPY docker_database.ini /
COPY db.py /
COPY st_functions.py /

RUN pip install pymongo
RUN pip install configparser
RUN pip install dnspython
RUN pip install streamlit

CMD streamlit run st_season_team_profile.py --server.port 8080 --server.enableCORS false