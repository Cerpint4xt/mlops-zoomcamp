FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
RUN mkdir -p output/yellow
COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install  --system --deploy

COPY ["starter.py", "./"]

ENTRYPOINT [ "python", "starter.py", "yellow", "2023", "05" ]