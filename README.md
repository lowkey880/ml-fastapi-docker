# ML FastAPI Docker

A Random Forest classifier trained on the Iris dataset, served with FastAPI and packaged with Docker.

## Run locally

```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

## Run with Docker

```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```

## Endpoints

- `GET /` — health check, returns `{"message": "ML API is running"}`
- `POST /predict` — returns the predicted Iris class

Example body:

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```
