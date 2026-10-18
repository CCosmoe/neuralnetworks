# Deploying the model to Render

The NumPy network is trained once (`train.py`), its weights are saved to `model/`, and a FastAPI
service (`api/`) loads them and serves predictions. The service is packaged with Docker and runs
on Render.

```
train.py ──> model/spiral_model.npz ──> api/ (FastAPI) ──> Dockerfile ──> Render builds + runs it (public HTTPS URL)
```

## API

| Method | Path          | Description                                     |
|--------|---------------|-------------------------------------------------|
| GET    | `/health`     | Health check                                    |
| GET    | `/model-info` | Architecture and training metrics               |
| POST   | `/predict`    | Classify points (1 to 1000 per request)         |
| GET    | `/docs`       | Interactive Swagger UI                          |

```bash
curl -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"points": [[0.3, -0.5], [-0.6, 0.2]]}'
```

## 1. Run it locally

```bash
pip install -r requirements-dev.txt
python train.py                    # optional: retrains and overwrites model/
pytest
uvicorn api.main:app --reload      # http://localhost:8000/docs
```

With Docker:

```bash
docker build -t spiral-api .
docker run -p 8080:8080 spiral-api  # http://localhost:8080/docs
```

## 2. Deploy on Render

1. Sign up at https://render.com with GitHub. The free tier needs no credit card.
2. **New + → Web Service**, then select this repository.
3. Settings:
   - **Branch:** `master`
   - **Language:** Docker (detected from the `Dockerfile`)
   - **Instance type:** Free
   - **Advanced → Health Check Path:** `/health`
4. Click **Deploy Web Service**. Render builds the image from the `Dockerfile` and starts it.

Render sets the `$PORT` environment variable, and the `Dockerfile` starts uvicorn on `${PORT}`, so
no extra configuration is needed.

## 3. Automatic deploys

- **Render** watches the `master` branch. Every push to `master` rebuilds the image and redeploys.
- **GitHub Actions** (`.github/workflows/tests.yml`) runs the test suite on every push and pull
  request, so a failing change is visible before it is merged into `master`.

## Free tier notes

A free Render service goes to sleep after about 15 minutes without traffic. The first request after
that takes up to a minute while it wakes up; later requests are fast.
