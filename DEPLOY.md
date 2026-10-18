# Deploying the model to Google Cloud Run

The NumPy network is trained once (`train.py`), its weights are saved to `model/`, and a FastAPI
service (`api/`) loads them and serves predictions. The service is packaged with Docker and runs
on Google Cloud Run.

```
train.py ──> model/spiral_model.npz ──> api/ (FastAPI) ──> Docker image ──> Cloud Run (public HTTPS URL)
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

## 2. One-time Google Cloud setup

1. Create an account at https://console.cloud.google.com and set up billing. The free tier covers this
   project, but Cloud Run requires a billing account to be linked.
2. Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
3. Log in and create a project (project IDs must be globally unique):

```bash
gcloud auth login
gcloud projects create YOUR_PROJECT_ID
gcloud config set project YOUR_PROJECT_ID
gcloud billing accounts list
gcloud billing projects link YOUR_PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 3. First deploy (manual)

From the repository root:

```bash
gcloud run deploy spiral-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --max-instances 2
```

`--source .` uploads the code, builds the `Dockerfile` with Cloud Build, stores the image in
Artifact Registry and deploys it. When it finishes it prints the service URL. Test it:

```bash
URL=$(gcloud run services describe spiral-api --region us-central1 --format 'value(status.url)')
curl "$URL/health"
curl -X POST "$URL/predict" -H "Content-Type: application/json" -d '{"points": [[0.3, -0.5]]}'
```

## 4. Automatic deploys with GitHub Actions (CI/CD)

`.github/workflows/deploy.yml` runs the tests on every push and pull request. On pushes to `master`
it also deploys to Cloud Run and smoke-tests the live URL. It authenticates with Workload Identity
Federation, so no service-account key is stored in GitHub.

One-time setup (replace `YOUR_PROJECT_ID`; the repo is `CCosmoe/neuralnetworks`):

```bash
PROJECT_ID=YOUR_PROJECT_ID
REPO=CCosmoe/neuralnetworks
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format 'value(projectNumber)')
SA=github-deployer@$PROJECT_ID.iam.gserviceaccount.com

gcloud services enable iamcredentials.googleapis.com
gcloud iam service-accounts create github-deployer --display-name "GitHub Actions deployer"

for role in roles/run.admin roles/iam.serviceAccountUser roles/cloudbuild.builds.editor \
            roles/artifactregistry.admin roles/storage.admin roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member "serviceAccount:$SA" --role $role
done

gcloud iam workload-identity-pools create github --location global --display-name "GitHub"
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location global --workload-identity-pool github \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository == '$REPO'"

gcloud iam service-accounts add-iam-policy-binding $SA \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

echo "GCP_WORKLOAD_IDENTITY_PROVIDER=projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-provider"
echo "GCP_SERVICE_ACCOUNT=$SA"
```

Then in GitHub go to **Settings → Secrets and variables → Actions → Variables** and add:

| Variable                         | Value                          |
|----------------------------------|--------------------------------|
| `GCP_PROJECT_ID`                 | your project ID                |
| `GCP_REGION`                     | `us-central1`                  |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | printed by the commands above  |
| `GCP_SERVICE_ACCOUNT`            | printed by the commands above  |

Until `GCP_PROJECT_ID` is set, the workflow only runs tests and skips the deploy job.

## Cost and cleanup

Cloud Run scales to zero when idle, and `--max-instances 2` caps how far it can scale. A demo
like this stays within the free tier. To remove everything:

```bash
gcloud run services delete spiral-api --region us-central1
gcloud projects delete YOUR_PROJECT_ID   # deletes all resources in the project
```
