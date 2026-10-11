#!/usr/bin/env bash
# Deploy RideCourt to Google Cloud Run.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - PROJECT_ID env var set (your GCP project)
#
# Usage:  PROJECT_ID=my-project ./scripts/deploy.sh
set -euo pipefail

: "${PROJECT_ID:?PROJECT_ID environment variable is required}"
REGION="asia-southeast1"
SERVICE="ridecourt"
SA="${SERVICE}-sa"
# One instance: case state and the live event stream are kept in memory, so a resolve and its
# event stream must reach the same instance.
MAX_INSTANCES="${MAX_INSTANCES:-1}"
SA_EMAIL="${SA}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== RideCourt Cloud Run deployment ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE}"
echo ""

# ------------------------------------------------------------------
# 1. Create a dedicated service account (idempotent).
# ------------------------------------------------------------------
if gcloud iam service-accounts describe "${SA_EMAIL}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Service account ${SA_EMAIL} already exists."
else
  echo "Creating service account ${SA}..."
  gcloud iam service-accounts create "${SA}" \
    --display-name="RideCourt runtime SA" \
    --project="${PROJECT_ID}"
fi

# ------------------------------------------------------------------
# 2. Grant only roles/aiplanet.user to the service account.
# ------------------------------------------------------------------
echo "Granting roles/aiplatform.user to ${SA_EMAIL}..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  --quiet >/dev/null

# ------------------------------------------------------------------
# 3. Deploy to Cloud Run (builds from source via the Dockerfile).
# ------------------------------------------------------------------
echo "Deploying ${SERVICE} to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --source . \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --allow-unauthenticated \
  --service-account "${SA_EMAIL}" \
  --max-instances "${MAX_INSTANCES}" \
  --set-env-vars "LLM_PROVIDER=gemini,GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,MODEL=gemini-3.8-flash" \
  --quiet

# ------------------------------------------------------------------
# 4. IAM propagation retry loop.
# The Vertex AI API may reject requests for up to a few minutes after
# the service account is granted roles/aiplatform.user. Retry until
# the service responds with a healthy status.
# ------------------------------------------------------------------
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "=== Waiting for IAM propagation (may take 1-3 minutes)... ==="
MAX_RETRIES=12
SLEEP_SEC=15

for i in $(seq 1 "${MAX_RETRIES}"); do
  echo "  Attempt ${i}/${MAX_RETRIES}: ${SERVICE_URL}/healthz"
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/healthz" || echo "000")
  if [ "${HTTP_CODE}" = "200" ]; then
    echo ""
    echo "=== Deployment successful! ==="
    echo "Service URL: ${SERVICE_URL}"
    echo "Health:      ${SERVICE_URL}/healthz"
    exit 0
  fi
  echo "  Got HTTP ${HTTP_CODE}, retrying in ${SLEEP_SEC}s..."
  sleep "${SLEEP_SEC}"
done

echo ""
echo "WARNING: IAM propagation may still be in progress."
echo "The service account has roles/aiplatform.user but Vertex AI"
echo "may need a few more minutes. Check again later:"
echo "  curl ${SERVICE_URL}/healthz"
exit 0
