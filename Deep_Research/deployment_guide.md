# Deploying Deep Research to Google Cloud Run

This guide assumes you have the **Google Cloud SDK (`gcloud`)** installed and authenticated.

## Prerequisites

1.  **Google Cloud Project**: You need a project with billing enabled.
2.  **APIs Enabled**: You need to enable the Cloud Run and Artifact Registry APIs.
    ```powershell
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com
    ```

## Step 1: Initialize Google Cloud Config

If you haven't already, log in and set your project:

```powershell
# Login to Google Cloud
gcloud auth login

# Set your project ID (replace YOUR_PROJECT_ID)
gcloud config set project YOUR_PROJECT_ID
```

## Step 2: Build and Deploy

You can use a single command to build the image (using Google Cloud Build) and deploy it to Cloud Run.

Run this command from the `Deep_Research` directory:

```powershell
gcloud run deploy deep-research-service `
  --source . `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated
```

**What this does:**
- `--source .`: Uploads the current directory and builds the container image in the cloud.
- `--platform managed`: Uses the fully managed Cloud Run platform.
- `--region us-central1`: Deploys to the US Central region (you can change this).
- `--allow-unauthenticated`: Makes the application public (anyone with the URL can access it). **Remove this flag if you want to require IAM authentication.**

## Step 3: Access your Service

Once the command finishes, it will print a URL ending in `.run.app`.

Example output:
```
Service [deep-research-service] has been deployed successfully.
Service URL: https://deep-research-service-xyz123-uc.a.run.app
```

You can now send POST requests to this URL:
- Endpoint: `https://YOUR-URL.run.app/deep-research`

### Note on Environment Variables
If your application needs the `OPENAI_API_KEY`, you must pass it during deployment:

```powershell
gcloud run deploy deep-research-service `
  --source . `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --set-env-vars OPENAI_API_KEY=sk-your-key-here
```
**Better Security:** It is recommended to use **Secret Manager** for API keys instead of plain text environment variables in production.
