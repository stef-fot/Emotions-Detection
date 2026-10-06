# Serverless deployment

Deploy from the **repository root** (root `main.py` re-exports the handlers and
the `EmotionDetection` package is included):

```bash
gcloud functions deploy emotion-detector --gen2 --runtime python312 \
  --trigger-http --allow-unauthenticated --entry-point analyze --source .
```

For containers (Cloud Run, Fly, Render...) use the root `Dockerfile` instead.
