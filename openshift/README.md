# OpenShift AI Serving Notes

This folder contains a minimal KServe `InferenceService` template. Build a container image
that runs `python scripts/serve.py` and pushes the adapter into the image or mounts it from
persistent storage, then update the image reference in the YAML.

Example build command (adjust for your registry):
```bash
podman build -t quay.io/your-org/nemotron-phishing-serve:latest -f Dockerfile .
```

Then apply:
```bash
oc apply -f openshift/inference_service.yaml
```

Update the notebook `notebooks/02_openshift_ai_infer.ipynb` with the routed endpoint.
