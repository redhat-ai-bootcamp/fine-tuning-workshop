"""OpenShift AI / Kubeflow pipeline for Nemotron phishing workshop.

Usage:
  python pipelines/compile_pipeline.py --output nemotron_pipeline.yaml

Then upload the compiled YAML to OpenShift AI.
Build a container image that contains this repository and pass it
as the `workshop_image` parameter when running the pipeline.
Ensure KAGGLE_USERNAME and KAGGLE_KEY are available to the pipeline pods
(for example, via a secret and environment variables).
"""

import kfp
from kfp import dsl


def _container_op(name, image, command, args, pvol):
    return dsl.ContainerOp(
        name=name,
        image=image,
        command=command,
        arguments=args,
        pvolumes={"/workspace": pvol},
    )


@dsl.pipeline(
    name="nemotron-phishing-workshop",
    description="Download, preprocess, fine-tune, test, and benchmark Nemotron.",
)
def nemotron_workshop_pipeline(
    workshop_image: str = "quay.io/your-org/nemotron-workshop:latest",
    pvc_name: str = "nemotron-workspace",
    model_name: str = "nvidia/Nemotron-4-Mini-HF",
    max_emails: int = 50000,
):
    volume = dsl.PipelineVolume(pvc=pvc_name)

    download = _container_op(
        name="download-dataset",
        image=workshop_image,
        command=["python", "scripts/download_dataset.py"],
        args=["--output_dir", "/workspace/data/raw"],
        pvol=volume,
    )

    preprocess = _container_op(
        name="prepare-jsonl",
        image=workshop_image,
        command=["python", "scripts/prepare_jsonl.py"],
        args=[
            "--input_dir",
            "/workspace/data/raw/maildir",
            "--output_dir",
            "/workspace/data/processed",
            "--max_emails",
            str(max_emails),
        ],
        pvol=volume,
    )

    train = _container_op(
        name="train-model",
        image=workshop_image,
        command=["python", "scripts/train.py"],
        args=[
            "--data_dir",
            "/workspace/data/processed",
            "--output_dir",
            "/workspace/outputs",
            "--model_name",
            model_name,
        ],
        pvol=volume,
    )

    test = _container_op(
        name="test-model",
        image=workshop_image,
        command=["python", "scripts/test_model.py"],
        args=[
            "--test_file",
            "/workspace/data/processed/test.jsonl",
            "--adapter_dir",
            "/workspace/outputs/adapter",
            "--model_name",
            model_name,
        ],
        pvol=volume,
    )

    benchmark = _container_op(
        name="benchmark-model",
        image=workshop_image,
        command=["python", "scripts/benchmark_model.py"],
        args=[
            "--local",
            "--adapter_dir",
            "/workspace/outputs/adapter",
            "--model_name",
            model_name,
        ],
        pvol=volume,
    )

    preprocess.after(download)
    train.after(preprocess)
    test.after(train)
    benchmark.after(test)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(nemotron_workshop_pipeline, "nemotron_pipeline.yaml")
