#!/usr/bin/env python3
import argparse

import kfp

from openshift_ai_pipeline import nemotron_workshop_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the OpenShift AI pipeline")
    parser.add_argument(
        "--output",
        default="nemotron_pipeline.yaml",
        help="Path for the compiled pipeline YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kfp.compiler.Compiler().compile(nemotron_workshop_pipeline, args.output)
    print(f"Wrote pipeline to {args.output}")


if __name__ == "__main__":
    main()
