import os
import yaml
import shutil
import mlflow
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from collections import defaultdict

from utils.path_helper import set_project_root_path

from typing import List


def get_or_create_experiment(tracking_uri: str, name: str) -> str:
    """
    Creates mlflow experiment with specified name of
    retrieves it if it is already created or deleted
    """

    mlflow.set_tracking_uri(uri=tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name=name)
    if experiment is None:
        experiment_id = client.create_experiment(name=name)
    else:
        if dict(experiment)["lifecycle_stage"] == "deleted":
            client.restore_experiment(dict(experiment)["experiment_id"])
        experiment_id = dict(experiment)["experiment_id"]
    return experiment_id


def get_experiment_metrics(run_id):
    return mlflow.tracking.MlflowClient().get_run(run_id).data.metrics


def get_experiment_params(run_id):
    return mlflow.tracking.MlflowClient().get_run(run_id).data.params


def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(
    run_id, client: mlflow.tracking.MlflowClient = None, include_artifacts: bool = False
):
    """
    Fetch params, metrics, tags, and artifacts in the specified run.
    Example::
        uri = 'uri'
        mlflow.set_tracking_uri(uri)
        experiment = mlflow.get_experiment_by_name('Hyperparam-OVA')  # 16
        runs = mlflow.list_run_infos(experiment.experiment_id)
        id16_runs = mlflow.list_run_infos('16')
        for run in id16_runs:
            run_id = run.run_id
            fetch_logged_data(run_id)

    """
    client = mlflow.tracking.MlflowClient() if client is None else client
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id)) if include_artifacts else None
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }
