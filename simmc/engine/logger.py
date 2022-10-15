from copy import deepcopy
from collections import OrderedDict

import mlflow
import mlflow.pytorch

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import torch


class MLflowLogger(Logger):
    def __init__(self, experiment_name: str, tracking_uri: str, engine: pl.LightningModule):
        super(MLflowLogger, self).__init__()

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name

        self.engine = engine

    @rank_zero_only
    def log_hyperparams(self, params):
        params = _convert_params(params)
        params = _flatten_dict(params)

        mlflow.log_params(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        mlflow.log_metrics(metrics, step)

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback) -> None:
        if hasattr(checkpoint_callback, "best_model_path"):
            ckpt = torch.load(checkpoint_callback.best_model_path)["state_dict"]
            new_ckpt = OrderedDict()

            for k in ckpt:
                new_k = k[6:]  # remove "model." prefix
                new_ckpt[new_k] = ckpt[k]

            model = deepcopy(self.engine.model)
            model.load_state_dict(new_ckpt)
            mlflow.pytorch.log_model(model, "model")

    @property
    def name(self):
        return self.experiment_name

    @property
    def version(self):
        return mlflow.active_run().info.run_id
