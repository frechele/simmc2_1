import argparse
import gin

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

import mlflow
import mlflow.pytorch

from simmc.engine.logger import MLflowLogger


MLFLOW_TRACKING_URI = "http://localhost:5000"


@gin.configurable(denylist=["args"])
def train(args, engine, experiment_name: str):
    mlflow_logger = MLflowLogger(experiment_name, MLFLOW_TRACKING_URI, engine)

    mlflow.log_artifact(args.config)

    checkpoint_callback = plc.ModelCheckpoint(
        verbose=True,
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    swa_callback = plc.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer_args = {
        "precision": 16,
        "gradient_clip_val": 1.0,
        "logger": mlflow_logger,
        "callbacks": [checkpoint_callback, swa_callback],
    }

    trainer = pl.Trainer.from_argparse_args(args,
        **trainer_args
    )

    trainer.fit(engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    train(args)
