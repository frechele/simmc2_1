import argparse
import gin

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc

import mlflow
import mlflow.pytorch


MLFLOW_TRACKING_URI = "http://localhost:5000"


@gin.configurable(denylist=["args"])
def train(args, engine, experiment_name: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    mlflow.pytorch.autolog()
    mlflow.pytorch.log_model(engine.model, "model")
    mlflow.log_artifact(args.config)

    swa_callback = plc.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer_args = {
        "precision": 16,
        "gradient_clip_val": 1.0,
        "callbacks": [swa_callback],
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
