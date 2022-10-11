import argparse
import gin

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc


@gin.configurable(denylist=["args"])
def train(args, engine):
    ckpt_callback = plc.ModelCheckpoint(
        verbose=True,
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    swa_callback = plc.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer_args = {
        "gradient_clip_val": 1.0,
        "callbacks": [ckpt_callback, swa_callback],
    }

    trainer = pl.Trainer.from_argparse_args(args,
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
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
