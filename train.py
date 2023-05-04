from data import MNISTSupervisedDataModule
from model import MNISTSupervisedModel
import pytorch_lightning as pl

if __name__ == "__main__":
    data_module = MNISTSupervisedDataModule(
        debug=False,
        batch_size=512,
        batch_size_val=6000,
        val_fraction=0.1,
        num_workers=4,
        num_workers_val=0,
    )
    model = MNISTSupervisedModel(lr=0.004, num_layers=4)

    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", logger=[])

    # trainer.fit(model, train_dataloaders=data_module.train_dataloader())
    trainer.fit(model, datamodule=data_module)
#