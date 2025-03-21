import argparse
from models.model import select_model, ECGModel
from data.data import load_data
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

seed_everything(42, workers=True) # sets seeds for numpy, torch and python.random.

root_path = "/home/nahuel/ecg/generalization/"

def main(dataset_name, model_name, lr, batch_size, path, return_sequences=False):

    print("Training model with dataset:", dataset_name, "and model:", model_name)

    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(dataset_name=dataset_name, batch_size=batch_size, return_sequences=return_sequences)

    # Load model
    model_module = select_model(model_name, 
                                n_classes=4)
    model = ECGModel(model_module, lr=lr) # Already has configured optimizer and scheduler


    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=False, mode="min")
    checkpoint = ModelCheckpoint(dirpath=path,
                                 monitor="val_loss",
                                 filename="best_model",
                                 save_top_k=1,
                                 mode="min")
    
    steps_per_epoch = len(train_dataloader.dataset) // batch_size
    trainer = Trainer(accelerator="auto", # If your machine has GPUs, it will use the GPU Accelerator for training
                      callbacks=[early_stopping, checkpoint],
                      default_root_dir=path, 
                      max_epochs=100, # 1000 by default
                      enable_progress_bar=False, # Disables the progress bar
                      log_every_n_steps=steps_per_epoch)#, # Doesn´t seem to work
                      #logger=wandb_logger) 
    #Trainer(deterministic=True) To make the training reproducible

    print("Starting training...")
    trainer.fit(model, 
                train_dataloader, 
                val_dataloader) 
    print("Training finished.")

    print("Evaluating model...")
    model = ECGModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model_module, map_location="cpu")
    model.evaluate(test_dataloader, "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, default="MIT-BIH", choices=["MIT-BIH", "INCART"]
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CNN",
        choices=["CNN", "Seq2Seq"],
    )
    parser.add_argument(
        "--path",
        type=str,
        default="models/saved/",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Whether to use class weights for the loss function",
    )
    parser.add_argument(
        "--r_r",
        action="store_true",
        help="Whether to use class weights for the loss function",
    )
    parser.add_argument(
        "--return_sequences",
        action="store_true",
    )
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    lr = 1e-3
    batch_size = 20 #32
    patience_epochs = 10
    path = root_path+args.path
    use_class_weights = args.use_class_weights
    r_r = args.r_r
    return_sequences = args.return_sequences

    #wandb_logger = WandbLogger(project='ECG', name=experiment_name)
    config = {
        "dataset": dataset,
        "model": model,
        "learning_rate": lr,
        "batch_size": batch_size,
        "patience_epochs": patience_epochs,
    }
    #wandb_logger.experiment.config.update(config)
    #wandb_logger.experiment.config["dataset"] = dataset

    main(dataset, model, lr, batch_size, path, return_sequences)