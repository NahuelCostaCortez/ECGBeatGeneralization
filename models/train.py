import sys
import os

# Agregar la carpeta que contiene 'data.py' al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

import argparse
import numpy as np
from model import select_model, ECGModel, evaluate
from data import load_data
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

#seed_everything(42, workers=True) # sets seeds for numpy, torch and python.random.

root_path = "/home/nahuel/ecg/generalization/"

def main(dataset_name, model_name, path, use_class_weights=None):

    print("Training model with dataset:", dataset_name, "and model:", model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    batch_size = 32
    train_dataloader, val_dataloader, test_dataloader, class_counts = load_data(dataset_name, batch_size)
    if use_class_weights and class_counts is not None:
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights)
        # move to the same device as the model
        class_weights = class_weights.to(device)

    # Load model
    model_module = select_model(model_name)
    model = ECGModel(model_module, class_weights) # Already has configured optimizer and scheduler


    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss",
                                 filename="best_model",
                                 save_top_k=1,
                                 mode="min")
    
    steps_per_epoch = len(train_dataloader.dataset) // batch_size
    trainer = Trainer(accelerator="auto", # If your machine has GPUs, it will use the GPU Accelerator for training
                      callbacks=[early_stopping, checkpoint],
                      default_root_dir=path, 
                      max_epochs=1000, # epochs = 1000 by default
                      enable_progress_bar=False, # Disables the progress bar
                      log_every_n_steps=steps_per_epoch) # Doesn´t seem to work
    #Trainer(deterministic=True) To make the training reproducible

    print("Starting training...")
    trainer.fit(model, 
                train_dataloader, 
                val_dataloader) 
    print("Training finished.")

    print("Evaluating model...")
    model = ECGModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, ECGModel(model))
    evaluate(model, test_dataloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MIT-toy corresponds to https://www.kaggle.com/datasets/shayanfazeli/heartbeat?select=mitbih_test.csv
    parser.add_argument(
        "--dataset", type=str, default="MIT-toy", choices=["MIT-toy", "MIT-BIH", "INCART"]
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
        default=root_path+"models/saved/",
    )
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    path = args.path

    main(dataset, model, path)