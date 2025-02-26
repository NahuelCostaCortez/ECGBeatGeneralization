# Project Structure

This document provides an overview of the project's directory structure and the purpose of each component.

## Main Directories and files

- `train.py`: main experiment script

### 📁 data/
Contains all the data-related files and datasets used in the project:
- `data.py`: Main data processing and handling script
- `preprocess.py`: Data preprocessing utilities
- Dataset directories:
  - `MIT-BIH/`: MIT-BIH Arrhythmia Database
  - `NSR/`: Normal Sinus Rhythm Database
  - `INCART/`: St. Petersburg INCART Arrhythmia Database
  - `MIT-toy/`: Toy dataset for testing purposes
  - `Icentia11k/`: Icentia11k Dataset for ECG classification

### 📁 models/
Contains all model-related implementations and training scripts:
- `seq2seq.py`: Sequence-to-sequence model implementation
- `cnn.py`: Convolutional Neural Network model implementation
- `model.py`: Base model definitions
- `saved/`: Directory for storing trained model checkpoints

### 📁 notebooks/
Jupyter notebooks for analysis and experimentation:
- `evaluation.ipynb`: Model evaluation and results analysis
- `playground.ipynb`: Experimental notebook for testing and development
- `visualization.ipynb`: Data and results visualization notebook

## Configuration Files

- `requirements.txt`: Python dependencies and their versions
- `.gitignore`: Specifies which files Git should ignore
- `README.md`: Main project documentation and overview
- `config.yaml`: Configuration files for experiments

## Project Overview
This project focuses on ECG (Electrocardiogram) signal analysis and classification using various deep learning approaches, particularly sequence-to-sequence models and CNNs. It incorporates multiple ECG databases (MIT-BIH, NSR, INCART, Icentia11k). The project emphasizes model generalization across different databases and includes comprehensive evaluation metrics for various ECG beat types (N, S, V, F, Q).