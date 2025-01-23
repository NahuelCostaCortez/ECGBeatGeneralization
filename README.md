# ECG Classification Project

This project implements ECG signal classification using deep learning models.

## Environment Setup

This project uses `uv` for dependency management. Follow these steps to set up your environment:

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a new virtual environment:
```bash
uv venv --python 3.11 _name_
```

3. Activate the virtual environment:
```bash
source _name_/bin/activate
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```


## Project Structure

- `data/`: Contains data processing scripts and datasets
  - `data.py`: Dataset loading and preprocessing
- `models/`: Model implementations and training scripts
  - `cnn.py`: CNN model architecture
  - `seq2seq.py`: Seq2Seq model architecture
  - `train.py`: Training script
- `notebooks/`: Jupyter notebooks for analysis and evaluation

## Usage

1. Prepare your data and place it in the `data/` directory
2. Train a model:
```bash
python models/train.py --model CNN --dataset MIT-BIH
```