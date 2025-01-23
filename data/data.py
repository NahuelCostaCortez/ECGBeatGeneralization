import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import shutil
import wfdb
from scipy.ndimage import zoom
from scipy.signal import savgol_filter, resample
from preprocess import normalize, qsPeaks
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

root_path = "/home/nahuel/ecg/generalization/"
SEED = 42

class PhysioNetDataset():
    def __init__(self, dataset, path : str = False, download : bool = False, use_numpy: bool = True):
        self.path = path
        self.download = download
        self.use_numpy = use_numpy
        self.lead = -1
        self.num_classes = -1
        self.class_counts = -1

        if dataset == "MIT-BIH":
            url = "https://physionet.org/files/mitdb/1.0.0/"
            self.lead = 0 # Lead II
        elif dataset == "INCART":
            url = "https://physionet.org/files/incartdb/1.0.0/"
            self.lead = 1 # Lead II

        if download:
            self.download_dataset(url)

        self.files = sorted(list(set([file.split('.')[0] for file in os.listdir(path) if file.endswith('.atr') or file.endswith('.hea')])))
        self.X = []
        self.y = []
        self.pre_process()

    def download_dataset(self, url):
        command = [
            "wget",
            "-r",  # Recursive download
            "-N",  # Only download newer files
            "-c",  # Continue interrupted downloads
            "-np",  # Do not ascend to the parent directory
            "--cut-dirs=5",  # Remove the first 5 directory levels
            "-P", self.path,  # Destination folder
            url  # Download URL
        ]

        # Execute the wget command
        try:
            subprocess.run(command, check=True)
            print("Download completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during download: {e}")
        
        # Move the files to the root directory and remove unnecessary subfolders
        physionet_folder = os.path.join(self.path, "physionet.org")
        if os.path.exists(physionet_folder):
            for file_name in os.listdir(physionet_folder):
                shutil.move(os.path.join(physionet_folder, file_name), self.path)
            # Remove the intermediate folder
            shutil.rmtree(os.path.join(self.path, "physionet.org"))
            # Remove the specific record 102-0.atr
            record_to_remove = os.path.join(self.path, "102-0.atr")
            if os.path.exists(record_to_remove):
                os.remove(record_to_remove)
                print("Record 102-0.atr removed.")
            else:
                print("Record 102-0.atr not found.")
            print("Done.")
        else:
            print("Couldn't remove 'physionet.org' intermediate folder.")

    def pre_process(self, target_fs=360):
        beat_len = 280 # pre-fixed length of segments
        seg_values = []
        seg_labels = []

        # Process each file
        for item in range(len(self.files)):
            print(f"Processing record {item + 1}/{len(self.files)}")
            # Load ECG file
            record, annotation = self.__getitem__(item)
            signal = record.p_signal[:, self.lead]  # Usar Lead II
            # Resample the signal
            if record.fs != target_fs:
                # Calculate the number of samples for the target frequency
                num_samples = int(len(signal) * target_fs / record.fs)
                signal = resample(signal, num_samples)
            # Smooth the signal
            signal = savgol_filter(signal, window_length=20, polyorder=2) # windows length dependerá del dataset
            # Normalize signal
            signal = normalize(signal)

            rPeaks = annotation.sample + 1
            # Identify rest of peaks PQST
            peaks = qsPeaks(signal, rPeaks, record.fs)
            tpeaks = peaks[:, 6]

            # Group into AAMI classes
            seg_values_classified = []
            seg_labels_classified = []

            for ind, annot in enumerate(annotation.symbol):
                if annot in ['N', 'L', 'R', 'e', 'j']:
                    label = 'N'
                elif annot in ['A', 'a', 'J', 'S']:
                    label = 'S'
                elif annot in ['V', 'E']:
                    label = 'V'
                elif annot == 'F':
                    label = 'F'
                elif annot in ['/', 'f', 'Q']:
                    label = 'Q'
                else:
                    continue  # Skip other classes

                if tpeaks[ind] == 0: # Skip if no T-peak is found
                    continue

                # Segment and normalize the signal
                if ind == 0: # First segment
                    segment = signal[0:tpeaks[ind] - 1]

                    # make sure the segment is not longer than the record
                    segment = segment[:min(record.fs, len(segment))]

                    # Redimensionamos el segmento a 'beat_len'
                    t_sig = zoom(segment, beat_len / len(segment), order=1)  # 'order=1' es para una interpolación lineal
                    
                else:
                    # Reshape the segment to 'beat_len'
                    segment = signal[tpeaks[ind-1]:tpeaks[ind] - 1]
                    t_sig = zoom(segment, beat_len / len(segment), order=1)

                # Append the segment and label
                seg_values_classified.append(t_sig)
                seg_labels_classified.append(label)
            
            # Append the segments and labels
            seg_values.append(seg_values_classified)
            seg_labels.append(seg_labels_classified)

        # Convert to numpy arrays
        X = np.concatenate(seg_values[1:]) # Skip the first element because the tpeak is wrong
        y = np.concatenate(seg_labels[1:])

        # Check samples per class and remove classes with less than 200 samples
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            if count < 200:
                X = X[y != label]
                y = y[y != label]

        # Check if the majority class is not more than 3 times the second most frequent class
        sorted_counts = sorted(counts, reverse=True)
        if len(sorted_counts) > 1 and sorted_counts[0] > 3 * sorted_counts[1]:
            # Remove samples from the majority class to balance the dataset
            np.random.seed(SEED)
            majority_class = unique[np.argmax(counts)]
            indices_to_remove = np.random.choice(np.where(y == majority_class)[0], size=int(sorted_counts[0] - 3 * sorted_counts[1]), replace=False)
            X = np.delete(X, indices_to_remove, axis=0)
            y = np.delete(y, indices_to_remove)

        self.X = X
        self.y = y
        self.num_classes = len(np.unique(y))
        self.class_counts = np.unique(y, return_counts=True)[1]

    def __getitem__(self, idx, pre_process=False):
        record = wfdb.rdrecord(self.path + "/" + self.files[idx])
        annotation = wfdb.rdann(self.path + "/" + self.files[idx], 'atr')
        if pre_process:
            return self.X, self.y
        return record, annotation
    
def plot_ecg(record):
    """
    Plots multiple leads of an ECG signal along with annotations for a given record and annotation.
    
    :param record: The ECG record object (from wfdb.rdrecord)
    :param annotation: The annotation object (from wfdb.rdann)
    """
    # Número de derivaciones (leads)
    num_leads = record.p_signal.shape[1]
    
    # Crear una figura con subgráficos para cada derivación
    fig, axs = plt.subplots(num_leads, 1, figsize=(12, 10), sharex=True)  # Reducir tamaño de la figura
    if num_leads == 1:
        axs = [axs]  # Asegura que axs sea una lista, incluso si solo hay un lead
    
    # Tiempo en segundos
    time = np.arange(record.p_signal.shape[0]) / record.fs  # Convertir muestras a segundos
    
    # Trazar cada derivación
    for i in range(num_leads):
        axs[i].plot(time, record.p_signal[:, i], linewidth=0.25)
        
        # Etiquetas y cuadrícula
        axs[i].set_ylabel(f'V{i+1}/mV', fontsize=8)  # Reducir tamaño de las etiquetas
        axs[i].tick_params(axis='y', labelsize=8)
        # Configurar una cuadrícula más densa
        axs[i].grid(True, which='major', linestyle='--', linewidth=0.5)
        
        
    # Ajustar la visualización
    axs[-1].set_xlabel('Time (seconds)', fontsize=10)  # Reducir tamaño de la etiqueta
    fig.suptitle(f"Study {record.record_name} example", fontsize=10)  # Reducir tamaño del título
    
    # Ajustar márgenes para que todo se vea bien
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Ajustar para que el título no se solape
    plt.show()

def plot_info(data, num_samples=5):
    """
    Plots a specified number of random samples for each class in the dataset.

    :param MITData: The dataset object containing X (data) and y (labels)
    :param num_samples: Number of random samples to plot for each class
    """
    class_labels = np.unique(data.y)  # Get unique class labels
    
    # Plotting the distribution of classes with class counts and different colors for each bar
    fig, axs = plt.subplots(1, figsize=(10, 6))
    class_counts = [np.sum(data.y == label) for label in class_labels]
    bars = axs.bar(class_labels, class_counts, color=plt.get_cmap('viridis')(np.linspace(0, 1, len(class_labels))))
    for bar, count in zip(bars, class_counts):
        axs.text(bar.get_x() + bar.get_width()/2, bar.get_height(), count, ha='center', va='bottom')
    axs.set_xlabel('Class Labels')
    axs.set_ylabel('Number of Samples')
    axs.set_title('Distribution of Classes')
    plt.tight_layout()
    plt.show()
    
    # Plotting the samples for each class
    fig, axs = plt.subplots(len(class_labels), num_samples, figsize=(20, 10), sharey=True)
    for i, label in enumerate(class_labels):
        # Find indices of samples belonging to the current class
        class_indices = np.where(data.y == label)[0]
        # Select a specified number of random indices
        random_indices = np.random.choice(class_indices, size=num_samples, replace=False)
        # Plot the selected samples
        for j, idx in enumerate(random_indices):
            axs[i, j].plot(data.X[idx])
            axs[i, j].set_title(f"Class {label}, Sample {j+1}")
    plt.tight_layout()
    plt.show()

def load_data(dataset, batch_size=32):
    """
    Loads the dataset for training and testing.

    :param dataset: The name of the dataset to load.
    :return: The train, validation, and test dataloaders.
    """

    class_counts = None

    if dataset == "MIT-toy":
        # Load the MIT-toy dataset
        df_train = pd.read_csv(root_path+"data/MIT-toy/mitbih_train.csv", header=None)
        df_test = pd.read_csv(root_path+"data/MIT-toy/mitbih_test.csv", header=None)
        # Convert pandas dataframes to tensors for training and testing
        train_dataset = TensorDataset(torch.tensor(df_train.values[:,:-1].astype(np.float32)), torch.tensor(df_train.values[:,-1].astype(np.int64)))
        class_counts = np.unique(df_train.values[:,-1], return_counts=True)[1].astype(np.float32)
        test_dataloader = DataLoader(TensorDataset(torch.tensor(df_test.values[:,:-1].astype(np.float32)), torch.tensor(df_test.values[:,-1].astype(np.int64))), batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))
    else:
        # Load other datasets using PhysioNetDataset
        train_dataset = PhysioNetDataset(dataset, path=root_path+'data/'+dataset)
        # Split the dataset into training and testing sets
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(SEED))
        class_counts = train_dataset.class_counts.astype(np.float32)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    return train_dataloader, val_dataloader, test_dataloader, class_counts

def get_shot(dataset, shot, num_classes):
    # Set a random seed for reproducibility
    np.random.seed(SEED)
    from torch.utils.data import Subset

    # Obtén todas las etiquetas
    all_labels = np.array([label for _, label in dataset])

    # Encuentra índices por clase
    indices_by_class = {cls: np.where(all_labels == cls)[0] for cls in range(num_classes)}
    
    # Selecciona muestras balanceadas
    selected_indices = []

    for cls, indices in indices_by_class.items():
        selected_indices.extend(np.random.choice(indices, size=shot, replace=False))

    # Crea el subset
    subset = Subset(dataset, selected_indices)

    # Crea el dataloader
    dataloader = DataLoader(subset, batch_size=shot, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    return dataloader