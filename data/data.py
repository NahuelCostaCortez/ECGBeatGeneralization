import subprocess
import os
import shutil
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import savgol_filter, resample
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

root_path = "/home/nahuel/ecg/generalization/"

class PhysioNetDataset():
    def __init__(self, dataset, path : str = False, download : bool = False, use_numpy: bool = True):
        self.path = path
        self.download = download
        self.use_numpy = use_numpy
        self.lead = 0

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
            signal = np.array([resample(signal, int(len(sample) * 409.6 / 1000))/ 1000 for sample in samples_10]) # Resample to 400 Hz and convert to 1e-4V
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
        self.X = np.concatenate(seg_values[1:]) # Skip the first element because the tpeak is wrong
        self.y = np.concatenate(seg_labels[1:])

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
    
def onoffset(interval, mode):
    """
    Function calculates on/off set of QRS complex.
    
    :param interval: The ECG signal interval
    :param mode: Mode either 'on' for onset or 'off' for offset
    :return: ind - the index of the onset or offset
    """
    slope = []
    
    # Calculate the slope of the interval
    for i in range(1, len(interval) - 1):
        slope.append(interval[i + 1] - interval[i - 1])
    
    # Using MIN_SLOPE to determine onset placement
    if mode == 'on':
        ind = np.argmin(np.abs(slope))
    elif mode == 'off':
        slope_th = 0.2 * np.max(np.abs(slope))
        slope_s = np.where(np.abs(slope) >= slope_th)[0]
        ind = slope_s[0] if len(slope_s) > 0 else -1  # Return -1 if no slope_s is found
    else:
        raise ValueError("Invalid mode, please select 'on' or 'off'")
    
    return ind

def qsPeaks(ECG, Rposition, fs):
    """
    Q, S peaks detection.
    
    :param ECG: The ECG signal
    :param Rposition: Positions of R-peaks
    :param fs: Sampling frequency
    :return: ECGpeaks (detected Q, S, P, T peaks)
    """
    # Average heart beat length
    aveHB = len(ECG) / len(Rposition)
    
    # Initialize an array to store the fiducial points
    fid_pks = np.zeros((len(Rposition), 7), dtype=int)
    # fiducial points: P wave onset, Q wave onset, R wave peak, 
    # S wave onset, T wave onset, R wave offset, T wave offset
    
    # Set up the search windows (in samples)
    windowS = round(fs * 0.1)
    windowQ = round(fs * 0.05)
    windowP = round(aveHB / 3)
    windowT = round(aveHB * 2 / 3)
    windowOF = round(fs * 0.04)
    
    # Process each R-position
    for i in range(len(Rposition)):
        thisR = Rposition[i]
        
        # First
        if i == 0:
            fid_pks[i, 3] = thisR
            fid_pks[i, 5] = thisR + windowS
        # Last
        elif i == len(Rposition) - 1:
            fid_pks[i, 3] = thisR
            fid_pks[i, 1] = thisR - windowQ
        else:
            if (thisR + windowT) < len(ECG) and (thisR - windowP) >= 1:
                # Detect Q and S peaks
                fid_pks[i, 3] = thisR
                Sp = np.argmin(ECG[thisR:thisR + windowS])
                thisS = Sp + thisR
                fid_pks[i, 4] = thisS
                Qp = np.argmin(ECG[thisR - windowQ:thisR])
                thisQ = thisR - (windowQ + 1) + Qp
                fid_pks[i, 2] = thisQ
                
                # Detect QRS onset and offset
                interval_q = ECG[thisQ - windowOF:thisQ]
                thisON = thisQ - (windowOF + 1) + onoffset(interval_q, 'on')
                
                interval_s = ECG[thisS:thisS + windowOF]
                thisOFF = thisS + onoffset(interval_s, 'off') - 1
                
                fid_pks[i, 1] = thisON
                fid_pks[i, 5] = thisOFF
    
    # Detect P and T waves
    for i in range(1, len(Rposition) - 1):
        lastOFF = fid_pks[i - 1, 5]
        thisON = fid_pks[i, 1]
        thisOFF = fid_pks[i, 5]
        nextON = fid_pks[i + 1, 1]
        
        if thisON > lastOFF and thisOFF < nextON:

            
            Tzone = ECG[thisOFF:int(nextON - round((nextON - thisOFF) / 3))]
            Pzone = ECG[lastOFF + int(round(2 * (thisON - lastOFF) / 3)):thisON]

            try:
                thisT = np.argmax(Tzone)
                thisP = np.argmax(Pzone)
            except Exception as e:
                print("Error in Tzone or Pzone:", e)
                continue
            
            fid_pks[i, 0] = lastOFF + round(2 * (thisON - lastOFF) / 3) + thisP - 1
            fid_pks[i, 6] = thisOFF + thisT - 1
    
    # Filter out invalid peaks (those with 0 value)
    #ECGpeaks = []
    #for i in range(len(Rposition)):
        #if np.prod(fid_pks[i, :]) != 0:
    #    ECGpeaks.append(fid_pks[i, :])
    
    return np.array(fid_pks) #np.array(ECGpeaks)

def normalize(signal, min_val=-5, max_val=5):
    """
    Function to normalize the ECG signal.
    
    :param signal: The ECG signal
    :return: Normalized ECG signal
    """
    signal = (signal - min_val) / (max_val - min_val)
    return signal

def load_data(dataset, batch_size=32):
    """
    Loads the dataset for training and testing.

    :param dataset: The name of the dataset to load.
    :return: The train, validation, and test dataloaders.
    """

    if dataset == "MIT-toy":
        # Load the MIT-toy dataset
        df_train = pd.read_csv(root_path+"data/MIT-toy/mitbih_train.csv", header=None)
        df_test = pd.read_csv(root_path+"data/MIT-toy/mitbih_test.csv", header=None)
        # Convert pandas dataframes to tensors for training and testing
        train_dataset = TensorDataset(torch.tensor(df_train.values[:,:-1].astype(np.float32)), torch.tensor(df_train.values[:,-1].astype(np.int64)))
        test_dataloader = DataLoader(TensorDataset(torch.tensor(df_test.values[:,:-1].astype(np.float32)), torch.tensor(df_test.values[:,-1].astype(np.int64))), batch_size=batch_size, shuffle=False)
    else:
        # Load other datasets using PhysioNetDataset
        train_dataset = PhysioNetDataset(dataset, path=root_path+'data/'+dataset)
        # Split the dataset into training and testing sets
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def get_shot(dataset, shot, num_classes):
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
    dataloader = DataLoader(subset, batch_size=shot, shuffle=True)

    return dataloader