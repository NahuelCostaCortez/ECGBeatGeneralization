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
from sklearn.model_selection import train_test_split

root_path = "/home/nahuel/ecg/generalization/"
SEED = 42

class PhysioNetDataset():
	def __init__(self, dataset, path : str = False, download : bool = False, r_r: bool = False, return_sequences: bool = False, pre_process: bool = True):
		self.dataset = dataset
		self.path = path
		self.r_r = r_r
		self.download = download
		self.lead = -1
		self.num_classes = -1
		self.class_counts = -1
		self.return_sequences = return_sequences

		if self.dataset == "MIT-BIH":
			url = "https://physionet.org/files/mitdb/1.0.0/"
			self.lead = 0 # Lead II
		elif self.dataset == "INCART":
			url = "https://physionet.org/files/incartdb/1.0.0/"
			self.lead = 0 # Lead II
		elif self.dataset == "NSR":
			url = "https://physionet.org/files/nsrdb/1.0.0/"
			self.lead = 0 # Lead II

		if download:
			self.download_dataset(url)

		self.files = sorted(list(set([file.split('.')[0] for file in os.listdir(path) if file.endswith('.atr') or file.endswith('.hea')])))
		self.X = []
		self.y = []
		if pre_process:
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

	def pre_process(self, target_fs=360, max_time=10, classes= ['F', 'N', 'S', 'V']):
		beat_len = 280 # pre-fixed length of segments
		seg_values = []
		seg_labels = []

		# Process each file
		for item in range(len(self.files)):
			# Load ECG file
			record, annotation = self.__getitem__(item)
			signal = record.p_signal[:, self.lead]  # Usar Lead II
			
			'''
			# Normalize signal
			signal = normalize(signal)
			# Resample the signal
			if record.fs != target_fs:
				# Calculate the number of samples for the target frequency
				num_samples = int(len(signal) * target_fs / record.fs)
				signal = resample(signal, num_samples)
			# Smooth the signal
			signal = savgol_filter(signal, window_length=20, polyorder=2) # windows length dependerá del dataset

			'''

			rPeaks = annotation.sample + 1
			
			# Identify rest of peaks PQST
			if not self.r_r:
				peaks = qsPeaks(signal, rPeaks, record.fs)
				tpeaks = peaks['T_peak']
			else:
				tpeaks = rPeaks

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
					if tpeaks[ind]-1 <= 0:
						continue
					segment = signal[0:tpeaks[ind] - 1]

					# make sure the segment is not longer than the record
					segment = segment[:min(record.fs, len(segment))]
					
				else:
					if tpeaks[ind-1] <= 0:
						continue
					segment = signal[tpeaks[ind-1]:tpeaks[ind]-1]

				# Redimensionamos el segmento a 'beat_len'  
				t_sig = zoom(segment, beat_len / len(segment), order=1) # 'order=1' es para una interpolación lineal
				t_sig = normalize(t_sig)
				# Append the segment and label
				seg_values_classified.append(t_sig)
				seg_labels_classified.append(label)
			
			# Append the segments and labels
			seg_values.append(seg_values_classified)
			seg_labels.append(seg_labels_classified)

		# Convert to numpy arrays
		X = np.concatenate(seg_values)
		y = np.concatenate(seg_labels)

		# Check samples per class and remove classes with less than 200 samples
		#unique, counts = np.unique(y, return_counts=True)
		#for label, count in zip(unique, counts):
		#    if count < 200:
		#        X = X[y != label]
		#        y = y[y != label]

		# Check if the majority class is not more than 3 times the second most frequent class
		#sorted_counts = sorted(counts, reverse=True)
		#if len(sorted_counts) > 1 and sorted_counts[0] > 3 * sorted_counts[1]:
		#    # Remove samples from the majority class to balance the dataset
		#    np.random.seed(SEED)
		#    majority_class = unique[np.argmax(counts)]
		#    indices_to_remove = np.random.choice(np.where(y == majority_class)[0], size=int(sorted_counts[0] - 3 * sorted_counts[1]), replace=False)
		#    X = np.delete(X, indices_to_remove, axis=0)
		#    y = np.delete(y, indices_to_remove)

		self.X = X
		self.y = y
		self.num_classes = len(np.unique(y))
		self.class_counts = np.unique(y, return_counts=True)[1]
		if self.return_sequences:
			data = self.X
			labels = self.y

			all_indices = []

			for cl in classes:
				idx_class = np.where(labels == cl)[0]
				idx_class = np.random.permutation(idx_class)
				all_indices.append(idx_class)
			
			all_indices = np.concatenate(all_indices)
			_data = data[all_indices]
			_labels = labels[all_indices]
			_labels = np.array([convert_label_to_int(item) for item in _labels])

			data = _data[:int((len(_data)/ max_time) * max_time), :]
			_labels = _labels[:int((len(_data) / max_time) * max_time)]

			# split data into sublist of 100=se_len values
			data = [data[i:i + max_time] for i in range(0, len(data), max_time)]
			labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]
			
			# shuffle
			permute = np.random.permutation(len(labels[:-1]))
			data = np.asarray(data[:-1])
			labels = np.asarray(labels[:-1])
			self.X = data[permute]
			self.y = labels[permute]

		print('Records processed!')

	def __getitem__(self, idx, pre_process=False):
		record = wfdb.rdrecord(self.path + "/" + self.files[idx])
		annotation = wfdb.rdann(self.path + "/" + self.files[idx], 'atr')
		if pre_process:
			return self.X, self.y
		return record, annotation
    
	def __call__(self, max_time=10, batch_size=20):
		# dividir en train/test
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

		# SMOTE
		from imblearn.over_sampling import SMOTE
		X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
		y_train= y_train.flatten()
		sm = SMOTE(random_state=12)
		X_train, y_train = sm.fit_resample(X_train, y_train) # todas a al mismo numero de muestras
		X_train = X_train[:(X_train.shape[0]//max_time)*max_time, :] # shape [muestras, 180]
		y_train = y_train[:(y_train.shape[0]//max_time)*max_time] # shape [muestras]

		if self.return_sequences:
			X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]]) # shape [muestras, 10, 180]
			y_train = np.reshape(y_train, [-1, y_test.shape[1]]) # shape [muestras, 10]

			# Añadir GO
			classes = np.unique(self.y)
			char2numY = dict(zip(classes, range(len(classes))))
			char2numY['<GO>'] = len(char2numY)
			y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
			y_train = np.array(y_train)

		# Create datasets
		train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64)))
		test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.int64)))
		
		# Create dataloaders
		train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
		val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

		return train_dataloader, val_dataloader, test_dataloader
            

def convert_label_to_string(label):
    """
    Converts a label integer to its corresponding string representation.
    
    :param label: The label to convert.
    :return: The string representation of the label.
    """
    label_map = {
        0: 'N',
        1: 'S',
        2: 'V',
        3: 'F',
        4: 'Q'
    }
    return label_map.get(label, 'Unknown')

def convert_label_to_int(label):
    """
    Converts a label to its corresponding integer representation.
    
    :param label: The label to convert.
    :return: The integer representation of the label.
    """
    label_map = {
        'N': 0,
        'S': 1,
        'V': 2,
        'F': 3,
        'Q': 4
    }
    return label_map.get(label, -1)


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
    Plots information about the dataset including record details and class distribution.

    :param data: The dataset object containing X (data) and y (labels)
    :param num_samples: Number of random samples to plot for each class
    """
    # Get sample data
    record, annotation = data.__getitem__(6)
    print("ECG shape:", record.p_signal.shape, end='\n')

    print("\nkeys in record:")
    for key in record.__dict__.keys():
        print('\t'+key+':'+str(record.__dict__[key]))

    print("\nkeys in annotation")
    for key in annotation.__dict__.keys():
        print('\t'+key+':'+str(annotation.__dict__[key]))

    # Check data format
    is_beat_format = len(data.X.shape) == 3

    if is_beat_format:
        # For beat format, we need to flatten the labels to count individual beats
        if isinstance(data.y[0], (list, np.ndarray)):
            # If labels are per beat, flatten them
            all_labels = np.concatenate(data.y)
        else:
            # If same label applies to all beats in a sample
            num_beats = data.X.shape[1]
            all_labels = np.repeat(data.y, num_beats)
        
        class_labels = [convert_label_to_string(label) for label in np.unique(all_labels)]
        class_counts = [np.sum(all_labels == label) for label in np.unique(all_labels)]
    else:
        # Original case for single sequence data
        class_labels = np.unique(data.y)
        class_counts = [np.sum(data.y == label) for label in class_labels]

    # Plotting the distribution of classes with class counts and different colors for each bar
    fig, axs = plt.subplots(1, figsize=(12, 6))
    bars = axs.bar(class_labels, class_counts, color=plt.get_cmap('viridis')(np.linspace(0, 1, len(class_labels))))
    
    # Calculate maximum count for y-axis limit
    max_count = max(class_counts)
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, class_counts):
        axs.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count}\n({count/sum(class_counts):.1%})', 
                ha='center', va='bottom')
    
    # Set y-axis limit with 15% margin for labels
    axs.set_ylim(0, max_count * 1.15)
    
    axs.set_xlabel('Class Labels')
    axs.set_ylabel('Number of Samples')
    title = 'Distribution of Classes (per beat)' if is_beat_format else 'Distribution of Classes'
    axs.set_title(title)
    
    # Add total count in the plot
    total_samples = sum(class_counts)
    axs.text(0.98, 0.98, f'Total: {total_samples}', 
             transform=axs.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_data(data, num_samples=5):
    """
    Visualizes a specified number of random samples for each class in the dataset.

    :param data: The dataset object containing X (data) and y (labels)
    :param num_samples: Number of random samples to plot for each class
    """
    class_labels = np.unique(data.y)  # Get unique class labels
    
    # Check data format
    is_beat_format = len(data.X.shape) == 3
    
    # Plotting the samples for each class
    fig, axs = plt.subplots(len(class_labels), num_samples, figsize=(20, 10), sharey=True)
    for i, label in enumerate(class_labels):
        # Find indices of samples belonging to the current class
        class_indices = np.where(data.y == label)[0]
        # Select a specified number of random indices
        random_indices = np.random.choice(class_indices, size=num_samples, replace=False)
        # Plot the selected samples
        for j, idx in enumerate(random_indices):
            if is_beat_format:
                # For data with multiple beats
                sample = data.X[idx]  # Shape: (num_beats, seq_len)
                num_beats, seq_len = sample.shape
                
                # Plot each beat
                for beat_idx in range(num_beats):
                    # Calculate the offset for this beat
                    offset = beat_idx * seq_len
                    # Plot the beat
                    axs[i, j].plot(range(offset, offset + seq_len), sample[beat_idx])
                    # Add vertical line to separate beats (except after last beat)
                    if beat_idx < num_beats - 1:
                        axs[i, j].axvline(x=offset + seq_len, color='gray', linestyle='--', alpha=0.5)
                    # Add beat label
                    if isinstance(data.y[idx], (list, np.ndarray)):
                        beat_label = convert_label_to_string(data.y[idx][beat_idx])
                    else:
                        beat_label = convert_label_to_string(data.y[idx])
                    # Add text label above each beat
                    axs[i, j].text(offset + seq_len/2, axs[i, j].get_ylim()[1], 
                                 f'{beat_label}', horizontalalignment='center')
            else:
                # Original case for single sequence data
                axs[i, j].plot(data.X[idx])
            
            axs[i, j].set_title(f"Class {label}, Sample {j+1}")
            axs[i, j].grid(True)
    
    plt.tight_layout()
    plt.show()

'''
def load_data(dataset, batch_size=32, r_r=False, return_sequences=False):
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
        train_dataset = PhysioNetDataset(dataset, path=root_path+'data/'+dataset, r_r=r_r, return_sequences=return_sequences)
        class_counts = train_dataset.class_counts.astype(np.float32)
        # Split the dataset into training and testing sets
        X = train_dataset.X
        if return_sequences:
            y = np.array([[convert_label_to_int(item) for item in sequence] for sequence in train_dataset.y])
        else:
            y = np.array([convert_label_to_int(label) for label in train_dataset.y])
        train_dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(y.astype(np.int64)))
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(SEED))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    return train_dataloader, val_dataloader, test_dataloader, class_counts
'''

'''
def load_data(dataset, batch_size=32):
    dataset = PhysioNetDataset(dataset, path=root_path+'data/'+dataset)
    
    classes = ['F','N','S','V']
    max_time = 10

    inputs = dataset.X
    labels = dataset.y

    # Filtrar classes
    #indexes_keep = np.isin(labels, classes)
    #inputs = inputs[indexes_keep]
    #labels = labels[indexes_keep]
    #labels = np.array([[convert_label_to_int(item) for item in sequence] for sequence in labels])
    # ------
    all_indices = []

    for cl in classes:
        idx_class = np.where(labels == cl)[0]
        idx_class = np.random.permutation(idx_class)
        all_indices.append(idx_class)
    
    all_indices = np.concatenate(all_indices)
    inputs = inputs[all_indices]
    labels = labels[all_indices]
    labels = np.array([convert_label_to_int(item) for item in labels])
    # ------

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    # SMOTE
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=12)
    X_train, y_train = sm.fit_resample(X_train, y_train) # todas a al mismo numero de muestras
    
    # Convertir a secuencias de max_time
    X_train = np.array([X_train[i:i + max_time] for i in range(0, len(X_train), max_time) if i + max_time <= len(X_train)])
    y_train = np.array([y_train[i:i + max_time] for i in range(0, len(y_train), max_time) if i + max_time <= len(y_train)]).squeeze()
    X_test = np.array([X_test[i:i + max_time] for i in range(0, len(X_test), max_time) if i + max_time <= len(X_test)])
    y_test = np.array([y_test[i:i + max_time] for i in range(0, len(y_test), max_time) if i + max_time <= len(y_test)]).squeeze()

    # Añadir GO
    classes = np.unique(labels)
    char2numY = dict(zip(classes, range(len(classes))))
    char2numY['<GO>'] = len(char2numY)
    y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
    y_train = np.array(y_train)

    # Create datasets
    train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64)))
    test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.int64)))
    
    # Create dataloaders
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    return train_dataloader, val_dataloader, test_dataloader
'''

def load_data(dataset_name, max_time=10, classes= ['F', 'N', 'S', 'V'], batch_size=20, return_sequences=False):
    
	dataset = PhysioNetDataset(dataset_name, path=root_path+'data/'+dataset_name)
	data = dataset.X
	labels = dataset.y

	all_indices = []

	for cl in classes:
		idx_class = np.where(labels == cl)[0]
		idx_class = np.random.permutation(idx_class)
		all_indices.append(idx_class)

	all_indices = np.concatenate(all_indices)
	_data = data[all_indices]
	_labels = labels[all_indices]
	_labels = np.array([convert_label_to_int(item) for item in _labels])

	data = _data[:int((len(_data)/ max_time) * max_time), :]
	_labels = _labels[:int((len(_data) / max_time) * max_time)]

	# split data into sublist of 100=se_len values
	data = [data[i:i + max_time] for i in range(0, len(data), max_time)]
	labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]

	# shuffle
	permute = np.random.permutation(len(labels[:-1]))
	data = np.asarray(data[:-1])
	labels = np.asarray(labels[:-1])
	data = data[permute]
	labels = labels[permute]

	inputs = data
	labels = labels

	# dividir en train/test
	X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

	# SMOTE
	from imblearn.over_sampling import SMOTE
	X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
	y_train= y_train.flatten()
	sm = SMOTE(random_state=12)
	X_train, y_train = sm.fit_resample(X_train, y_train) # todas al mismo numero de muestras
	X_train = X_train[:(X_train.shape[0]//max_time)*max_time, :] # shape [muestras, 180]
	y_train = y_train[:(y_train.shape[0]//max_time)*max_time] # shape [muestras]

	if return_sequences:
		X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]]) # shape [muestras, 10, 180]
		y_train = np.reshape(y_train, [-1, y_test.shape[1]]) # shape [muestras, 10]

		# Añadir GO
		classes = np.unique(labels)
		char2numY = dict(zip(classes, range(len(classes))))
		char2numY['<GO>'] = len(char2numY)
		y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
		y_train = np.array(y_train)
	else:
		X_test = np.reshape(X_test, [-1, X_test.shape[2]]) # shape [muestras, seq_len]
		y_test = np.reshape(y_test, [-1]) # shape [muestras]

	# Create datasets
	train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64)))
	test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.int64)))

	# Create dataloaders
	train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

	return train_dataloader, val_dataloader, test_dataloader

def load_data_toy(classes= [0, 1, 2, 3], max_time=10, max_nlabel=100, batch_size=20):
    # Load the MIT-toy dataset
    df_train = pd.read_csv(root_path+"data/MIT-toy/mitbih_train.csv", header=None)
    df_test = pd.read_csv(root_path+"data/MIT-toy/mitbih_test.csv", header=None)
    # concatenate train and test
    df_train = pd.concat([df_train, df_test])

    inputs = df_train.values[:,:-1].astype(np.float32)
    labels = df_train.values[:,-1].astype(np.int64)

    print(np.unique(labels))

    # filtrar classes
    indexes_keep = np.isin(labels, classes)
    inputs = inputs[indexes_keep]
    labels = labels[indexes_keep]
    inputs = np.array([resample(signal, 280) for signal in inputs])

    print(inputs.shape, labels.shape)

    # dividir datos en secuencias
    inputs = np.array([inputs[i:i + max_time] for i in range(0, len(inputs), max_time) if i + max_time <= len(inputs)])
    labels = np.array([labels[i:i + max_time] for i in range(0, len(labels), max_time) if i + max_time <= len(labels)]).squeeze()

    # dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)

    # SMOTE
    from imblearn.over_sampling import SMOTE
    X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
    y_train= y_train.flatten()
    sm = SMOTE(random_state=12)
    X_train, y_train = sm.fit_resample(X_train, y_train) # todas a 19179
    X_train = X_train[:(X_train.shape[0]//max_time)*max_time, :]
    y_train = y_train[:(y_train.shape[0]//max_time)*max_time]
    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1]])

    # Añadir GO
    classes = np.unique(labels)
    char2numY = dict(zip(classes, range(len(classes))))
    char2numY['<GO>'] = len(char2numY)
    y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
    y_train = np.array(y_train)

    # Create datasets
    train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64)))
    
    # Create datasets
    train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64)))
    test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.int64)))
    
    # Create dataloaders
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))

    return train_dataloader, val_dataloader, test_dataloader

def get_shot(dataset, shot, num_classes):
    # Set a random seed for reproducibility
    np.random.seed(SEED)
    from torch.utils.data import Subset

    # Obtén todas las etiquetas
    all_labels = np.array([label for _, label in dataset])

    # Verifica si las etiquetas tienen múltiples time_steps
    if len(all_labels.shape) > 1:
        # Caso (samples, time_steps)
        # Solo selecciona muestras donde todas las etiquetas en time_steps son iguales
        indices_by_class = {}
        for cls in range(num_classes):
            # Encuentra muestras donde todas las etiquetas son de la misma clase
            class_mask = np.all(all_labels[:, 1:] == cls, axis=1) # [:, 1:] para que no se incluya el GO
            indices_by_class[cls] = np.where(class_mask)[0]
    else:
        # Caso original (samples,)
        indices_by_class = {cls: np.where(all_labels == cls)[0] for cls in range(num_classes)}
    
    # Selecciona muestras balanceadas
    selected_indices = []

    for cls, indices in indices_by_class.items():
        if len(indices) >= shot:  # Asegúrate de que hay suficientes muestras
            selected_indices.extend(np.random.choice(indices, size=shot, replace=False))
        else:
            # Si no hay suficientes muestras, usa reemplazo
            selected_indices.extend(np.random.choice(indices, size=shot, replace=True))
            print(f"Advertencia: Clase {cls} tiene solo {len(indices)} muestras, usando reemplazo para obtener {shot} muestras")

    # Crea el subset
    subset = Subset(dataset, selected_indices)

    # Crea el dataloader
    dataloader = DataLoader(subset, batch_size=shot, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    return dataloader