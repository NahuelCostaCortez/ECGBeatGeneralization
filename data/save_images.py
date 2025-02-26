from data import load_data
import os
import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'MIT-BIH'
train_dataloader, val_dataloader, test_dataloader = load_data(dataset_name=dataset_name)

# Diccionario para mapear números a etiquetas
label_map = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}

# Diccionario para almacenar ejemplos de cada clase
samples = {label: [] for label in ['F', 'N', 'S', 'V']}
found_counts = {label: 0 for label in ['F', 'N', 'S', 'V']}

shot = 10
# Buscar 5 muestras de cada clase en train
for batch_X, batch_y in train_dataloader:
	for idx in range(len(batch_y)):
		label_num = batch_y[idx].item()
		label = label_map[label_num]
		
		if found_counts[label] < shot:
			samples[label].append((batch_X[idx], batch_y[idx]))
			found_counts[label] += 1

	# Verificar si tenemos shot muestras de cada clase
	if all(count == shot for count in found_counts.values()):
		break

root_path = "/home/nahuel/ecg/generalization/"
# Guardar todas las muestras en archivos
save_path_train = os.path.join(root_path, 'data', dataset_name, 'ICL', 'train')
save_path_test = os.path.join(root_path, 'data', dataset_name, 'ICL', 'test')
import shutil

# Create or clear directories
for path in [save_path_train, save_path_test]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Crear una figura grande con todas las muestras de train
fig, axes = plt.subplots(4, shot, figsize=(25, 15))
fig.suptitle('Todas las muestras por clase (Train)', fontsize=16)

for i, (label, sample_list) in enumerate(samples.items()):
    for j, (X, y) in enumerate(sample_list):
        
        axes[i, j].plot(X)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
        axes[i, j].set_title(f'Clase {label} - Muestra {j+1}')

plt.tight_layout()
plt.savefig(os.path.join(save_path_train, 'all_samples.png'))
plt.close()

# Guardar muestras individuales de train
for label, sample_list in samples.items():
    for i, (X, y) in enumerate(sample_list):
        # Crear figura para esta muestra
        plt.figure(figsize=(20, 5))
        plt.plot(X)
        plt.xticks([])
        plt.yticks([])
        
        # Guardar figura
        filename = f'{label}_sample_{i+1}.png'
        plt.savefig(os.path.join(save_path_train, filename))
        plt.close()

print(f"Muestras de train guardadas en {save_path_train}")

# Guardar todas las muestras de test
test_sample_count = {label: 0 for label in ['F', 'N', 'S', 'V']}

for batch_X, batch_y in test_dataloader:
	for idx in range(len(batch_y)):
		label_num = batch_y[idx].item()
		label = label_map[label_num]
		test_sample_count[label] += 1
		
		# Crear figura para esta muestra
		plt.figure(figsize=(20, 5))
		plt.plot(batch_X[idx])
		plt.xticks([])
		
		# Guardar figura con el nombre empezando por la clase
		filename = f'{label}_{test_sample_count[label]}.png'
		plt.savefig(os.path.join(save_path_test, filename))
		plt.close()

print(f"Muestras de test guardadas en {save_path_test}")
print("Número de muestras de test guardadas por clase:")
for label, count in test_sample_count.items():
    print(f"Clase {label}: {count} muestras")