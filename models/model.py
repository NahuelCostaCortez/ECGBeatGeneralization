import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule
import cnn
from seq2seq import ConvLSTMSeq2Seq

class ECGModel(LightningModule):
    def __init__(self, model, class_weights=None):
        super().__init__()
        self.model = model
        self.class_weights = class_weights

    def forward(self, x):
        x = x.unsqueeze(1) # To convert (batch, time) to (batch, channels, time)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(output, y, weight=self.class_weights)
        else:
            loss = torch.nn.functional.cross_entropy(output, y)
        acc = (output.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        acc = (output.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'val_loss',  # Métrica a monitorear
            'interval': 'epoch',    # El programador se actualiza cada época
            'frequency': 1          # Se actualiza cada vez que se cumple el intervalo
        }
        return [optimizer], [scheduler]
    
def select_model(name, n_classes=5):
    if name == "CNN":
        model = cnn.CNN(n_classes=n_classes)
    elif name == "Seq2Seq":
        model = ConvLSTMSeq2Seq(char2numY=..., n_channels=..., input_depth=..., num_units=..., max_time=..., bidirectional=..., embed_size=...)
    return model

def fine_tune(model_name, model, train_dataloader, num_epochs=10):

    if model_name == "CNN":
        # Congelar las capas convolucionales
        model = cnn.freeze_layers(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Bucle de entrenamiento

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_dataloader:
            # Carga los datos y etiquetas en el dispositivo
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Paso 1: Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Paso 2: Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Imprimir el progreso
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}")

def evaluate(model, test_dataloader, device):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    f1 = f1_score(all_targets, all_preds, average="macro")
    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")