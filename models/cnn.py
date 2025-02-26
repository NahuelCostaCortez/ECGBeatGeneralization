import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CNN(nn.Module):
    def __init__(self, n_classes=5):
        super(CNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=0)  # "valid" padding in Keras -> padding=0
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.1)
        
        # Block 2
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.1)
        
        # Block 3
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(p=0.1)
        
        # Block 4
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, padding=0)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.drop4 = nn.Dropout(p=0.2)
        
        # Fully connected
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=n_classes)
        
        # Optional: Softmax as a final layer. However, when using nn.CrossEntropyLoss,
        # you typically do NOT apply softmax in the forward pass. We'll show both ways.
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x shape should be (batch_size, 1, 187) in PyTorch 
        since it's (batch, channels, length).
        """

        x = x.unsqueeze(1) # To convert (batch, time) to (batch, channels, time)

        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        
        # Global max pool across the temporal dimension
        # (batch, channels, length) -> (batch, channels)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.drop4(x)
        
        # Dense (fully connected) layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # If you are using nn.CrossEntropyLoss, you usually do NOT apply softmax here.
        # The CrossEntropyLoss in PyTorch expects raw logits.
        # But if you want to replicate the exact Keras output, you can do:
        #x = self.softmax(x)
        return x
    
    def configure_optimizers(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss',  # Métrica a monitorear
            'interval': 'epoch',    # El programador se actualiza cada época
            'frequency': 1          # Se actualiza cada vez que se cumple el intervalo
        }
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x, y)
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(output, y, weight=self.class_weights)
        else:
            loss = torch.nn.functional.cross_entropy(output, y)
        acc = (output.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss, acc
    
    def evaluate(self, test_dataloader, device):
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.metrics import classification_report
        self.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        f1 = f1_score(all_targets, all_preds, average="macro")
        acc = accuracy_score(all_targets, all_preds)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(all_targets, all_preds))
    

def freeze_layers(model):
    """
    Freezes all convolutional layers while keeping FC layers trainable
    """
    # Freeze all conv layers
    for name, param in model.named_parameters():
        if any(conv_name in name for conv_name in ['conv', 'pool', 'drop']):
            param.requires_grad = False
        
    # Verify which layers will be trained
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    return model

'''
class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=0)
        self.conv5 = nn.Conv1d(32, 256, kernel_size=3, padding=0)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv5(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''