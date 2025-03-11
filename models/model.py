from lightning import LightningModule
from models import cnn
from models.seq2seq import ConvLSTMSeq2Seq

class ECGModel(LightningModule):
    def __init__(self, model, class_weights=None, lr=1e-3):
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        self.lr = lr

    def forward(self, x):
        x, y = x
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self((x,y))
        loss, acc = self.model.training_step(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self((x,y))
        loss, acc = self.model.training_step(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers(learning_rate=1e-3)
    
    def evaluate(self, test_dataloader, device):
        self.model.evaluate(test_dataloader, device)

    def fine_tune(self, train_dataloader, num_epochs=10):
        self.model.fine_tune(train_dataloader, num_epochs)
    
def select_model(name, n_classes=5, segment_size=280, num_units=128, max_time=10, bidirectional=True, embed_size=10):
    if name == "CNN":
        model = cnn.CNN(n_classes=n_classes)
    elif name == "Seq2Seq":
        model = ConvLSTMSeq2Seq(n_classes=n_classes, 
                                n_channels=10, 
                                input_depth=segment_size, 
                                num_units=num_units, 
                                max_time=max_time, 
                                bidirectional=bidirectional, 
                                embed_size=embed_size)
    return model

'''
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
            outputs = model((inputs, labels))
            loss = criterion(outputs, labels)
            
            # Paso 2: Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Imprimir el progreso
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}")

    return model
'''