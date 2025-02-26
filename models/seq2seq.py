import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        n_classes,
        n_channels=10,
        input_depth=280,
        num_units=128,
        max_time=10,
        bidirectional=True,
        embed_size=10
    ):
        """
        Parámetros:
        -----------
        n_classes: dict
            Número de clases.
        n_channels: int
            Número de canales en la entrada para la parte convolucional.
        input_depth: int
            Dimensión "profunda" de la entrada (por ejemplo, 280).
        num_units: int
            Número de unidades en la LSTM.
        max_time: int
            Longitud de la secuencia en el tiempo (se usa para reshapes).
        bidirectional: bool
            Indica si se utiliza LSTM bidireccional en el encoder.
        embed_size: int
            Tamaño de la incrustación (embedding) para el decoder.
        """

        super(ConvLSTMSeq2Seq, self).__init__()
        self.num_classes = n_classes
        self.n_channels = n_channels
        self.input_depth = input_depth
        self.num_units = num_units
        self.max_time = max_time
        self.bidirectional = bidirectional
        self.embed_size = embed_size

        # -----------
        # CONVOLUCIONES
        # -----------
        # (batch*max_time, n_channels, input_depth/n_channels) -> conv -> pool
        # Se usa padding=1 para aproximar el 'same' de TF (con kernel_size=2).
        # Ajusta si deseas replicar exactamente la dimensión.
        self.conv1 = nn.Conv1d(in_channels=n_channels,
                               out_channels=32,
                               kernel_size=2,
                               stride=1,
                               padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32,
                               out_channels=64,
                               kernel_size=2,
                               stride=1,
                               padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=2,
                               stride=1,
                               padding=1)

        # -----------
        # ENCODER LSTM
        # -----------
        # El tamaño de entrada al LSTM encoder será el resultado de
        # "flatten" la salida de conv3 + pools.
        # Para un ejemplo típico:
        #   - input_depth=280, n_channels=10 => 280/10=28 "width"
        #   - conv+pool1 => ~14 (aprox)
        #   - conv+pool2 => ~7  (aprox)
        #   - conv3 => se mantiene en ~7
        #   => 128 canales * 7 = 896
        # Ajusta según el cálculo exacto final del padding/stride.
        encoder_input_size = 128 * 8  # Ajusta a tus cálculos exactos
        self.encoder_lstm = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=num_units,
            batch_first=True,
            bidirectional=bidirectional
        )

        # -----------
        # DECODER
        # -----------
        # Embedding para la entrada del decoder
        self.dec_embedding = nn.Embedding(num_embeddings=self.num_classes+1,
                                          embedding_dim=embed_size)

        # Si el encoder es bidireccional, el estado oculto tendrá 2*num_units
        decoder_input_size = embed_size
        decoder_hidden_size = num_units * 2 if bidirectional else num_units

        self.decoder_lstm = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=decoder_hidden_size,
            batch_first=True
        )

        # Capa final de proyección a la dimensión de vocabulario
        self.fc = nn.Linear(decoder_hidden_size, self.num_classes)


    def forward(self, inputs, dec_inputs, training=True):
        """
        Parámetros:
        -----------
        inputs: Torch tensor de forma (batch, max_time, input_depth)
            Entrada principal para el encoder.
        dec_inputs: Torch tensor de forma (batch, y_seq_length)
            Secuencia de índices para el decoder (entrada embebida).

        Retorna:
        --------
        logits: Torch tensor de forma (batch, y_seq_length, num_classes)
            Salida final de proyección de la LSTM decoder.
        """

        batch_size = inputs.size(0)

        # 1) RESHAPE para pasar a conv1d
        #    TF: tf.reshape(inputs, [-1, n_channels, input_depth/n_channels])
        #    => (batch*max_time, n_channels, 28) en el ejemplo
        x = inputs.view(-1, self.n_channels, self.input_depth // self.n_channels)

        # 2) CONVOLUCIONES + POOL
        x = F.relu(self.conv1(x))   # (batch*max_time, 32, ~28)
        x = self.pool1(x)          # (batch*max_time, 32, ~14)

        x = F.relu(self.conv2(x))   # (batch*max_time, 64, ~14)
        x = self.pool2(x)          # (batch*max_time, 64, ~7)

        x = F.relu(self.conv3(x))   # (batch*max_time, 128, ~7)

        # 3) REFORMA para (batch, max_time, -1)
        #   => (batch*max_time=320, 128 * 8=1024)
        x = x.view(-1, x.size(1) * x.size(2))
        #   => ahora: (batch, max_time, 128 * 7)
        x = x.view(batch_size, self.max_time, -1)

        # 4) LSTM ENCODER
        #    out_enc: (batch, max_time, hidden_size * (2 si bidir)) 
        #    (h_enc, c_enc): (num_layers * num_directions, batch, hidden_size)
        out_enc, (h_enc, c_enc) = self.encoder_lstm(x)

        # 5) PREPARAR ENTRADA AL DECODER: Embedding de dec_inputs
        #    dec_inputs: (batch, y_seq_length)
        if training:
            dec_inputs = dec_inputs[:,:-1] # Ignore the last token to fit the decoder input
        dec_embed = self.dec_embedding(dec_inputs)  # (batch, y_seq_length, embed_size)

        # 6) ESTADO INICIAL DEL DECODER
        #    Si no es bidireccional, podemos pasar (h_enc, c_enc) directamente.
        #    Si es bidireccional, concatenamos forward y backward en la dimensión hidden.
        if self.bidirectional:
            # h_enc, c_enc son shape (2, batch, num_units).
            # Los concatenamos a lo largo de la dimensión hidden, para que quede
            # shape (1, batch, 2*num_units) y sea compatible con el decoder.
            h_enc_bidir = torch.cat((h_enc[0], h_enc[1]), dim=1).unsqueeze(0)
            c_enc_bidir = torch.cat((c_enc[0], c_enc[1]), dim=1).unsqueeze(0)
            # out_dec: (batch, y_seq_length, decoder_hidden_size)
            #print("dec input:", dec_inputs.shape)
            #print("dec embed:", dec_embed.shape)
            out_dec, _ = self.decoder_lstm(dec_embed, (h_enc_bidir, c_enc_bidir))
        else:
            out_dec, _ = self.decoder_lstm(dec_embed, (h_enc, c_enc))

        # 7) CAPA FULLY CONNECTED (logits)
        #    out_dec: (batch, y_seq_length, decoder_hidden_size)
        logits = self.fc(out_dec)  # (batch, y_seq_length, num_classes)

        return logits
    
    def configure_optimizers(self, learning_rate=1e-3):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        return [optimizer], [] # No scheduler for now
    
    def training_step(self, logits, y):
        y = y[:,1:] # Ignore the first token ON
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # Reshape para que CrossEntropyLoss funcione con [N, C], donde
        # N = batch_size * seq_length, C = num_classes
        logits_flat = logits.view(-1, self.num_classes)      # [batch_size*seq_length, num_classes]
        targets_flat = y.contiguous().view(-1)               # [batch_size*seq_length]

        # Pérdida base (entropía cruzada)
        loss_ce = criterion(logits_flat, targets_flat) # Escalar (float)

        # === Agregar regularización L2 (similar a tf.nn.l2_loss) ===
        # tf.nn.l2_loss(v) = sum(v^2) / 2. Por consistencia, se multiplica luego por beta.
        l2_reg = 0.0
        for name, param in self.named_parameters():
            # Omitir bias
            if 'bias' not in name:
                l2_reg += 0.5 * torch.sum(param**2)

        beta = 0.001
        loss = loss_ce + beta * l2_reg
        acc = (logits.argmax(dim=2) == y).float().mean()
        return loss, acc
    
    def fine_tune(self, train_dataloader, num_epochs=10):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)  # Mover el modelo al dispositivo correcto
        optimizers, _ = self.configure_optimizers()
        optimizer = optimizers[0]  # Obtener el optimizador
        
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            running_acc = 0.0
            
            for batch in train_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()  # Resetear gradientes
                
                logits = self(inputs, targets)
                loss, acc = self.training_step(logits, targets)
                
                loss.backward()  # Calcular gradientes
                optimizer.step()  # Actualizar pesos
                
                running_loss += loss.item()
                running_acc += acc.item()
            
            avg_loss = running_loss / len(train_dataloader)
            avg_acc = running_acc / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    def evaluate(self, test_dataloader, device="cpu"):
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.metrics import classification_report
        self.to(device)
        self.eval()
        acc = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                dec_input = torch.zeros((len(inputs), 1), dtype=torch.long) + torch.unique(targets)[-1] # ON
                for _ in range(10):
                    outputs = self(inputs, dec_input, training=False)
                    prediction = outputs[:, -1].argmax(dim=-1)
                    dec_input = torch.cat([dec_input, prediction[:, None]], dim=1)

                acc.append(dec_input[:, 1:] == targets) # full sequence because targets don't have ON in test
                y_true = targets.flatten()
                y_pred = dec_input[:, 1:].flatten()
                all_preds.append(y_pred)
                all_targets.append(y_true)
            
        acc = np.concatenate(acc).mean()
        print(f"Accuracy own: {acc:.4f}")
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        f1 = f1_score(all_targets, all_preds, average="macro")
        acc = accuracy_score(all_targets, all_preds)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(all_targets, all_preds))