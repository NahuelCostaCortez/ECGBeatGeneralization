import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        char2numY,
        n_channels=10,
        input_depth=280,
        num_units=128,
        max_time=10,
        bidirectional=False,
        embed_size=10
    ):
        """
        Parámetros:
        -----------
        char2numY: dict
            Diccionario de caracteres a índices para la salida.
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
        self.char2numY = char2numY
        self.vocab_size = len(char2numY)
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
        encoder_input_size = 128 * 7  # Ajusta a tus cálculos exactos
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
        self.dec_embedding = nn.Embedding(num_embeddings=self.vocab_size,
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
        self.fc = nn.Linear(decoder_hidden_size, self.vocab_size)


    def forward(self, inputs, dec_inputs):
        """
        Parámetros:
        -----------
        inputs: Torch tensor de forma (batch, max_time, input_depth)
            Entrada principal para el encoder.
        dec_inputs: Torch tensor de forma (batch, y_seq_length)
            Secuencia de índices para el decoder (entrada embebida).

        Retorna:
        --------
        logits: Torch tensor de forma (batch, y_seq_length, vocab_size)
            Salida final de proyección de la LSTM decoder.
        """

        batch_size = inputs.size(0)

        # 1) REFORMA/RESHAPE para pasar a conv1d
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
        #   => primero: (batch*max_time, 128 * 7)
        x = x.view(-1, x.size(1) * x.size(2))
        #   => ahora: (batch, max_time, 128 * 7)
        x = x.view(batch_size, self.max_time, -1)

        # 4) LSTM ENCODER
        #    out_enc: (batch, max_time, hidden_size * (2 si bidir)) 
        #    (h_enc, c_enc): (num_layers * num_directions, batch, hidden_size)
        out_enc, (h_enc, c_enc) = self.encoder_lstm(x)

        # 5) PREPARAR ENTRADA AL DECODER: Embedding de dec_inputs
        #    dec_inputs: (batch, y_seq_length)
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
            out_dec, _ = self.decoder_lstm(dec_embed, (h_enc_bidir, c_enc_bidir))
        else:
            out_dec, _ = self.decoder_lstm(dec_embed, (h_enc, c_enc))

        # 7) CAPA FULLY CONNECTED (logits)
        #    out_dec: (batch, y_seq_length, decoder_hidden_size)
        logits = self.fc(out_dec)  # (batch, y_seq_length, vocab_size)

        return logits