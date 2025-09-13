import spacy
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn

import torchvision.models as models

spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self, freq_threshold):
        self.intToString = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stringToInt = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.intToString)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stringToInt[word] = idx
                    self.intToString[idx] = word
                    idx += 1

    def convertCaptionsToInts(self, text):
        integer_text = self.tokenizer_eng(text)

        return [
            self.stringToInt[token] if token in self.stringToInt else self.stringToInt["<UNK>"]
            for token in integer_text
        ]

class pretrainedEncoder(nn.Module):
    def __init__(self, embedding_size, train_CNN=False):
        super(pretrainedEncoder, self).__init__()

        self.train_CNN = train_CNN

        self.resnet50 = models.resnet50(weights='DEFAULT')
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, embedding_size)

        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(0.5)

        if not self.train_CNN:
            for name, param in self.resnet50.named_parameters():
                param.requires_grad = train_CNN

            for param in self.resnet50.fc.parameters():
                param.requires_grad = True

    def forward(self, images):
        features = self.resnet50(images)

        features = features.view(features.size(0), -1)

        return self.dropout_layer(self.relu_layer(features))

class lstmDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, layer_count):


        super(lstmDecoder, self).__init__()

        self.embed_layer = nn.Embedding(vocab_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_size, layer_count)
        self.linear_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout_layer(self.embed_layer(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm_layer(embeddings)

        outputs = self.linear_layer(hiddens)

        return outputs


class ImageCaptioner(nn.Module):
    def __init__(self, embedded_size, hidden_size, vocab_size, layer_count):
        super(ImageCaptioner, self).__init__()
        self.encoder = pretrainedEncoder(embedded_size)
        self.decoder = lstmDecoder(embedded_size, hidden_size, vocab_size, layer_count)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def gen_caption(self, input_image, vocab, max_length=50):
        result = []

        with torch.no_grad():
            x = self.encoder(input_image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm_layer(x, states)
                output = self.decoder.linear_layer(hiddens.squeeze(0))

                output = self.temperature_sample(output, 1)

                predicted = output.squeeze(0)

                result.append(predicted.item())
                x = self.decoder.embed_layer(predicted).unsqueeze(0)

                if vocab.intToString[predicted.item()] == "<EOS>":
                    break

        return [vocab.intToString[idx] for idx in result]
    
    def temperature_sample(self, logits, temperature):
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        
        sampled_idx = torch.multinomial(probs, 1)
        
        return sampled_idx