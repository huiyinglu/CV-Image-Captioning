import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))
        
    def forward(self, features, captions):
        cap_embedding = self.word_embeddings(captions[:, :-1])
        
        # Concatenate the features and caption inputs and feed to LSTM cell(s).
        # Permutation is not required due to batch_first=True.
        # Features passed in has shape (batch_size, embed_size)
        # Insert 1 in the first dimension before concatenate ==> feature has dimension (batch_size, 1, embed_size)
        features = features.unsqueeze(1)
        
        # cancatenate along dim 1, embeddings has dimension (batch, num_seq, embed_size)
        embeddings = torch.cat((features, cap_embedding), 1) 
        
        #lstm_output, _ = self.lstm(embeddings, self.hidden)
        lstm_output, self.hidden = self.lstm(embeddings)
        
        # Convert LSTM outputs to word predictions
        outputs = self.linear(lstm_output)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        lstm_state = None
        
        # Note: input now has the dimension (batch_size, seq_num, emb_dim)
        for i in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            output = self.linear(lstm_out)
            prediction = torch.argmax(output, dim=2)
            predicted_index = prediction.item()
            sentence.append(predicted_index)
            
            if predicted_index == 1:
                break
                
            # Get next input
            inputs = self.word_embeddings(prediction)
            
        return sentence
            