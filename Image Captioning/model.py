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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,batch_first=True):
        
        super(DecoderRNN,self).__init__()        
        self.embed = nn.Embedding(vocab_size, embed_size)        
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0, 
                             batch_first=True
                           )
        
        self.fcl = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fcl.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)
        
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        captions = self.embed(captions)
        
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions),dim=1)
        outputs, _ = self.lstm(inputs)
        
        outputs = self.fcl(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        
        outputs = []
        count = 0
        word_info = None
        
        while count < max_len and word_info != 1 :
            
            output_lstm, states = self.lstm(inputs, states)
            output = self.fcl(output_lstm)
            
            prob, word = output.max(2)
            
            word_info = word.item()
            outputs.append(word_info)
            
            inputs = self.embed(word)
            
            count+=1
        
        return outputs