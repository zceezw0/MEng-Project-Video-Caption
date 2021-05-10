import torch
import torchvision
import torch.nn as nn
from config import myConfig
from torch.autograd import Variable

class Encoder(nn.Module):

    def __init__(self,config):
        super(Encoder, self).__init__()
        """
        Encoder of the Seq2Seq Model
        """
        self.inputEncoderDims=config.inputEncoderDims
        self.outputEncoderDims=config.outputEncoderDims
        """
        use the nn.LSTM as Encoder,and this is a single layer LSTM
        """
        self.encoderLSTM=nn.LSTM(input_size=self.inputEncoderDims,hidden_size=self.outputEncoderDims
                                 ,num_layers=1,batch_first=True)

    def forward(self, input):
        """
        :param input:input features
        :return:output and hidden states of encoder
        """
        output,states=self.encoderLSTM(input)
        return output,states

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        """
        Decoder of the Seq2Seq Model
        """
        self.init_state=None
        self.inputDecoderDims = config.inputDecoderDims
        self.outputDecoderDims = config.outputDecoderDims
        self.tokenizerOutputdims=config.tokenizerOutputdims
        """
        use the nn.LSTM as Decoder,and this is a single layer LSTM
        """
        self.decoderLSTM = nn.LSTM(input_size=self.inputDecoderDims, hidden_size=self.outputDecoderDims
                                   , num_layers=1, batch_first=True)
        """
        The output of lstm in the decoder is then sent to the fully connected layer 
        and passed through the activation function to get the final result
        """
        self.dense=nn.Linear(self.outputDecoderDims,self.tokenizerOutputdims)
        self.relu=nn.ReLU()

    def get_init_state(self,init_state):
        """
        Get the init state of decoder
        """
        self.init_state=init_state

    def forward(self, input):
        """
        :param input:input features
        :return:output and hidden states of encoder
        """
        output,states=self.decoderLSTM(input,self.init_state)
        output=self.dense(output)
        output=self.relu(output)
        return output,states

class Seq2SeqModel(nn.Module):

    def __init__(self,config):
        super(Seq2SeqModel, self).__init__()
        self.encoder=Encoder(config)
        self.decoder=Decoder(config)
    def forward(self,video_features,target_sentence):
        """
        :param video_features:input features
        :param target_sentence:word one-hot coding
        :return:get the final output of model
        """
        video_output,init_states=self.encoder(video_features)
        self.decoder.get_init_state(init_states)
        final_output=self.decoder(target_sentence)
        return final_output

if __name__=="__main__":
    model=Seq2SeqModel(myConfig)
    video_features=torch.randn(10,9,1024)
    target_sentence=torch.randn(10,9,1500)
    output=model(video_features,target_sentence)
    print(output)