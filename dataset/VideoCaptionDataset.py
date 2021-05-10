import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from torch.utils.data import Dataset
import json
from config import myConfig
import os

class VideoCaptionDataset(Dataset):
    def __init__(self,config,json):
        """
        VideoCaptionDataset Class config
        """
        self.config=config
        self.numDecoderTokens=config.numDecoderTokens
        self.vocaburayList=[]
        self.contentList=[]
        self.JsonPath=json
        self.JsonContent=None

        self.inputSentenceLength=config.inputSentenceLength

        """
        Generate the Vocabulary Tokenizer of the dataset according to the json
        """
        self.Tokenizer=Tokenizer(self.numDecoderTokens)
        self.getTokenizer()

        """
        Get the features 
        """
        self.featuresPath=config.featuresPath
        self.features = self.loadFeatures()

        self.encoderInput,self.decoderInput,self.decoderTarget,self.videoID=self.loadDataset(self.contentList)

    def getTokenizer(self):
        with open(self.JsonPath, encoding='utf-8-sig', errors='ignore') as file:
            self.JsonContent = json.load(file, strict=False)
            for video in self.JsonContent.keys():
                content=self.JsonContent[video]
                for caption in content['caption']:
                    """
                    Add the begin signal and end signal of the sentence
                    """
                    caption = "<bos> " + caption + " <eos>"
                    if len(caption.split()) > self.config.longestSentence or len(caption.split()) < self.config.shortestSentence:
                        """
                        Discard sentences that are too long or too short
                        """
                        continue
                    else:
                        self.contentList.append([caption,video])

        for content in self.contentList:
            self.vocaburayList.append(content[0])
        """
        Use Tokenizer to encode all the words that have appeared
        """
        self.Tokenizer.fit_on_texts(self.vocaburayList)

    def loadDataset(self, datasetList):
        """
        :param datasetList:list of [caption,videoID]
        :return:encoderInput,decoderInput,decoderTarget,videoIDS
        """
        encoderInput = []
        decoderInput = []
        decoderTarget = []
        videoIDS = []
        videoCaptions = []
        for caption,videoID in datasetList:
            videoCaptions.append(caption)
            videoIDS.append(videoID)
            encoderInput.append(self.features[videoID])
        """
        Encode sentences into one-hot coding
        """
        videoSequences = self.Tokenizer.texts_to_sequences(videoCaptions)
        videoSequences = np.array(videoSequences)
        """
        Pad all sentences to a uniform length
        """
        videoSequences = pad_sequences(videoSequences, padding='post', truncating='post',maxlen=self.inputSentenceLength)
        for id in range(len(videoSequences)):
            target = to_categorical(videoSequences[id], self.numDecoderTokens)
            decoderInput.append(target[:-1])
            decoderTarget.append(videoSequences[id][1:])
        return encoderInput,decoderInput,decoderTarget,videoIDS

    def loadFeatures(self):
        featureDict={}
        """
        load feature npy file
        """
        featureFiles=os.listdir(self.featuresPath)
        for featureFile in featureFiles:
            videoName=featureFile.split(".")[0]
            featureDict[videoName]=os.path.join(self.featuresPath,featureFile)
        return featureDict

    def __getitem__(self, index):
        features=np.load(self.encoderInput[index])
        decoderInput=self.decoderInput[index]
        decoderTarget=self.decoderTarget[index]
        return features,decoderInput,decoderTarget

    def __len__(self):
        return len(self.encoderInput)

if __name__=="__main__":
    dataset=VideoCaptionDataset(myConfig,myConfig.totalJson)
    print(1)


