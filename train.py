import torch
from torch.utils.data import DataLoader
from config import myConfig
from model.Seq2Seq import Seq2SeqModel
from dataset.VideoCaptionDataset import VideoCaptionDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
import torch.nn as nn
import os
import argparse

#use which device to run model,if gpu is available,we can use cuda,if not,we can use cpu with lower speed.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load our model according to algorithm
model = Seq2SeqModel(config=myConfig)

#load dataset by pytorch
"""
trainDataloader:
It is train data generator,
    shuffle=True is to disrupt the order of data   
    drop_last=True is to drop last batch if the total amount of data cannot divide batchsize
testDataloader:
It is test data generator,
    shuffle=False 
    drop_last=False 
"""
trainDataset = VideoCaptionDataset(config=myConfig, json=myConfig.trainJson)
trainDataloader = DataLoader(dataset=trainDataset, batch_size=myConfig.BatchSize, shuffle=True, drop_last=True)
testDataset = VideoCaptionDataset(config=myConfig, json=myConfig.valJson)
testDataloader = DataLoader(dataset=testDataset, batch_size=16, shuffle=False, drop_last=False)

train_size = len(trainDataloader)
test_size = len(testDataloader)

"""
criterion: 
it is cross entropy loss
    input:softmax tensor,target
optimizer:
I use Adam optimizer to optim my model
    lr=0.0008,lr means learning rate
    weight_decay=5e-4,weight_decay means decay rate of Adam optimizer
scheduler
MultiStepLR is used to decay the learning rate
    milestones=[50, 80], gamma=0.2:decay the learning rate when epoch=50,80,and decay rate is 0.2
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.2)

#set the model to device
model.to(device)

def train(epoch):
    """
    :param epoch:epoch means training epoch
    :return:None
    """
    print('====== Train Model ======')
    train_loss=0
    #set the model to training mode
    model.train()
    for id, data in enumerate(tqdm(trainDataloader, leave=False, total=train_size)):
        features, decoderInput, decoderTarget = data
        """
        features is features of the video,it can be i3d feature.
        decoderInput is the previous word's one-hot coding,<BOS> when sentence begins
        decoderTarget is the current word's one-hot coding,<EOS> when sentence ends
        """
        features = features.view(features.size()[0], features.size()[3], -1)
        features = features.to(device)
        decoderInput = decoderInput.to(device)
        decoderTarget = decoderTarget.to(device, dtype=torch.long)
        decoderTarget = decoderTarget.view(-1, )
        #clear all gradients in the optimizer
        optimizer.zero_grad()
        #get the outputs of the model
        outputs, _ = model(features, decoderInput)
        outputs = outputs.contiguous().view(-1, myConfig.tokenizerOutputdims)
        #calculate the loss
        loss = criterion(outputs, decoderTarget)
        train_loss += loss.item()

        """
        view some intermediate results in training
        """
        if (id + 1) % 10 == 0:
            print("Train Loss is %.5f" % (train_loss / 10))
            with open("loss.txt", "a") as F:
                F.writelines("{}\n".format(train_loss / 10))
                train_loss = 0
                F.close()

        """
        backpropagation gradient,make the optimizer take effect,and update model parameters
        """
        loss.backward()
        optimizer.step()

    if (epoch+1)%5==0:
        model.eval()
        # set the model to eval mode,freeze all parameters
        total=0
        correct=0
        with torch.no_grad():
            #Prohibit the use of gradients
            print('====== Test Model ======')
            for data in tqdm(testDataloader, leave=False, total=train_size):
                """
                Similar to training, input features, get output, and calculate accuracy
                """
                features, decoderInput, decoderTarget = data
                features = features.view(features.size()[0], features.size()[3], -1)
                features = features.to(device)
                decoderInput = decoderInput.to(device)
                decoderTarget = decoderTarget.to(device, dtype=torch.long)
                decoderTarget = decoderTarget.view(-1, )
                outputs,_ = model(features, decoderInput)
                outputs = outputs.contiguous().view(-1, myConfig.tokenizerOutputdims)
                _, predicted = outputs.max(1)
                total += decoderTarget.size(0)
                correct += predicted.eq(decoderTarget).sum().item()

        #view the accuracy of the model
        accuracy=correct/total*100
        print("====== Epoch {} accuracy is {}% ====== ".format(epoch+1,accuracy))

    if (epoch+1)%10==0:
        print('====== Saving model ======')
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_train'):
            os.mkdir('checkpoint_train')
        torch.save(state, './final_checkpoint/checkpoint_%03d.pth'%(epoch+1+200))

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--debug", "-D", action="store_true")
    a.add_argument("--loss_only", "-L", action="store_true")
    args = a.parse_args()

    """
    train_data_loader:
    It is train data generator,
        use the python iter to load data
    """
    dataset = VideoCaptionDataset(config=myConfig, json=myConfig.trainJson)
    vocab = dataset.vocab
    train_data_loader = iter(dataset.train_data_loader)

    """
    build our model ,the difference with the Seq2Seq model is this model adds a reconstruction network 
    """
    decoder = None
    if myConfig.use_recon:
        reconstructor = None
        lambda_recon = torch.autograd.Variable(torch.tensor(1.), requires_grad=True)
        lambda_recon = lambda_recon.to(myConfig.device)

    """
    save training loss
    """
    train_loss = 0
    if myConfig.use_recon:
        train_dec_loss = 0
        train_rec_loss = 0

    forward_reconstructor = None
    for iteration, batch in enumerate(train_data_loader, 1):
        """
            features is features of the video,it can be i3d feature.
            encoder_outputs is the previous word's one-hot coding,<BOS> when sentence begins
            targets is the current word's one-hot coding,<EOS> when sentence ends
        """
        features, encoder_outputs, targets = batch
        encoder_outputs = encoder_outputs.to(myConfig.device)
        targets = targets.to(myConfig.device)
        targets = targets.long()
        target_masks = targets > myConfig.init_word2idx['<PAD>']

        forward_decoder=None
        decoder['model'].train()
        """
        get the output of the decoder
        """
        decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(decoder, encoder_outputs, targets, target_masks, myConfig.decoder_teacher_forcing_ratio)

        """
        if use reconstructor:
            get the forward reconstructor's loss
        """
        if myConfig.use_recon:
            reconstructor['model'].train()
            recon_loss = forward_reconstructor(decoder_hiddens, encoder_outputs, reconstructor)

        """
        if use reconstructor:
            add the reconstructor to the decoder loss,and lambda_recon means the hyperparameter(weight) of the recon loss
        if not use  reconstructor:
            The loss is decoder loss
        """
        if myConfig.use_recon:
            loss = decoder_loss + lambda_recon * recon_loss
        else:
            loss = decoder_loss

        """
        This part is for the backpropagation of gradient
            first:Clear the gradient in the optimizer,including the decoder's optimizer and reconstructor's optimizer
            second：Backpropagation the gradient
            third:
                Use the clip_grad_norm_ to clip the gradient.
                In the process of training RNN, it was found that this mechanism has a great influence on the results.
                Gradient clipping is generally used to solve the problem of gradient explosion (gradient explosion)
            fourth: Save the loss
        """
        decoder['optimizer'].zero_grad()
        if myConfig.use_recon:
            reconstructor['optimizer'].zero_grad()
        loss.backward()
        if myConfig.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(decoder['model'].parameters(), myConfig.gradient_clip)
        decoder['optimizer'].step()
        if myConfig.use_recon:
            reconstructor['optimizer'].step()
            train_dec_loss += decoder_loss.item()
            train_rec_loss += recon_loss.item()
        train_loss += loss.item()

        """
        view some intermediate results in training
        """
        if args.debug or iteration % myConfig.log_every == 0:
            n_trains = myConfig.log_every * myConfig.batch_size
            train_loss /= n_trains
            if myConfig.use_recon:
                train_dec_loss /= n_trains
                train_rec_loss /= n_trains

            train_loss = 0
            if myConfig.use_recon:
                train_dec_loss = 0
                train_rec_loss = 0

        """ 
        view some intermediate results in Validation
        """
        if args.debug or iteration % myConfig.validate_every == 0:
            val_loss = 0
            if myConfig.use_recon:
                val_dec_loss = 0
                val_rec_loss = 0
            gt_captions = []
            pd_captions = []
            """
            val_data_loader:
            It is val data generator,
                use the python iter to load data
            """
            val_data_loader = iter(dataset.val_data_loader)
            for batch in val_data_loader:
                """
                features is features of the video,it can be i3d feature.
                encoder_outputs is the previous word's one-hot coding,<BOS> when sentence begins
                targets is the current word's one-hot coding,<EOS> when sentence ends
                """
                features, encoder_outputs, targets = batch
                encoder_outputs = encoder_outputs.to(myConfig.device)
                targets = targets.to(myConfig.device)
                targets = targets.long()
                target_masks = targets > myConfig.init_word2idx['<PAD>']
                """
                Get the output of the model
                """

                """
                Same as training,save the loss
                """
                if myConfig.use_recon:
                    reconstructor['model'].eval()
                    recon_loss = forward_reconstructor(decoder_hiddens, encoder_outputs, reconstructor)

                if myConfig.use_recon:
                    loss = decoder_loss + lambda_recon * recon_loss
                else:
                    loss = decoder_loss

                if myConfig.use_recon:
                    val_dec_loss += decoder_loss.item() * myConfig.batch_size
                    val_rec_loss += recon_loss.item() * myConfig.batch_size
                val_loss += loss.item() * myConfig.batch_size

                _, _, targets = batch
                gt_idxs = targets.cpu().numpy()
                pd_idxs = decoder_output_indices.cpu().numpy()
                """
                Compare the result of GT and Predict 
                """
            n_vals = len(val_data_loader) * myConfig.batch_size
            val_loss /= n_vals
            if myConfig.use_recon:
                val_dec_loss /= n_vals
                val_rec_loss /= n_vals

        """ 
        Save the model checkpoint
        """
        if iteration % myConfig.save_every == 0:
            if not os.path.exists(myConfig.save_dpath):
                os.makedirs(myConfig.save_dpath)
                """
                get the save path
                """
            ckpt_fpath = os.path.join(myConfig.save_dpath, "{}_checkpoint.tar".format(iteration))

            """
            'iteration': training epoch
            'dec': decoder network and checkpoint
            'rec': reconstructor network and checkpoint
            'dec_opt': decoder optimizer
            'rec_opt': reconstructor optimizer
            'loss': loss value
            """
            torch.save({
                    'iteration': iteration,
                    'dec': decoder['model'].state_dict(),
                    'rec': reconstructor['model'].state_dict(),
                    'dec_opt': decoder['optimizer'].state_dict(),
                    'rec_opt': reconstructor['optimizer'].state_dict(),
                    'loss': loss,
                }, ckpt_fpath)

        if iteration == myConfig.n_iterations:
            break
if __name__=="__main__":
    for i in range(150):
        scheduler.step(i)
        train(i)
