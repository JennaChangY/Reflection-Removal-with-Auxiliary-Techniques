import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
import os
import argparse
from models.model_Decomposition import Generator_cascade_withEdge
from models.model_Edge import Edge_UNet
from models.model_classifier_VGG19 import VGG19_6channel
from models.VGG import similarity_VGG_bn
from tensorboardX import SummaryWriter
from datasets.load_train_data import Zhang_n_CEILNet, Zhang_data, Zhang_test_data, CEILNet_data
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def Perceptual_loss(X, Y, gpu_num):
    
    mse = nn.MSELoss(reduction = 'mean').cuda(gpu_num)
    
    # Load vgg network
    vgg = similarity_VGG_bn(3).cuda(gpu_num)
    vgg.eval()
    
    X_features = vgg(X)
    Y_features = vgg(Y)
    perceptual_loss = 0
    w = 0.01
    for (o, t) in zip(X_features, Y_features):
        perceptual_loss += w *mse(o, t)
        w /= 0.25
    
    return perceptual_loss

def Classify_loss(classifier, B_input, R):
    
    cdata = torch.cat((B_input,R.cuda(0)),1)
    cdata = Variable(cdata).cuda(0)
    classify_output = classifier(cdata)
    
    return classify_output.sum()


# Training
def train(Dec, classifier, Edge, opt_dec, epoch, train_loader, Zhang_train_loader, writer):
    
    Dec.train()
    Edge.eval()
    classifier.eval()

    mse = nn.MSELoss(reduction = 'mean').cuda(1)
    L1 = nn.L1Loss(reduction='mean').cuda(1)
    
    acc_L2_loss = 0
    acc_L1_loss = 0
    acc_classify_loss = 0
    acc_L2_2nd_loss = 0
    acc_L1_2nd_loss = 0
    acc_classify_2nd_loss = 0
    acc_perceptual_loss = 0
    acc_perceptual_2nd_loss = 0

    iter_Zhang_loader = iter(Zhang_train_loader)
    
    for batch_idx, (I, R, B) in enumerate(train_loader):
        
        if batch_idx % 100 == 0:
            print("----train iter[%d]----" %batch_idx)
            
        batchSize = I.shape[0]
            
        ##### Train (Zhang+CEILNet) syn #####
        I, B, R = Variable(I).cuda(0), Variable(B).cuda(1), Variable(R).cuda(1)

        # Dec(I) = (B_dec,R_dec)
        B_edge = Edge(I)
        I_B_edge = Variable(torch.cat((I, B_edge),1)).cuda(0)
        res = Dec(I_B_edge)
        if len(res) % 2 == 1:
            B_dec_pre, B_dec, R_dec = res[-3], res[-1], res[-2]
        else:
            B_dec, R_dec = res[-2], res[-1]
            
        # Dec(B_dec) = (B_dec2,R_dec2)
        res = Dec(I_B_edge, fromG2=True, pre_result=B_dec)
        B_dec2, R_dec2 = res[-1], res[-2]
        
        
        ## Compute losses        
        # 1. L2 loss [(B_dec,R_dec), (B,R)]
        # L2_loss
        # L2_2nd_loss
        L2_loss = (mse(B, B_dec.cuda(1)) + mse(B, B_dec_pre.cuda(1))) * batchSize * 4
        L2_R_loss = mse(R, R_dec.cuda(1)) * batchSize * 4
        L2_2nd_loss = mse(B, B_dec2.cuda(1)) * batchSize * 4 * 2
        L2_R_2nd_loss = mse(R, R_dec2.cuda(1)) * batchSize * 4 * 2
             
        # 2. L1 loss [(B_dec,R_dec), (B,R)]
        # L1_loss
        # L1_2nd_loss
        L1_loss = (L1(B, B_dec.cuda(1)) + L1(B, B_dec_pre.cuda(1))) * batchSize
        L1_R_loss = L1(R, R_dec.cuda(1)) * batchSize
        L1_2nd_loss = L1(B, B_dec2.cuda(1)) * batchSize * 2
        L1_R_2nd_loss = L1(R, R_dec2.cuda(1)) * batchSize * 2
        
        # 3. Classify loss [B_dec]
        # classify_loss
        # classify_2nd_loss
        classify_loss = Classify_loss(classifier, B_dec, R) * 0.5
        classify_2nd_loss = Classify_loss(classifier, B_dec2, R) * 0.5 * 2
        
        # 4. Perceptual loss [(B_dec,R_dec), (B,R)]
        # perceptual_loss
        # perceptual_2nd_loss
        perceptual_loss = Perceptual_loss(B, B_dec.cuda(1), 1) * batchSize * 150
        perceptual_2nd_loss = Perceptual_loss(B, B_dec2.cuda(1), 1) * batchSize * 150 * 2
        
        ##### Train Zhang #####
        I, R, B = next(iter_Zhang_loader)
        Zhang_batchSize = I.shape[0]
        I, B = Variable(I).cuda(0), Variable(B).cuda(1)

        # Dec(I) = (B_dec,R_dec)
        B_edge = Edge(I)
        I_B_edge = Variable(torch.cat((I, B_edge),1)).cuda(0)
        res = Dec(I_B_edge)
        if len(res) % 2 == 1:
            B_dec_pre, B_dec, R_dec = res[-3], res[-1], res[-2]
        else:
            B_dec, R_dec = res[-2], res[-1]
            
        # Dec(B_dec) = (B_dec2,R_dec2)
        res = Dec(I_B_edge, fromG2=True, pre_result=B_dec)
        B_dec2, R_dec2 = res[-1], res[-2]
        
        
        ## Compute losses        
        # 1. L2 loss [(B_dec,R_dec), (B,R)]
        # L2_loss
        # L2_2nd_loss
        L2_loss += (mse(B, B_dec.cuda(1)) +  mse(B, B_dec_pre.cuda(1))) * Zhang_batchSize * 4
        L2_2nd_loss += mse(B, B_dec2.cuda(1)) * Zhang_batchSize * 4 * 2
             
        # 2. L1 loss [(B_dec,R_dec), (B,R)]
        # L1_loss
        # L1_2nd_loss
        L1_loss += (L1(B, B_dec.cuda(1)) + L1(B, B_dec_pre.cuda(1))) * Zhang_batchSize
        L1_2nd_loss += L1(B, B_dec2.cuda(1)) * Zhang_batchSize * 2
        
        # 4. Perceptual loss [(B_dec,R_dec), (B,R)]
        # perceptual_loss
        # perceptual_2nd_loss
        perceptual_loss += Perceptual_loss(B, B_dec.cuda(1), 1) * Zhang_batchSize * 150
        perceptual_2nd_loss += Perceptual_loss(B, B_dec2.cuda(1), 1) * Zhang_batchSize * 150 * 2
        
        ## Update parameters
        # Update Decomposition 
        dec_loss = L2_loss + L2_R_loss + L1_loss + L1_R_loss + classify_loss.cuda(1) + perceptual_loss + L2_2nd_loss + L2_R_2nd_loss + L1_2nd_loss + L1_R_2nd_loss + classify_2nd_loss.cuda(1) + perceptual_2nd_loss
        acc_L2_loss += (L2_loss.data / (len(train_loader.dataset)+len(Zhang_train_loader.dataset))) + (L2_R_loss.data / len(train_loader.dataset))
        acc_L1_loss += (L1_loss.data / (len(train_loader.dataset)+len(Zhang_train_loader.dataset))) + (L1_R_loss.data / len(train_loader.dataset))
        acc_classify_loss += classify_loss.data / len(train_loader.dataset)
        acc_L2_2nd_loss += (L2_2nd_loss.data / (len(train_loader.dataset)+len(Zhang_train_loader.dataset))) + (L2_R_2nd_loss.data / len(train_loader.dataset))
        acc_L1_2nd_loss += (L1_2nd_loss.data / (len(train_loader.dataset)+len(Zhang_train_loader.dataset))) + (L1_R_2nd_loss.data / len(train_loader.dataset))
        acc_classify_2nd_loss += classify_2nd_loss.data / len(train_loader.dataset)
        acc_perceptual_loss += perceptual_loss.data / (len(train_loader.dataset)+len(Zhang_train_loader.dataset))
        acc_perceptual_2nd_loss += perceptual_2nd_loss.data / (len(train_loader.dataset)+len(Zhang_train_loader.dataset))
        
        if (batch_idx % 3) != 0:
            opt_dec.zero_grad()
            dec_loss.backward(retain_graph=True)
            opt_dec.step()
    
    # Tensor board
    writer.add_scalars('train/decomposition_loss', {'L2_loss': acc_L2_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'L1_loss': acc_L1_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'classify_loss': acc_classify_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'perceptual_loss': acc_perceptual_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'L2_2nd_loss': acc_L2_2nd_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'L1_2nd_loss': acc_L1_2nd_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'classify_2nd_loss': acc_classify_2nd_loss}, epoch)
    writer.add_scalars('train/decomposition_loss', {'perceptual_2nd_loss': acc_perceptual_2nd_loss}, epoch)
    
def test(Dec, classifier, Edge, epoch, CEIL_test_loader, Zhang_test_loader, writer):

    Dec.eval()
    Edge.eval()
    classifier.eval()
    
    mse = nn.MSELoss(reduction = 'mean').cuda(1)
    L1 = nn.L1Loss(reduction='mean').cuda(1)
    
    acc_CEIL_dec_loss = 0
    acc_Zhang_dec_loss = 0
    
    for batch_idx, (I, R, B) in enumerate(CEIL_test_loader):
        
        batchSize = I.shape[0]
            
        I, B, R = Variable(I).cuda(0), Variable(B).cuda(1), Variable(R).cuda(1)

        ## Starts from I
        # Dec(I) = (B_dec,R_dec)
        with torch.no_grad():
            B_edge = Edge(I)
        I_B_edge = Variable(torch.cat((I, B_edge),1)).cuda(0)
        with torch.no_grad():
            res = Dec(I_B_edge)
        if len(res) % 2 == 1:
            B_dec_pre, B_dec, R_dec = res[-3], res[-1], res[-2]
        else:
            B_dec, R_dec = res[-2], res[-1]
            
        # Dec(B_dec) = (B_dec2,R_dec2)
        with torch.no_grad():
            res = Dec(I_B_edge, fromG2=True, pre_result=B_dec)
        B_dec2, R_dec2 = res[-1], res[-2]
        
        
        ## Compute losses
        # 1. L2 loss [(B_dec,R_dec), (B,R)]
        # L2_loss
        # L2_2nd_loss
        with torch.no_grad():
            L2_loss = (mse(B, B_dec.cuda(1)) + mse(R, R_dec.cuda(1)) + mse(B, B_dec_pre.cuda(1))) * batchSize * 4
            L2_2nd_loss = (mse(B, B_dec2.cuda(1)) + mse(R, R_dec2.cuda(1))) * batchSize * 4 * 2
             
        # 2. L1 loss [(B_dec,R_dec), (B,R)]
        # L1_loss
        # L1_2nd_loss
        with torch.no_grad():
            L1_loss = (L1(B, B_dec.cuda(1)) + L1(R, R_dec.cuda(1)) + L1(B, B_dec_pre.cuda(1))) * batchSize
            L1_2nd_loss = (L1(B, B_dec2.cuda(1)) + L1(R, R_dec2.cuda(1))) * batchSize * 2
        
        # 3. Classify loss [B_dec]
        # classify_loss
        # classify_2nd_loss
        with torch.no_grad():
            classify_loss = Classify_loss(classifier, B_dec, R) * 0.5
            classify_2nd_loss = Classify_loss(classifier, B_dec2, R) * 0.5 * 2

                
        # Test Decomposition 
        dec_loss = L2_loss.data + L1_loss.data + classify_loss.cuda(1).data + L2_2nd_loss.data + L1_2nd_loss.data + classify_2nd_loss.cuda(1).data
        acc_CEIL_dec_loss += dec_loss
        
    for batch_idx, (I, R, B) in enumerate(Zhang_test_loader):
        
        batchSize = I.shape[0]
            
        I, B = Variable(I).cuda(0), Variable(B).cuda(1)

        ## Starts from I
        # Dec(I) = (B_dec,R_dec)
        with torch.no_grad():
            B_edge = Edge(I)
        I_B_edge = Variable(torch.cat((I, B_edge),1)).cuda(0)
        with torch.no_grad():
            res = Dec(I_B_edge)
        if len(res) % 2 == 1:
            B_dec_pre, B_dec, R_dec = res[-3], res[-1], res[-2]
        else:
            B_dec, R_dec = res[-2], res[-1]
            
        # Dec(B_dec) = (B_dec2,R_dec2)
        with torch.no_grad():
            res = Dec(I_B_edge, fromG2=True, pre_result=B_dec)
        B_dec2, R_dec2 = res[-1], res[-2]
        
        
        ## Compute losses
        # 1. L2 loss [(B_dec,R_dec), (B,R)]
        # L2_loss
        # L2_2nd_loss
        with torch.no_grad():
            L2_loss = (mse(B, B_dec.cuda(1)) + mse(B, B_dec_pre.cuda(1))) * batchSize * 4
            L2_2nd_loss = mse(B, B_dec2.cuda(1)) * batchSize * 4 * 2
             
        # 2. L1 loss [(B_dec,R_dec), (B,R)]
        # L1_loss
        # L1_2nd_loss
        with torch.no_grad():
            L1_loss = (L1(B, B_dec.cuda(1)) + L1(B, B_dec_pre.cuda(1))) * batchSize
            L1_2nd_loss = L1(B, B_dec2.cuda(1)) * batchSize * 2

                
        # Test Decomposition 
        dec_loss = L2_loss.data + L1_loss.data + L2_2nd_loss.data + L1_2nd_loss.data
        acc_Zhang_dec_loss += dec_loss
            
    writer.add_scalars('test/losses', {'CEILNet_dec_loss': acc_CEIL_dec_loss / len(CEIL_test_loader.dataset)}, epoch)
    writer.add_scalars('test/losses', {'Zhang_dec_loss': acc_Zhang_dec_loss / len(Zhang_test_loader.dataset)}, epoch)
    writer.add_scalars('test/losses', {'total_dec_loss': (acc_CEIL_dec_loss+acc_Zhang_dec_loss) / (len(CEIL_test_loader.dataset)+len(Zhang_test_loader.dataset))}, epoch)
    
    
if __name__ == '__main__':

    WEIGHT_DIR = './weights/'
    SYN_DATA_DIR_CEILNET = './train_data/CEILNet/train/'
    SYN_DATA_DIR_ZHANG = './train_data/PerceptualLoss/' # combine two synthetic dataset while training
    REAL_DATA_DIR = './train_data/PerceptualLoss/real/'
    TEST_DATA_DIR_CEILNET = './train_data/CEILNet/test/'
    TEST_DATA_DIR_ZHANG = './train_data/PerceptualLoss/real_test'
    SAVE_WEIGHT_DIR = './train_models/'
    
    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--syn_train_batch_size',type=int,help='batch size of synthetic training data',default=20)
        parser.add_argument('--real_train_batch_size',type=int,help='batch size of real training data',default=4)
        parser.add_argument('--test_batch_size',type=int,help='batch size of testing data',default=20)
        parser.add_argument('--epochs',type=int,help='numbers of epoches',default=200)
        parser.add_argument('--lr',type=float,help='learning rate',default=1e-3)
        parser.add_argument('--load_hist',type=bool,help='use pretrained model',default=False)
        parser.add_argument('--test_flag',type=bool,help='Testing while training', default=False)

        parser.add_argument('--weight_dir',type=str,help='directory of models',default=WEIGHT_DIR)
        parser.add_argument('--save_weight_dir',type=str,help='directory of saving models',default=SAVE_WEIGHT_DIR)
        parser.add_argument('--syn_data_dir_ceilnet',type=str,help='directory of synthetic training data',default=SYN_DATA_DIR_CEILNET)
        parser.add_argument('--syn_data_dir_zhang',type=str,help='directory of synthetic training data',default=SYN_DATA_DIR_ZHANG)
        parser.add_argument('--real_data_dir',type=str,help='directory of real training data',default=REAL_DATA_DIR)
        parser.add_argument('--test_data_dir_ceilnet',type=str,help='directory of CEILNet synthetic testing data',default=TEST_DATA_DIR_CEILNET)
        parser.add_argument('--test_data_dir_zhang',type=str,help='directory of Zhang et al. real-world testing data',default=TEST_DATA_DIR_ZHANG)
        parser.add_argument('--size',type=int,help="256 | 128",default=256)

        opts = parser.parse_args()
        return opts

    args = get_args()
    if not os.path.isdir(args.save_weight_dir):
        os.makedirs(args.save_weight_dir)
    writer = SummaryWriter()

    # Instantiate the model
    Dec = Generator_cascade_withEdge(ns=[7,5,5], iteration=1).cuda(0)
    classifier = VGG19_6channel(num_classes=1).cuda(0)
    Edge = Edge_UNet().cuda(0)
    
    # Set learning rate
    opt_dec = optim.Adam(Dec.parameters(), lr=args.lr)

    # Load pre-trained weight
    classifier.load_state_dict(torch.load('%sReflClf' %(args.weight_dir)))
    Edge.load_state_dict(torch.load('%sEdge' %(args.weight_dir)))
    if args.load_hist:
         Dec.load_state_dict(torch.load('%sDec' %(args.weight_dir)))

    transformations = transforms.Compose([transforms.Resize((args.size, args.size)), ToTensor()])

    # Load data
    Zhang_train_root = args.real_data_dir
    Zhang_root = args.syn_data_dir_zhang
    CEILNet_train_root = args.syn_data_dir_ceilnet
    # real-world images from Zhang et al.
    Zhang_train = Zhang_data(root = Zhang_train_root, transforms = transformations)
    Zhang_train_loader = DataLoader(Zhang_train, batch_size=args.real_train_batch_size, shuffle=True)  
    # synthetic images from CEILNet and Zhang et al.
    Zhang_n_CEILNet_train = Zhang_n_CEILNet(CEILNet_root=CEILNet_train_root, Zhang_root=Zhang_root, transforms=transformations, Zhang_num_samples=562, CEILNet_num_samples=4778, Zhang_real=False)
    train_loader = DataLoader(Zhang_n_CEILNet_train, batch_size=args.syn_train_batch_size, shuffle=True)
                         
    if args.test_flag:
        Zhang_test_root = args.test_data_dir_zhang                         
        CEILNet_test_root = args.test_data_dir_ceilnet
        Zhang_test = Zhang_test_data(root=Zhang_test_root, transforms=transformations)
        CEILNet_test = CEILNet_data(root=CEILNet_test_root, transforms=transformations, num_samples=1084)
        Zhang_test_loader = DataLoader(Zhang_test, batch_size=args.test_batch_size, shuffle=True)
        CEILNet_test_loader = DataLoader(CEILNet_test, batch_size=args.test_batch_size, shuffle=True)


    # Train & test the model
    for epoch in range(1, 1 + args.epochs):        
        print("Start training[%d]" %epoch)
        Zhang_n_CEILNet_train = Zhang_n_CEILNet(CEILNet_root=CEILNet_train_root, Zhang_root=Zhang_root, transforms=transformations, Zhang_num_samples=562, CEILNet_num_samples=4778, Zhang_real=False)
        train_loader = DataLoader(Zhang_n_CEILNet_train, batch_size=args.syn_train_batch_size, shuffle=True)
        train(Dec, classifier, Edge, opt_dec, epoch, train_loader, Zhang_train_loader, writer)
        if args.test_flag:
            print("Start testing[%d]" %epoch)
            test(Dec, classifier, Edge, epoch, CEILNet_test_loader, Zhang_test_loader, writer)
        # Save the model for future use
        torch.save(Dec.state_dict(), '%sDec_%s' %(args.save_weight_dir, epoch))
    writer.close()
