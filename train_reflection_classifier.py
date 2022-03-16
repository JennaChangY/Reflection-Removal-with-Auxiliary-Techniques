import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from models.ReflectionClassifier_VGG19 import VGG19_6channel
from tensorboardX import SummaryWriter
from datasets.load_train_data import Zhang_n_CEILNet, SIR2
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

def train(model, optimizer, epoch, train_loader, loss_func, writer):
    model.train()
    
    acc_train_loss = 0
    acc_n_loss = 0
    acc_p_loss = 0
    
    for batch_idx, (I, R, B) in enumerate(train_loader):
        
        p_data = torch.cat((I,R),1)
        n_data = torch.cat((B,R),1)
        p_label = torch.ones(R.shape[0], 1)
        n_label = torch.zeros(R.shape[0], 1)
        p_data, p_label, n_data, n_label = Variable(p_data).cuda(), Variable(p_label).cuda(), Variable(n_data).cuda(), Variable(n_label).cuda()
        
        p_output = model(p_data)
        n_output = model(n_data)

        p_loss = loss_func(p_output, p_label.float())
        n_loss = loss_func(n_output, n_label.float())
        
        train_loss = p_loss + n_loss
        acc_train_loss += train_loss.data
        acc_n_loss += n_loss.data
        acc_p_loss += p_loss.data
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
    writer.add_scalars('train/losses', {'p_loss': acc_p_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/losses', {'n_loss': acc_n_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/total', {'train_loss': acc_train_loss / len(train_loader.dataset)}, epoch)

    
def test(model, test_loader, loss_func, writer):
    model.eval()
    
    acc_test_loss = 0
    correct = 0
    
    for batch_idx, (I, R, B) in enumerate(test_loader):
        
        p_data = torch.cat((I,R),1)
        n_data = torch.cat((B,R),1)
        p_label = torch.ones(R.shape[0], 1)
        n_label = torch.zeros(R.shape[0], 1)
        p_data, p_label, n_data, n_label = Variable(p_data).cuda(), Variable(p_label).cuda(), Variable(n_data).cuda(), Variable(n_label).cuda()
        
        with torch.no_grad():
            p_output = model(p_data)
            n_output = model(n_data)
        
        p_loss = loss_func(p_output, p_label.float())
        n_loss = loss_func(n_output, n_label.float())

        acc_test_loss += p_loss.data + n_loss.data
        
        n_pred = np.where(n_output.data.cpu().numpy() < 0.5, 0, 1).reshape((n_label.shape))
        p_pred = np.where(p_output.data.cpu().numpy() < 0.5, 0, 1).reshape((p_label.shape))
        correct += np.sum(n_pred == n_label.data.cpu().numpy()) + np.sum(p_pred == p_label.data.cpu().numpy())
    
    writer.add_scalars('test/losses', {'test_loss': acc_test_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test/accuracy', {'accuracy': 50. * correct / len(test_loader.dataset)}, epoch)

if __name__ == '__main__':

    WEIGHT_DIR = './weights/'
    SYN_DATA_DIR_CEILNET = './train_data/CEILNet/train/'
    SYN_DATA_DIR_ZHANG = './train_data/PerceptualLoss/' # combine two synthetic dataset while training
    TEST_DATA_DIR_SIR2 = './train_data/SIR2/'
    SAVE_WEIGHT_DIR = './train_models/'
    
    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size',type=int,help='batch size of training data',default=16)
        parser.add_argument('--test_batch_size',type=int,help='batch size of testing data',default=16)
        parser.add_argument('--epochs',type=int,help='numbers of epoches',default=200)
        parser.add_argument('--lr',type=float,help='learning rate',default=5e-6)
        parser.add_argument('--load_hist',type=bool,help='use pretrained model',default=False)
        parser.add_argument('--test_flag',type=bool,help='Testing while training', default=False)

        parser.add_argument('--weight_dir',type=str,help='directory of models',default=WEIGHT_DIR)
        parser.add_argument('--save_weight_dir',type=str,help='directory of saving models',default=SAVE_WEIGHT_DIR)
        parser.add_argument('--syn_data_dir_ceilnet',type=str,help='directory of synthetic training data',default=SYN_DATA_DIR_CEILNET)
        parser.add_argument('--syn_data_dir_zhang',type=str,help='directory of synthetic training data',default=SYN_DATA_DIR_ZHANG)
        parser.add_argument('--test_data_dir_sir2',type=str,help='directory of SIR2 testing data',default=TEST_DATA_DIR_SIR2)
        parser.add_argument('--size',type=int,help="256 | 128",default=256)

        opts = parser.parse_args()
        return opts
    

    args = get_args()
    if not os.path.isdir(args.save_weight_dir):
        os.makedirs(args.save_weight_dir)
    
    writer = SummaryWriter()
    
    # Instantiate the model
    model = VGG19_6channel(num_classes=1).cuda()
    if args.load_hist:
        model.load_state_dict(torch.load('%sReflClf' %(args.weight_dir)))

    # Choose Adam as the optimizer, initialize it with the parameters & settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.5)
    loss_func = nn.BCELoss().cuda(1)

    transformations = transforms.Compose([transforms.Resize((args.size, args.size)), ToTensor()])

    # Load data
    Zhang_root = args.syn_data_dir_zhang
    CEILNet_root = args.syn_data_dir_ceilnet
    Zhang_n_CEILNet_train = Zhang_n_CEILNet(CEILNet_root=CEILNet_root, Zhang_root=Zhang_root, transforms=transformations, Zhang_num_samples=2012, CEILNet_num_samples=4000, Zhang_real=False)
    train_loader = DataLoader(Zhang_n_CEILNet_train, batch_size=args.train_batch_size, shuffle=True)
    if args.test_flag:
        SIR2_root = args.test_data_dir_sir2
        SIR2_test = SIR2(root=SIR2_root, transforms=transformations)
        test_loader = DataLoader(SIR2_test, batch_size=args.test_batch_size, shuffle=True)
    
    # Train & test the model
    for epoch in range(1, 1 + args.epochs):  
        print("Start training[%d]" %epoch)
        Zhang_n_CEILNet_train = Zhang_n_CEILNet(CEILNet_root=CEILNet_root, Zhang_root=Zhang_root, transforms=transformations, Zhang_num_samples=2012, CEILNet_num_samples=4000, Zhang_real=False)
        train_loader = DataLoader(Zhang_n_CEILNet_train, batch_size=args.train_batch_size, shuffle=True)
        train(model, optimizer, epoch, train_loader, loss_func, writer)
        if args.test_flag:
            print("Start testing[%d]" %epoch)
            test(model, test_loader, loss_func, writer)

        # Save the model for future use
        torch.save(model.state_dict(), '%sReflClf_%s' %(args.save_weight_dir, epoch))
        
    writer.close()
