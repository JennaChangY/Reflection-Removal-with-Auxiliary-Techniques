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
from models.model_Edge import Edge_UNet
from tensorboardX import SummaryWriter
from datasets.load_train_data import Zhang_n_CEILNet, Zhang_test_data
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def computeGradient(image):
    
    image_grad = np.zeros((1, image.shape[0], image.shape[1]))
    gradients = np.gradient(image)
    x_grad = gradients[0]
    y_grad = gradients[1]
    image_grad[0,:,:] = np.sqrt(np.power(x_grad, 2) + np.power(y_grad, 2))
    
    return image_grad

def bgr2gray(bgr):   
    
    bgr = bgr.data.cpu().numpy()
    b, g, r = bgr[0,:,:], bgr[1,:,:], bgr[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def normalization(x):

    maxi = np.max(x)
    mini = np.min(x)
    x = (x - mini) / (maxi - mini)
    
    return x

# Training
def train_edge(Edge, opt_edge, epoch, train_loader, writer):
    
    Edge.train()

    loss_gpu_num = 1
    l1 = nn.L1Loss().cuda(loss_gpu_num)
    mse = nn.MSELoss().cuda(loss_gpu_num)
    
    acc_edge_L1_loss = 0
    acc_edge_L2_loss = 0
    
    for batch_idx, (I, R, B) in enumerate(train_loader):
        
        if batch_idx % 100 == 0:
            print("----train iter[%d]----" %batch_idx)
        
        batch_size = B.shape[0]

        # convert to gray scale
        B_edge_gt = np.zeros((batch_size, 1, B.shape[2], B.shape[3]))
        for i in range(batch_size):
            B_gray = bgr2gray(B[i])
            B_edge_gt[i] = normalization(computeGradient(B_gray))
            
        # Compute gradient ground truth
        B_edge_gt = torch.from_numpy(B_edge_gt).cuda()
        B_edge_gt = B_edge_gt.type(torch.FloatTensor)
        B_edge_gt = Variable(B_edge_gt).cuda()
        
        # Extract Edge
        I = Variable(I).cuda()
        B_edge = Edge(I)     

        # Compute losses
        edge_L1_loss = l1(B_edge, B_edge_gt) * batch_size
        edge_L2_loss = mse(B_edge, B_edge_gt) * batch_size
        
        # Update edge extractor
        edge_loss = edge_L1_loss + edge_L2_loss
        opt_edge.zero_grad()
        edge_loss.backward(retain_graph=True)
        opt_edge.step() 
        
        acc_edge_L1_loss += edge_L1_loss.data
        acc_edge_L2_loss += edge_L2_loss.data
        
    # Tensorboard
    writer.add_scalars('train_loss', {'edge_L1_loss': acc_edge_L1_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train_loss', {'edge_L2_loss': acc_edge_L2_loss / len(train_loader.dataset)}, epoch)

def test_edge(Edge, epoch, test_loader, writer):
          
    Edge.eval()

    loss_gpu_num = 1
    l1 = nn.L1Loss().cuda(loss_gpu_num)
    mse = nn.MSELoss().cuda(loss_gpu_num)
    
    acc_edge_L1_loss = 0
    acc_edge_L2_loss = 0
        
    # Zhang
    for batch_idx, (I, R, B) in enumerate(test_loader):

        batch_size = B.shape[0]
        
        # convert to gray scale
        B_edge_gt = np.zeros((B.shape[0], 1, B.shape[2], B.shape[3]))
        for i in range(B.shape[0]):
            B_gray = bgr2gray(B[i])
            B_edge_gt[i] = normalization(computeGradient(B_gray))
        
        # Compute gradient ground truth
        B_edge_gt = torch.from_numpy(B_edge_gt).cuda()
        B_edge_gt = B_edge_gt.type(torch.FloatTensor)
        B_edge_gt = Variable(B_edge_gt).cuda()
           
        ## Extract Edge
        I = Variable(I).cuda()
        with torch.no_grad():
            B_edge = Edge(I)

        # Compute losses
        with torch.no_grad():
            edge_L1_loss = l1(B_edge, B_edge_gt) * batch_size
            edge_L2_loss = mse(B_edge, B_edge_gt) * batch_size
        
        # Test Edge Extractor
        acc_edge_L1_loss += edge_L1_loss.data
        acc_edge_L2_loss += edge_L2_loss.data

    writer.add_scalars('test_loss', {'edge_L1_loss': acc_edge_L1_loss / len(test_loader.dataset)}, epoch)
    writer.add_scalars('test_loss', {'edge_L2_loss': acc_edge_L2_loss / len(test_loader.dataset)}, epoch)
    
if __name__ == '__main__':
    
    WEIGHT_DIR = './weights/'
    SYN_DATA_DIR_CEILNET = './train_data/CEILNet/train/'
    SYN_DATA_DIR_ZHANG = './train_data/PerceptualLoss/' # combine two synthetic dataset while training
    REAL_DATA_DIR = './train_data/PerceptualLoss/real/'
    TEST_DATA_DIR_ZHANG = './train_data/PerceptualLoss/real_test/'
    SAVE_WEIGHT_DIR = './train_models/'
    
    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size',type=int,help='batch size of training data',default=16)
        parser.add_argument('--test_batch_size',type=int,help='batch size of testing data',default=16)
        parser.add_argument('--epochs',type=int,help='numbers of epoches',default=200)
        parser.add_argument('--lr',type=float,help='learning rate',default=1e-3)
        parser.add_argument('--load_hist',type=bool,help='use pretrained model',default=False)
        parser.add_argument('--test_flag',type=bool,help='Testing while training', default=False)

        parser.add_argument('--weight_dir',type=str,help='directory of models',default=WEIGHT_DIR)
        parser.add_argument('--save_weight_dir',type=str,help='directory of saving models',default=SAVE_WEIGHT_DIR)
        parser.add_argument('--syn_data_dir_ceilnet',type=str,help='directory of synthetic training data',default=SYN_DATA_DIR_CEILNET)
        parser.add_argument('--syn_data_dir_zhang',type=str,help='directory of synthetic training data',default=SYN_DATA_DIR_ZHANG)
        parser.add_argument('--real_data_dir',type=str,help='directory of real training data',default=REAL_DATA_DIR)
        parser.add_argument('--test_data_dir_zhang',type=str,help='directory of Zhang et al. real-world testing data',default=TEST_DATA_DIR_ZHANG)
        parser.add_argument('--size',type=int,help="256 | 128",default=256)

        opts = parser.parse_args()
        return opts
    

    args = get_args()
    if not os.path.isdir(args.save_weight_dir):
        os.makedirs(args.save_weight_dir)
    
    writer = SummaryWriter()
    
    # Instantiate the model
    Edge = Edge_UNet().cuda()
    if args.load_hist:
        Edge.load_state_dict(torch.load('%sEdge' %(args.weight_dir)))
#     if torch.cuda.is_available():
#         Edge = nn.DataParallel(Edge, device_ids=[0,1])
    
    opt_edge = optim.Adam(Edge.parameters(), lr=args.lr)
    
    transformations = transforms.Compose([transforms.Resize((args.size, args.size)), ToTensor()])
    
    # Load data
    Zhang_train_root = args.real_data_dir
    Zhang_root = args.syn_data_dir_zhang
    CEILNet_root = args.syn_data_dir_ceilnet
    Zhang_n_CEILNet_train = Zhang_n_CEILNet(CEILNet_root=CEILNet_root, Zhang_root=Zhang_root, transforms=transformations, Zhang_num_samples=1000, CEILNet_num_samples=1000, Zhang_real=True)
    train_loader = DataLoader(Zhang_n_CEILNet_train, batch_size=args.train_batch_size, shuffle=True)
    if args.test_flag:
        Zhang_test_root = args.test_data_dir_zhang
        Zhang_test = Zhang_test_data(root=Zhang_test_root, transforms=transformations)
        Zhang_test_loader = DataLoader(Zhang_test, batch_size=args.test_batch_size, shuffle=True)

    # Train & test the model
    for epoch in range(1, 1 + args.epochs):
        print("Start training[%d]" %epoch)
        Zhang_n_CEILNet_train = Zhang_n_CEILNet(CEILNet_root=CEILNet_root, Zhang_root=Zhang_root, transforms=transformations, Zhang_num_samples=1000, CEILNet_num_samples=1000, Zhang_real=True)
        train_loader = DataLoader(Zhang_n_CEILNet_train, batch_size=args.train_batch_size, shuffle=True)
        train_edge(Edge, opt_edge, epoch, train_loader, writer)
        if args.test_flag:
            print("Start testing[%d]" %epoch)
            test_edge(Edge, epoch, Zhang_test_loader, writer)

        # Save the model for future use
        torch.save(Edge.state_dict(), '%sEdge_%s' %(args.save_weight_dir, epoch))
        
    writer.close()

