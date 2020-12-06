import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Resize, RandomResizedCrop
from models.model_Decomposition import Generator_cascade_withEdge
from models.model_Edge import Edge_UNet
from datasets.load_test_data import Load_Test_Data
import argparse
import os
import skimage
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def evaluate(Dec, Edge, eval_loader, result_root, save_name_list):
    
    Dec.eval()
    Edge.eval()
    
        
    total_time = 0
    for batch_idx, (I, T) in enumerate(eval_loader):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        I = Variable(I).cuda()
        
        # Edge Estimate
        with torch.no_grad():
            T_edge = Edge(I)

        I_T_edge = torch.cat((I, T_edge),1)
        # First Decomposition
        with torch.no_grad():
            res = Dec(I_T_edge)
        if len(res) % 2 == 1:
            _, T_dec, R_dec = res[-3], res[-1], res[-2]
        else:
            T_dec, R_dec = res[-2], res[-1]

        # Second Decomposition
        with torch.no_grad():
            res = Dec(I_T_edge, fromG2=True, pre_result=T_dec)
        T_dec2, R_dec2 = res[-1], res[-2]

        
        end.record()

        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
        
        
        I = np.transpose(I[0].data.cpu().numpy(), (1,2,0))
        T = np.transpose(T[0].data.cpu().numpy(), (1,2,0))
#         T_dec = np.transpose(T_dec[0].data.cpu().numpy(), (1,2,0))
#         R_dec = np.transpose(R_dec[0].data.cpu().numpy(), (1,2,0))
        T_dec2 = np.transpose(T_dec2[0].data.cpu().numpy(), (1,2,0))
        R_dec2 = np.transpose(R_dec2[0].data.cpu().numpy(), (1,2,0))
        
        # Save Results
        if not os.path.exists('%s%s'%(result_root, save_name_list[batch_idx])):
            os.makedirs('%s%s'%(result_root, save_name_list[batch_idx]))
        skimage.io.imsave('%s%s/I.png'%(result_root, save_name_list[batch_idx]), I)
        skimage.io.imsave('%s%s/T.png'%(result_root, save_name_list[batch_idx]), T)
        skimage.io.imsave('%s%s/ours.png'%(result_root, save_name_list[batch_idx]), T_dec2)
        skimage.io.imsave('%s%s/ours_R.png'%(result_root, save_name_list[batch_idx]), R_dec2)
  
    # Compute Runtime
    print("average estimate time: %f (milliseconds)" %(total_time/len(eval_loader.dataset)))
    
    

if __name__ == '__main__':
    
    DATA_DIR = './samples/test_data/'
    WEIGHT_DIR = './weights/'
    SAVE_DIR = './samples/test_results/'

    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--data_dir',type=str,help='directory of reference video frames',default=DATA_DIR)
        parser.add_argument('--weight_dir',type=str,help='directory of models',default=WEIGHT_DIR)
        parser.add_argument('--save_dir',type=str,help='directory of results',default=SAVE_DIR)
        parser.add_argument('--gpu',type=int, default=0)
        parser.add_argument('--size',type=int, default=256, help="256 | 128")

        opts = parser.parse_args()
        return opts
    
    args = get_args()  

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    cuda = True if torch.cuda.is_available() else False
    torch.cuda.set_device(args.gpu)

    # Instantiate the model
    Dec = Generator_cascade_withEdge(ns=[7,5,5], iteration=1).cuda()
    Edge = Edge_UNet().cuda()
    
    # Load pre-trained weight
    Dec.load_state_dict(torch.load('%sDec' %(args.weight_dir)))
    Edge.load_state_dict(torch.load('%sEdge' %(args.weight_dir)))
    
    transformations = transforms.Compose([transforms.Resize((args.size, args.size)), ToTensor()]) 
    
    # Load data
    dataset_test = Load_Test_Data(root = args.data_dir, transforms = transformations) 
    eval_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)    
    save_name_list = []
    save_name_list= dataset_test.get_data_list()
    
    # Evaluate
    evaluate(Dec, Edge, eval_loader, args.save_dir, save_name_list)

