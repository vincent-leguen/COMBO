# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from path import Path
import os
import math
import random
from glob import glob
import os.path as osp
import imageio
import sys
sys.path.append('/home/F07773/PrevisionPV/optical_flow/gefolki/python')
from algorithm import EFolki, GEFolki
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from utils import flow_viz
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

'''
VALIDATE_INDICES_KITTI2015 = [10, 11, 12, 25, 26, 30, 31, 40, 41, 42, 46, 52, 53, 72, 73, 74, 75, 76, 80, 81, 85, 86, 95, 96, 97, 98, 104, 116, 117, 120, 121, 126, 127, 153, 172, 175, 183, 184, 190, 199]
SINTEL_VALIDATE_INDICES = [
    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 340, 341, 342, 343, 344,
    345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364, 536, 537, 538, 539,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 659, 660, 661,
    662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
    674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
    686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
    967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978,
    979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990,
    991]
'''    
    
random.seed(0)
SINTEL_VALIDATE_INDICES = random.sample(range(1000), 130)

random.seed(0)
VALIDATE_INDICES_KITTI2015 = random.sample(range(200), 40)
    
    
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.occlusion_list = []
        self.wp_list = []
        self.wa_list = []
        self.gt_uncert_list = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            #print('TEST img1 ', img1.shape,  ' flow ', flow.shape, ' folki ',folki_12.shape)   
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1]) 
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)       
        
        if len(self.flow_list) > 0:
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
            
        else:
            flow = np.zeros((img1.shape[0],img1.shape[1],2))
            valid = np.ones((img1.shape[0],img1.shape[1]))

            
        #print('flow list ',len(self.flow_list), ' image_list ',len(self.image_list), ' wp  ',len(self.wp_list) ,' wa  ',len(self.wa_list))
        if len(self.occlusion_list)>0:
            occl = frame_utils.read_gen(self.occlusion_list[index])          
            occl = np.array(occl).astype(np.uint8)
            occl = torch.from_numpy(occl // 255).bool()          
            
        else:
            occl = np.zeros_like(img1)   
               
        
        flow = np.array(flow).astype(np.float32)
        
        
        if len(self.wp_list) > 0:
            wp = frame_utils.read_gen(self.wp_list[index])         
            wa = frame_utils.read_gen(self.wa_list[index])     
            gt_uncert = frame_utils.read_gen(self.gt_uncert_list[index])     
        else:
            wp = np.zeros_like(flow)  
            wa = np.zeros_like(flow)  
            gt_uncert = np.zeros_like(valid)  
            
        
        #if len(self.folki_list) > 0:
        #    #folki_12 = np.load(self.folki_list[index]) 
        #    folki_12 = frame_utils.read_gen(self.folki_list[index])         
        #    folki_12 = np.array(folki_12).astype(np.float32).transpose(1,2,0)
        #else:
        #    folki_12 = np.zeros_like(flow)     

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, wp, wa, gt_uncert, valid = self.augmentor(img1, img2, flow, wp, wa, gt_uncert, valid)
            else:
                img1, img2, flow, wp, wa, gt_uncert, occl = self.augmentor(img1, img2, flow, wp, wa, gt_uncert, occl)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #occl = torch.from_numpy(occl).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()        
        wp = torch.from_numpy(wp).permute(2, 0, 1).float()  
        wa = torch.from_numpy(wa).permute(2, 0, 1).float()  
            

        #occl = np.zeros_like(img1)    
        #if not self.sparse:
        #    valid = np.ones((img1.shape[1],img1.shape[2]))     
        
        valid_before = valid
        if valid is not None:
            valid = torch.from_numpy(valid).double()
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            valid = valid.double()
            #valid = torch.ones((img1.shape[1],img1.shape[2]))   
                  

            
        #print('img1 ', img1.shape,  ' flow ', flow.shape, ' occl ', occl.shape)   
        if occl.shape != valid.shape:
            occl = torch.zeros_like(valid)   
        #print('img1 ', img1.shape,  ' flow ', flow.shape,' valid ',valid.shape, ' after ',type(valid), occl.shape, type(occl))  
        return img1, img2, flow,  wp, wa, gt_uncert, valid , occl


            
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        #self.folki_list = v * self.folki_list
        self.image_list = v * self.image_list
        self.wp_list = v * self.wp_list
        self.wa_list = v * self.wa_list
        self.gt_uncert_list = v * self.gt_uncert_list
        return self
        
    def __len__(self):
        return len(self.image_list)
 

    
class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/raid/F07773/optical_flow/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        Wp = sorted(glob(osp.join(root,'supervision1', 'Wp_*.flo')))
        Wa = sorted(glob(osp.join(root,'supervision1', 'Wa_*.flo')))
        gt_uncert = sorted(glob(osp.join(root,'supervision1', 'gt_uncert_*.npy')))
        assert (len(images)//2 == len(flows))
        #print('LEN:  images ', len(images), len(flows), len(Wp), len(Wa))
            
        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]
                self.wp_list += [ Wp[i] ]
                self.wa_list += [ Wa[i] ]
                self.gt_uncert_list += [ gt_uncert[i] ]
                
                ### FOLKI flows
                I1_temp =  images[2*i].rsplit("/",maxsplit=1) # par ex: '.../116/12007.png' -> I1_temp[0]='.../116' , I1_temp[1]='12007.png'
                folder = I1_temp[0] # '.../116'
                I1_temp1 = I1_temp[1].split(".") # I1_temp1[0] = '12007'
                I1_name = I1_temp1[0]
                
                I2_temp =  images[2*i+1].rsplit("/",maxsplit=1)
                I2_temp1 = I2_temp[1].split(".") 
                I2_name = I2_temp1[0]
                name_I12 = Path( folder + '/' + I1_name + '-' + I2_name + '.npy')
                #name_I21 = Path( folder + '/' + I2_name + '-' + I1_name + '.npy')                 
                #self.folki_list += [ [name_I12] ]                
        #print(' CHAIRS image_list ', len(self.image_list) , ' flow list ',len(self.flow_list), ' wp_list ', len(self.wp_list),  ' gt_uncert_list ', len(self.gt_uncert_list))

                

                

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', mode='training', root='/raid/F07773/optical_flow/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        occlusion_root = osp.join(root, split, 'occlusions')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            occlusion_list = sorted(glob(osp.join(occlusion_root, scene, '*.png')))

            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id
                if mode == 'validation':
                    self.occlusion_list += [ occlusion_list[i]  ]
                
            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, 'frame_*.flo')))
                self.wp_list += sorted(glob(osp.join(flow_root, scene, 'Wp1_*.flo')))
                self.wa_list += sorted(glob(osp.join(flow_root, scene, 'Wa1_*.flo')))
                self.gt_uncert_list += sorted(glob(osp.join(flow_root, scene, 'gt_uncert1_*.npy')))
                
        #print('SINTEL image_list ', len(self.image_list) , ' flow list ',len(self.flow_list), ' wp_list ', len(self.wp_list),  ' gt_uncert_list ', len(self.gt_uncert_list))
        
        
        
class MpiSintel_resplit(FlowDataset):
    def __init__(self, aug_params=None, split='training', mode='training', root='/raid/F07773/optical_flow/Sintel', dstype='clean'):
        super(MpiSintel_resplit, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        occlusion_root = osp.join(root, split, 'occlusions')

        if split == 'test':
            self.is_test = True

        total_indice = 0
        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            occlusion_list = sorted(glob(osp.join(occlusion_root, scene, '*.png')))
            flow_list = sorted(glob(osp.join(flow_root, scene, 'frame_*.flo')))
            wp_list = sorted(glob(osp.join(flow_root, scene, 'Wp1_*.flo')))
            wa_list = sorted(glob(osp.join(flow_root, scene, 'Wa1_*.flo')))
            gt_uncert_list = sorted(glob(osp.join(flow_root, scene, 'gt_uncert1_*.npy')))
                
            for i in range(len(image_list)-1):              
                if (total_indice not in SINTEL_VALIDATE_INDICES) & (mode == 'training'):                  
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.flow_list += [ flow_list[i] ]
                    self.wp_list += [ wp_list[i] ]
                    self.wa_list += [ wa_list[i] ]
                    self.gt_uncert_list += [ gt_uncert_list[i] ]
    
                elif (total_indice in SINTEL_VALIDATE_INDICES) & (mode=='validation'): 
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.flow_list += [ flow_list[i] ]
                    self.occlusion_list += [ occlusion_list[i]  ]
                    self.wp_list += [ wp_list[i] ]
                    self.wa_list += [ wa_list[i] ]
                    self.gt_uncert_list += [ gt_uncert_list[i] ]                    
                total_indice += 1
        #print(' image_list ', len(self.image_list) , ' flow list ',len(self.flow_list), ' folki ', len(self.folki_list))
        


 

                    
                
class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/raid/F07773/optical_flow/FlyingThing3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    Wp = sorted(glob(osp.join(fdir, 'Wp_*.flo')) )
                    Wa = sorted(glob(osp.join(fdir, 'Wa_*.flo')) )
                    gt_uncert = sorted(glob(osp.join(fdir, 'gt_uncert_*.npy'))) 
                           
                    
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                            self.wp_list += [ Wp[i] ]
                            self.wa_list += [ Wa[i] ]
                            self.gt_uncert_list += [ gt_uncert[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
                            self.wp_list += [ Wp[i+1] ]
                            self.wa_list += [ Wa[i+1] ]
                            self.gt_uncert_list += [ gt_uncert[i+1] ]   
                            
        #print(' THINGS image_list ', len(self.image_list) , ' flow list ',len(self.flow_list), ' wp_list ', len(self.wp_list),  ' gt_uncert_list ', len(self.gt_uncert_list)) 
''' 
class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/raid/F07773/optical_flow/FlyingThing3D', split='training', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        if split == 'training':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')) )
                        flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                        for i in range(len(flows)-1):
                            if direction == 'into_future':
                                self.image_list += [ [images[i], images[i+1]] ]
                                self.flow_list += [ flows[i] ]
                            elif direction == 'into_past':
                                self.image_list += [ [images[i+1], images[i]] ]
                                self.flow_list += [ flows[i+1] ]

        elif split == 'validation':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TEST/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TEST/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]

                valid_list = np.loadtxt('things_val_test_set.txt', dtype=np.int32)
                self.image_list = [self.image_list[ind] for ind, sel in enumerate(valid_list) if sel]
                self.flow_list = [self.flow_list[ind] for ind, sel in enumerate(valid_list) if sel]
'''

                
class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', mode='training', root='/raid/F07773/optical_flow/KITTI/KITTI2015'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        Wp = sorted(glob(osp.join(root, 'flow_occ/Wp_*.flo')))
        Wa = sorted(glob(osp.join(root, 'flow_occ/Wa_*.flo')))
        gt_uncert = sorted(glob(osp.join(root, 'flow_occ/gt_uncert_*.npy')))     
    
        
        #occlusion_list = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        self.occlusion_list = []

        #for img1, img2, occ in zip(images1, images2, occlusion_list):
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            idx1 = os.path.splitext(os.path.basename(img1))[0][:-3]
            idx2 = os.path.splitext(os.path.basename(img2))[0][:-3]

            # dans le training set, il y a un resplit entre training et validation
            self.extra_info += [ [int(idx1)] ]
            self.image_list += [ [img1, img2] ]
            
            if split == 'training':
                #print('idx ', int(idx1))
                self.flow_list.append( flow_list[int(idx1)] )      
                self.wp_list.append( Wp[int(idx1)] )   
                self.wa_list.append( Wa[int(idx1)] )
                self.gt_uncert_list.append( gt_uncert[int(idx1)] )
                
        #print(' KITTI image_list ', len(self.image_list) , ' flow list ',len(self.flow_list), ' wp_list ', len(self.wp_list),  ' gt_uncert_list ', len(self.gt_uncert_list))                


        
class KITTI_resplit(FlowDataset):
    def __init__(self, aug_params=None, split='training', mode='training', root='/raid/F07773/optical_flow/KITTI/KITTI2015'):
        super(KITTI_resplit, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        #occlusion_list = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        self.occlusion_list = []

        #for img1, img2, occ in zip(images1, images2, occlusion_list):
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            idx1 = os.path.splitext(os.path.basename(img1))[0][:-3]
            idx2 = os.path.splitext(os.path.basename(img2))[0][:-3]

            ### FOLKI flows
            
            I1_temp =  img1.rsplit("/",maxsplit=1) # par ex: '.../116/12007.png' -> I1_temp[0]='.../116' , I1_temp[1]='12007.png'
            folder = I1_temp[0] # '.../116'
            I1_temp1 = I1_temp[1].split(".") # I1_temp1[0] = '12007'
            I1_name = I1_temp1[0]
            
            I2_temp =  img2.rsplit("/",maxsplit=1)
            I2_temp1 = I2_temp[1].split(".") 
            I2_name = I2_temp1[0]
            name_I12 = Path( folder + '/' + I1_name + '-' + I2_name + '.npy')
            #name_I21 = Path( folder + '/' + I2_name + '-' + I1_name + '.npy')                 
            #self.folki_list += [ [name_I12] ]                

            # First time: compute and save Folki flows
            '''
            if (not os.path.exists(name_I12)):
                im1 = imageio.imread(img1).astype(np.float32) / 255.
                im2 = imageio.imread(img2).astype(np.float32) / 255.
                im1_ = rgb2gray(im1)    
                im2_ = rgb2gray(im2)    
                u, v = EFolki(im1_, im2_, iteration=10, radius=[3], rank=3, levels=7)   
                ### Flow 12
                folki_flow = np.stack([u,v]) # [2;W;H]
                im_flow = flow_viz.flow_to_image(folki_flow.transpose(1,2,0))
                np.save(name_I12, folki_flow)
                #print('save ', name_I12)

            else:
                self.folki_list += [ name_I12 ] 
            '''
            
            # dans le training set, il y a un resplit entre training et validation
            if ( int(idx1) in VALIDATE_INDICES_KITTI2015 ) & (mode=='validation'):          
                self.extra_info += [ [int(idx1)] ]
                self.image_list += [ [img1, img2] ]
                self.flow_list.append( flow_list[int(idx1)] )
                
            elif ( int(idx1) not in VALIDATE_INDICES_KITTI2015 ) & (mode == 'training'): # mode = 'training'    
                self.extra_info += [ [int(idx1)] ]
                self.image_list += [ [img1, img2] ]
                self.flow_list.append( flow_list[int(idx1)] )
            

            
            
class KITTIRaw(FlowDataset):
    def __init__(self, aug_params=None, sp_file='/home/F07773/PrevisionPV/optical_flow/ARFlow/datasets/kitti_train_2f_sv.txt'):
        super(KITTIRaw, self).__init__(aug_params, sparse=True)
        self.sp_file = sp_file
        root = '/raid/F07773/optical_flow/KITTI/KITTI_Raw/'
        n_frames = 2
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                sample = [root+sp[i] for i in range(n_frames)]
                self.image_list += [ sample ]

                

          
            
        

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='/raid/F07773/optical_flow/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)
        #print('---  INIT HD1k')

        Wp = sorted(glob(osp.join(root,'hd1k_flow_gt', 'flow_occ/Wp_*.flo')))
        Wa = sorted(glob(osp.join(root,'hd1k_flow_gt', 'flow_occ/Wa_*.flo')))
        gt_uncert = sorted(glob(osp.join(root,'hd1k_flow_gt', 'flow_occ/gt_uncert_*.npy')))
        #print('Wp ' , len(Wp), len(Wa), len(gt_uncert))
        
        for i in range(len(Wp)):
            self.wp_list += [ Wp[i] ]
            self.wa_list += [ Wa[i] ]
            self.gt_uncert_list += [ gt_uncert[i] ]            
     
        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

        #print(' HD1K image_list ', len(self.image_list) , ' flow list ',len(self.flow_list), ' wp_list ', len(self.wp_list),  ' gt_uncert_list ', len(self.gt_uncert_list))
  
  
def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """

    if args.stage == 'autoflow':
        #aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = AutoFlow()
        
    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final   + things + 5*hd1k  + 200*kitti  

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things
            
            
    elif args.stage == 'sintel_resplit':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel_resplit(aug_params, split='training', mode='training', dstype='clean')
        sintel_final = MpiSintel_resplit(aug_params, split='training', mode='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI_resplit({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100   *sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_clean + 100*sintel_final + things
            
            
    elif args.stage == 'sintel_only':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        
        train_dataset = sintel_clean + sintel_final 
            
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
        
    elif args.stage == 'kitti_resplit':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI_resplit(aug_params, split='training')        

    elif args.stage == 'kitti_all':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')       
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        train_dataset = 1*sintel_clean + 1*sintel_final + 2000*kitti  + things

    elif args.stage == 'kitti_semisup':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        kitti_sup = KITTI(aug_params, split='training')
        kitti_raw = KITTIRaw(aug_params)
        train_dataset = 10000 * kitti_sup + kitti_raw
        #train_dataset = kitti_sup

    elif args.stage == 'kitti_semisup_resplit':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        kitti_sup = KITTI_resplit(aug_params, split='training')
        kitti_raw = KITTIRaw(aug_params)
        train_dataset = 400 * kitti_sup + kitti_raw
        #train_dataset = kitti_sup
        
    elif args.stage == 'chairs_sup_kitti_unsup':
        aug_params = {'crop_size': [368,496], 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        chairs = FlyingChairs(aug_params, split='training')   
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        kitti_raw = KITTIRaw(aug_params)
        print('CHAIRS ', len(chairs), ' kitti raw ', len(kitti_raw))
        train_dataset = ConcatDataset([chairs, kitti_raw])
        
        
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

