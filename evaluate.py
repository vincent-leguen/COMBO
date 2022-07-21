import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

#from raft import RAFT
#from raft_dualbranch_wa_wo_uncert import RAFT
from raft_dualbranch_wa_learned_mask import RAFT


#from raft_dualbranch_wa_learned_mask_cert import RAFT

from utils.utils import InputPadder, forward_interpolate
from uncertainty import compute_GT_uncertainty

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=True, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        #print('output_path ',output_path, ' frame_id ',str(frame_id))
        output_filename = os.path.join(output_path, str(frame_id).zfill(6)+'_10.png')
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    epe_p_list, epe_a_list, epe_p_mask_list, epe_a_mask_list = [],[],[],[]
  
    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt,_,_,gt_uncert,_,_ = val_dataset[val_id]    
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        flow_up_p, flow_up_a, flow_pred, uncertainty_mask = model(image1, image2, iters=24, test_mode=True)

        gt_uncert = gt_uncert[np.newaxis,:,:]
        gt_uncert = np.repeat(gt_uncert, 2, axis=0)       

        
        # erreur calculee avec la GT uncertainty
        #flow_pred_gtuncert = (1-gt_uncert) * flow_up_p[0].cpu() + gt_uncert * flow_up_a[0].cpu()
        #epe = torch.sum((flow_pred_gtuncert - flow_gt.cpu())**2, dim=0).sqrt()
        
        epe = torch.sum((flow_pred[0].cpu() - flow_gt)**2, dim=0).sqrt()
        
        
        epe_p = torch.sum((flow_up_p[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_a = torch.sum((flow_up_a[0].cpu() - flow_gt)**2, dim=0).sqrt()
        
        uncert = uncertainty_mask[-1].cpu().repeat(2,1,1)

        gt_uncert = gt_uncert > 0.1 # seuillage dur
        
        #print('flow_up_p[0] ', flow_up_p[0].shape, ' flow_gt ',flow_gt.shape, ' uncert ',gt_uncert.shape, (1-uncert).shape)
        epe_p_mask = torch.sum((flow_up_p[0].cpu() - flow_gt)**2 * (1-gt_uncert) , dim=0).sqrt()
        epe_a_mask = torch.sum((flow_up_a[0].cpu() - flow_gt)**2 * (gt_uncert) , dim=0).sqrt()
        

        epe_p_mask = epe_p_mask[~torch.any(epe_p_mask.isnan(),dim=1)]
        epe_a_mask = epe_a_mask[~torch.any(epe_a_mask.isnan(),dim=1)]
        #print('epe_p_mask ',epe_p_mask, ' epe_a_mask ', epe_a_mask)
        
        #epe_p_mask = epe_p_mask[epe_p_mask >0]
        #epe_a_mask = epe_a_mask[epe_a_mask > 0]        
        
        epe_list.append(epe.view(-1).numpy())
        epe_p_list.append(epe_p.view(-1).numpy())
        epe_a_list.append(epe_a.view(-1).numpy())
        epe_p_mask_list.append(epe_p_mask.view(-1).numpy())
        epe_a_mask_list.append(epe_a_mask.view(-1).numpy())
        
    epe = np.mean(np.concatenate(epe_list))
    epe_p = np.mean(np.concatenate(epe_p_list))
    epe_a = np.mean(np.concatenate(epe_a_list))
    epe_p_mask = np.mean(np.concatenate(epe_p_mask_list))
    epe_a_mask = np.mean(np.concatenate(epe_a_mask_list))    
    print("Validation Chairs TEST EPE: %f  wp: %f  wa:%f  wp_mask:%f  wa_mask:%f " % (epe, epe_p, epe_a, epe_p_mask, epe_a_mask))
    return {'chairs': epe}

    
## Special RAFT
'''
@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    epe_p_mask_list, epe_a_mask_list = [],[]
  
    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt,_,_,gt_uncert,_,_ = val_dataset[val_id]    
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        gt_uncert = gt_uncert[np.newaxis,:,:]
        gt_uncert = np.repeat(gt_uncert, 2, axis=0) 
        #gt_uncert = gt_uncert > 0.1 # seuillage dur
 
        _, flow_pr = model(image1, image2, iters=24, test_mode=True)
        #_, flow_pr = model(image1, image2, iters=24, test_mode=True, uncert=uncertainty)
        #_, flow_pr = model(image1, image2, iters=24, test_mode=True, uncert=None)
        #print(' len flow_pr ', len(flow_pr))
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_p_mask = torch.sum((flow_pr[0].cpu() - flow_gt)**2 * (1-gt_uncert) , dim=0).sqrt()
        epe_a_mask = torch.sum((flow_pr[0].cpu() - flow_gt)**2 * (gt_uncert) , dim=0).sqrt()
        
        #epe_p_mask = epe_p_mask[~torch.any(epe_p_mask.isnan(),dim=1)]
        #epe_a_mask = epe_a_mask[~torch.any(epe_a_mask.isnan(),dim=1)]
        #print('epe_p_mask ',epe_p_mask, ' epe_a_mask ', epe_a_mask)
        
        
        #epe_p_mask = epe_p_mask[epe_p_mask >0]
        #epe_a_mask = epe_a_mask[epe_a_mask > 0]
        
        epe_list.append(epe.view(-1).numpy())
        epe_p_mask_list.append(epe_p_mask.view(-1).numpy())
        epe_a_mask_list.append(epe_a_mask.view(-1).numpy())
        
    epe = np.mean(np.concatenate(epe_list))
    epe_p_mask = np.mean(np.concatenate(epe_p_mask_list))
    epe_a_mask = np.mean(np.concatenate(epe_a_mask_list))    
    print("Validation Chairs TEST EPE: %f wp_mask:%f  wa_mask:%f " % (epe,  epe_p_mask, epe_a_mask))
    return epe
'''

    
@torch.no_grad()
def validate_things(model, iters=24):
    model.eval()
    epe_list = []
    epe_mask_list = []
  
    val_dataset = datasets.FlyingThings3D(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt,folki_12, _,_ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        uncertainty, mask = compute_GT_uncertainty(image1, image2, folki_12)
        uncertainty = uncertainty.unsqueeze(1)
        uncertainty.repeat(1,2,1,1) 
        
        print(' THINGS image 1 ', image1.shape, flow_gt.shape)
        _, flow_pr = model(image1, image2, iters=24, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_mask = torch.sum((flow_pr[0].cpu() * mask.cpu() - flow_gt * mask.cpu())**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        epe_mask_list.append(epe_mask.view(-1).numpy())
        
    epe = np.mean(np.concatenate(epe_list))
    epe_mask = np.mean(np.concatenate(epe_mask_list))
    print("Validation Chairs TEST EPE: %f, EPE mask: %f " % (epe,epe_mask))
    return {'chairs': epe}

    
    
    
    
@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            #if (val_id % 10) ==0 :
            #    print(val_id, '/', len(val_dataset))        
            image1, image2, flow_gt,_, _,_,_,_ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_up_p, flow_up_a, flow_pred, uncertainty_mask = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pred[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation Sintel (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)
    return results

    
@torch.no_grad()
def validate_sintel_resplit(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel_resplit(split='training', mode='validation', dstype=dstype)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []        

        for val_id in range(len(val_dataset)):
            #if (val_id % 10) ==0 :
            #    print(val_id, '/', len(val_dataset))        
            image1, image2, flow_gt,_, _,_,_,_ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_up_p, flow_up_a, flow_pred, uncertainty_mask = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pred[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)      

        
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        

        
        print("Validation Sintel resplit on %d images (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (len(val_dataset), dstype,epe, px1, px3, px5))
        results[str(dstype+'_resplit')] = np.mean(epe_list)
    return results


    
@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training',mode='validation')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, _,_,valid_gt,_ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_up_p, flow_up_a, flow_pred, uncertainty_mask = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pred[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.contiguous().view(-1)
        mag = mag.contiguous().view(-1)
        val = valid_gt.contiguous().view(-1) >= 0.5
        #print('flow ', flow.shape, ' flow gt ',flow_gt.shape,'epe ', epe.shape, ' valid_gt ', valid_gt.shape)
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        #print(val_id, 'EPE = ', epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_kitti_resplit(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI_resplit(split='training',mode='validation')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, _,_,valid_gt,_ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_up_p, flow_up_a, flow_pred, uncertainty_mask = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pred[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.contiguous().view(-1)
        mag = mag.contiguous().view(-1)
        val = valid_gt.contiguous().view(-1) >= 0.5
        #print('flow ', flow.shape, ' flow gt ',flow_gt.shape,'epe ', epe.shape, ' valid_gt ', valid_gt.shape)
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        #print(val_id, 'EPE = ', epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI resplit: %f, %f" % (epe, f1))
    return {'kitti_resplit_epe': epe, 'kitti_resplit_f1': f1}
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    print('begin sintel submission')
    create_sintel_submission(model.module, warm_start=True)
    print('end sintel submission')
    #print('begin kitti submission')
    #create_kitti_submission(model.module)
    #print('end kitti submission')
    
    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)
            
        elif args.dataset == 'things':
            validate_things(model.module)
            
        elif args.dataset == 'sintel':
            validate_sintel(model.module)
            
        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'sintel_resplit':
            validate_sintel_resplit(model.module)
            
        elif args.dataset == 'kitti_resplit':
            validate_kitti_resplit(model.module)            
            
