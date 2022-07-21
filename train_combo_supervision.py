from __future__ import print_function, division
import sys
sys.path.append('core')

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from combo_model import RAFT
from utils.utils import InputPadder, coords_grid, bilinear_sampler
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter
print('CUDA AVAILABLE ', torch.cuda.is_available())

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return lossP
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 1


def sequence_loss(epoch, flow_preds_p, flow_preds_a, uncert, flow_gt, wp_star, wa_star, gt_uncert, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    # uncert : list 12 of torch.Size([10, 1, 368, 496])
    # gt_uncert: torch.Size([10, 368, 496])
    
    scheduled_sampling = np.maximum(0.01 , 1 - epoch/50)
    
    n_predictions = len(flow_preds_p)    
    final_flow_loss, wp_flow_loss, wa_flow_loss, uncert_loss = 0.0, 0.0, 0.0, 0.0
    
    if len(uncert) == 1: # GT uncertainty
        uncert = [ uncert[0] for i in range(n_predictions)]

    # exclude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
         
        # scheduled_sampling -> 0. At beginning, use GT uncert, then rely more and more on learned uncert
        if random.random() < scheduled_sampling:
            uncertainty_b2wh = gt_uncert.unsqueeze(1)
            uncertainty_b2wh = uncertainty_b2wh.repeat(1,2,1,1)      
        else:
            uncertainty_b2wh = uncert[i].repeat(1,2,1,1)
        

        
        
        # uncertainty loss
        temp = uncert[i]
        i_loss = (temp[:,0,:,:]-gt_uncert).abs()
        uncert_loss += i_weight * (valid[:, None] * i_loss).mean()
        
        # wp flow loss
        i_loss = (flow_preds_p[i]-wp_star).abs()
        wp_flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        
        # wa flow loss
        i_loss = (flow_preds_a[i]-wa_star).abs()
        wa_flow_loss += i_weight * (valid[:, None] * i_loss).mean()       
        
        # Final flow loss
        flow_pred = (1-uncertainty_b2wh) * flow_preds_p[i] + uncertainty_b2wh * flow_preds_a[i]
        i_loss = (flow_pred - flow_gt).abs()
        final_flow_loss += i_weight * (valid[:, None] * i_loss).mean()
 
    flow_pred = (1-uncert[i]) * flow_preds_p[-1] + uncert[i] * flow_preds_a[-1]  
    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return final_flow_loss, wp_flow_loss, wa_flow_loss, uncert_loss, metrics




def photometric_sequence_loss(flow_preds_p, image1, image2, uncert, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    # flow_preds[-1], GT : [batch, 2, W, H]
    n_predictions = len(flow_preds_p)    
    photometric_loss = 0.0
    criterion = nn.L1Loss()
    
    if len(uncert) == 1: # GT uncertainty
        uncert = [ uncert[0] for i in range(n_predictions)]
    
    for i in range(n_predictions):
        coords2 = coords_grid(image2.shape[0], image2.shape[2], image2.shape[3]).cuda()
        coords21 = coords2 + flow_preds_p[i]
        im21 = bilinear_sampler(image2, coords21.permute(0,2,3,1))
        i_weight = gamma**(n_predictions - i - 1)     
        uncertainty_b3wh = uncert[i].repeat(1,3,1,1)
        photometric_loss += i_weight * criterion( (1-uncert[i])*im21, (1-uncert[i])*image1) 
    return photometric_loss 
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10,factor=0.5,verbose=True)
    return optimizer, scheduler
    
    
class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.T0 = time.time()

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str, ' time = ', time.time()-self.T0)
        self.T0 = time.time()

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, step, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], step)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    epoch = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    
    should_keep_training = True
    while should_keep_training:
        t0_epoch = time.time()
        ep_loss_total, ep_loss_final, ep_loss_wp, ep_loss_wa, ep_loss_uncert, ep_loss_photo, ep_loss_wpa = 0,0,0,0,0,0,0   
        for i_batch, data_blob in enumerate(train_loader):
            t0 = time.time() 
            optimizer.zero_grad()
            image1, image2, flow, wp_star, wa_star, gt_uncert, valid ,_ = [x.cuda() for x in data_blob]
            #print('image1 ', image1.shape, ' flow ',flow.shape, wp_star.shape, wa_star.shape, gt_uncert.shape)

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                
            ## MODEL PREDICTION
            flow_predictions_p, flow_predictions_a, uncertainty_masks = model(image1, image2, iters=args.iters)        
            #print('flow_predictions_p ', len(flow_predictions_p), flow_predictions_p[0].shape, ' uncert ', len(uncertainty_masks), uncertainty_masks[0].shape)  
                
            ## SUPERVISED LOSS                      
            final_flow_loss, wp_flow_loss, wa_flow_loss, uncert_loss, metrics = sequence_loss(epoch, flow_predictions_p, flow_predictions_a, uncertainty_masks, flow, wp_star, wa_star, gt_uncert, valid, args.gamma)
   
            ## NORM Wpa
            norm_wpa = 0
            for k in range(len(flow_predictions_a)):
                norm_wpa += torch.norm(flow_predictions_a[k]) / len(flow_predictions_a)
                norm_wpa += torch.norm(flow_predictions_p[k]) / len(flow_predictions_p)
                
   
            ## PHOTOMETRIC LOSS
            photometric_loss = photometric_sequence_loss(flow_predictions_p, image1, image2, uncertainty_masks, gamma=0.8)
            
            loss = args.lambda_final * final_flow_loss + args.lambda_wp * wp_flow_loss + args.lambda_wa * wa_flow_loss + args.lambda_uncert * uncert_loss  + args.lambda_photo * photometric_loss + args.lambda_wpa * norm_wpa
           
    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            ep_loss_total += loss.item()
            ep_loss_final += final_flow_loss.item()
            ep_loss_wp += wp_flow_loss.item()
            ep_loss_wa += wa_flow_loss.item()
            ep_loss_uncert += uncert_loss.item()
            ep_loss_photo += photometric_loss.item()
            ep_loss_wpa += norm_wpa.item()
            
            scaler.step(optimizer)         
            scaler.update()


            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                #PATH = 'checkpoints/%s.pth' % (args.name)
                PATH = '/raid/F07773/optical_flow/checkpoints/%s_%d.pth' % (args.name, total_steps+1)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        res = evaluate_gt.validate_chairs(model.module)                               
                        scheduler.step(res['chairs'])
                        results.update(res)
                        
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'sintel_resplit':
                        results.update(evaluate.validate_sintel_resplit(model.module))
                    elif val_dataset == 'kitti_resplit':
                        results.update(evaluate.validate_kitti_resplit(model.module))
                        
                loss_analysis = {'wp_loss':ep_loss_wp/i_batch, 'wa_loss':ep_loss_wa/i_batch, 'photo':ep_loss_photo/i_batch, 'loss_alpha':ep_loss_uncert/i_batch, 'wpa':ep_loss_wpa/i_batch}     
                results.update(loss_analysis)        
                logger.write_dict(total_steps+1, results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
                
            #print('ibatch ', i_batch , '/', len(train_loader), ' time ',time.time()-t0)        
 
            if (total_steps % 500 == 0) & (total_steps > 1):
                print('epoch ', epoch, i_batch,'/', len(train_loader), 'step ', total_steps, ' time ',time.time()-t0, ' loss ',ep_loss_total/i_batch,' final ',ep_loss_final/i_batch,' wp ',ep_loss_wp/i_batch, 'wa ',ep_loss_wa/i_batch, ' photo ', ep_loss_photo/i_batch,' loss uncert ',ep_loss_uncert/i_batch, ' wpa ',ep_loss_wpa/i_batch,  'certainty ', torch.max(uncertainty_masks[-1]).item(),torch.mean(uncertainty_masks[-1]).item(), 'gt uncert ',torch.max(gt_uncert),torch.mean(gt_uncert)) 
      
        #print('EPOCH ', epoch, ' time = ',time.time()-t0_epoch)
        epoch = epoch + 1

          
  
    logger.close()
    PATH = '/raid/F07773/optical_flow/checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=250000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    
    parser.add_argument('--lambda_final', default=1, type=float, help='') 
    parser.add_argument('--lambda_wp', default=1, type=float, help='')
    parser.add_argument('--lambda_wa', default=1, type=float, help='')
    parser.add_argument('--lambda_uncert', default=0.1, type=float, help='')
    parser.add_argument('--lambda_photo', default=0.1, type=float, help='')
    parser.add_argument('--lambda_wpa', default=0.01, type=float, help='')
    
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)