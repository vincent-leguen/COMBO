import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock, UpdateBlock_withMask
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4            
        '''
        if args.small:
            self.hidden_dim = hdim = 48
            self.context_dim = cdim = 32
            args.corr_levels = 4
            args.corr_radius = 3
        '''    


        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
            self.update_block_a = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
            self.update_block_a = UpdateBlock_withMask(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow=None,flow_init=None, upsample=True, test_mode=False, uncert=None):
        """ Estimate optical flow between pair of frames """
        
        # image1, image2: [batch_size/2, 3, H, W]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
    
        #vol_corr = corr_fn.corr(fmap1, fmap2)
        #print('image1 ',image1.shape, ' f1map ',fmap1.shape , ' vol_corr ',vol_corr.shape)
        
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net_p, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net_p = torch.tanh(net_p)
            inp = torch.relu(inp)
            net_a = net_p.clone()

        coords0_p, coords1_p = self.initialize_flow(image1)
        coords0_a, coords1_a = self.initialize_flow(image1)
        
        #if flow_init is not None:
        #    #print('folki init ', flow_init)
        #    flow_init = F.interpolate(flow_init, scale_factor=1/8)
        #    coords1 = coords1 + flow_init

        flow_predictions_p, flow_predictions_a = [], []
        uncertainty_masks = []
        for itr in range(iters):
            coords1_p = coords1_p.detach()
            corr_p = corr_fn(coords1_p) # index correlation volume
            coords1_a = coords1_a.detach()
            corr_a = corr_fn(coords1_a) # index correlation volume           
           
            flow_p = coords1_p - coords0_p
            flow_a = coords1_a - coords0_a
            with autocast(enabled=self.args.mixed_precision):
                net_p, up_mask_p, delta_flow_p = self.update_block(net_p, inp, corr_p, flow_p)
                net_a, up_mask_a, delta_flow_a, uncertainty_mask = self.update_block_a(net_a, inp, corr_a, flow_a)

            # F(t+1) = F(t) + \Delta(t)
            coords1_p = coords1_p + delta_flow_p
            coords1_a = coords1_a + delta_flow_a

            # upsample predictions
            if up_mask_p is None:
                flow_up_p = upflow8(coords1_p - coords0_p)
                flow_up_a = upflow8(coords1_a - coords0_a)
            else:
                flow_up_p = self.upsample_flow(coords1_p - coords0_p, up_mask_p)
                flow_up_a = self.upsample_flow(coords1_a - coords0_a, up_mask_a)
                
            flow_predictions_p.append(flow_up_p)
            flow_predictions_a.append(flow_up_a)
            
            
            uncertainty_mask = upflow8(uncertainty_mask)
            #uncertainty_mask = (uncertainty_mask-torch.min(uncertainty_mask)) / (torch.max(uncertainty_mask)-torch.min(uncertainty_mask))
            #uncertainty_mask = torch.sigmoid( uncertainty_mask -0.5 )
            uncertainty_masks.append(uncertainty_mask)
            
            #uncertainty_mask =  upflow8(uncertainty_mask) 
            #uncertainty_masks.append(uncertainty_mask)
            
            
        if test_mode:
            uncertainty_mask_b2wh = uncertainty_mask.repeat(1,2,1,1)
            #print('uncertainty_mask_b2wh ',uncertainty_mask_b2wh.shape, torch.max(uncertainty_mask_b2wh).item(), torch.mean(uncertainty_mask_b2wh).item())
            flow_pred = (1-uncertainty_mask_b2wh)*flow_up_p + uncertainty_mask_b2wh*flow_up_a
            #return flow_up_p, flow_up_a, flow_pred, uncertainty_mask
            
            
            uncertainty_mask_b2wh_down = F.interpolate(uncertainty_mask_b2wh, scale_factor=1/8)
            #print('uncertainty_mask_b2wh_down ',uncertainty_mask_b2wh_down.shape, ' coords1_p ', coords1_p.shape, coords0_p.shape)
            coords_diff = (1-uncertainty_mask_b2wh_down)*(coords1_p-coords0_p) + uncertainty_mask_b2wh_down*(coords1_a-coords0_a)
            return coords_diff, flow_pred
            
            
            
        return flow_predictions_p, flow_predictions_a, uncertainty_masks
