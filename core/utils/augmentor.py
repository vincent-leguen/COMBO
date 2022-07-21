import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, wp, wa, gt_uncert, occl):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            gt_uncert = cv2.resize(gt_uncert, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            occl = cv2.resize(occl, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]  
            wp = cv2.resize(wp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            wp = wp * [scale_x, scale_y]  
            wa = cv2.resize(wa, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            wa = wa * [scale_x, scale_y]            

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                occl = occl[:, ::-1]
                gt_uncert = gt_uncert[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                wp = wp[:, ::-1] * [-1.0, 1.0]
                wa = wa[:, ::-1] * [-1.0, 1.0]
 

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                occl = occl[::-1, :]
                gt_uncert = gt_uncert[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                wp = wp[::-1, :] * [1.0, -1.0]
                wa = wa[::-1, :] * [1.0, -1.0]


        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        occl = occl[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        gt_uncert = gt_uncert[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        wp = wp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        wa = wa[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow, wp, wa, gt_uncert, occl

    def __call__(self, img1, img2, flow, wp, wa, gt_uncert, occl=None):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, wp, wa, gt_uncert, occl = self.spatial_transform(img1, img2, flow, wp, wa, gt_uncert, occl)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        wp = np.ascontiguousarray(wp)
        wa = np.ascontiguousarray(wa)
        gt_uncert = np.ascontiguousarray(gt_uncert)
        occl = np.ascontiguousarray(occl)
        return img1, img2, flow, wp, wa, gt_uncert, occl

        
        
        
class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, wp, wa, gt_uncert, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        wp = wp.reshape(-1, 2).astype(np.float32)
        wa = wa.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)
        gt_uncert = gt_uncert.reshape(-1).astype(np.float32)
        
        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]
        wp0 = wp[valid>=1]
        wa0 = wa[valid>=1]
        gt_uncert0 = gt_uncert[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]
        wp1 = wp0 * [fx, fy]
        wa1 = wa0 * [fx, fy]
        gt_uncert1 = gt_uncert0 #* [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]
        wp1 = wp1[v]
        wa1 = wa1[v]
        gt_uncert1 = gt_uncert1[v]
        
        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        wp_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        wa_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)
        gt_uncert_img = np.zeros([ht1, wd1], dtype=np.float32)
        
        flow_img[yy, xx] = flow1
        wp_img[yy, xx] = wp1
        wa_img[yy, xx] = wa1
        valid_img[yy, xx] = 1
        gt_uncert_img[yy, xx] = gt_uncert1
        return flow_img, wp_img, wa_img, gt_uncert_img, valid_img

    def spatial_transform(self, img1, img2, flow, wp, wa, gt_uncert, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, wp, wa, gt_uncert, valid = self.resize_sparse_flow_map(flow,wp, wa, gt_uncert, valid, fx=scale_x, fy=scale_y)
   

        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                wp = wp[:, ::-1] * [-1.0, 1.0]
                wa = wa[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
                gt_uncert = gt_uncert[:, ::-1]

                
        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        wp = wp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        wa = wa[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]    
        gt_uncert = gt_uncert[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]  
       
        return img1, img2, flow, wp, wa, gt_uncert, valid


    def __call__(self, img1, img2, flow,wp, wa, gt_uncert, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, wp, wa, gt_uncert, valid = self.spatial_transform(img1, img2, flow, wp, wa, gt_uncert, valid)
        
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        wp = np.ascontiguousarray(wp)
        wa = np.ascontiguousarray(wa)
        gt_uncert = np.ascontiguousarray(gt_uncert)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, wp, wa, gt_uncert, valid
