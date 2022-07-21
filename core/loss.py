import torch
import kornia

def gradient(image):
    image_grad = kornia.spatial_gradient(image) # image_grad [batch,3,2,W,H]
    img_dx = image_grad[:,:,0,:,:]
    img_dy = image_grad[:,:,1,:,:]
    return img_dx, img_dy


def smooth_grad_1st(flo, image, alpha=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x * dx2.abs()
    loss_y = weights_y * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.
    