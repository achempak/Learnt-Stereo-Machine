import torch


def unproj_grid(grid_params, img_shape, feats, K, R):
    #K = torch.stack((K, torch.zeros((3, 1))), dim=1)
    KR = torch.mm(K, R)

    feats = feats.permute(1,2,0)
    fh, fw = feats.size()[:2]
    rsz_h = float(fh) / img_shape[0]
    rsz_w = float(fw) / img_shape[1]

    # Create voxel grid
    grid_range = torch.linspace(grid_params[0], grid_params[1], grid_params[2])
    grid = torch.stack(torch.meshgrid([grid_range, grid_range, grid_range]))
    rs_grid = torch.reshape(grid, [3, -1])
    rs_grid = torch.cat((rs_grid, torch.ones((1, grid_params[2]**3))), dim=0)

    # Project grid
    im_p = torch.mm(KR, rs_grid)
    im_x, im_y, im_z = im_p[0, :], im_p[1, :], im_p[2, :]
    im_x = (im_x / im_z) * rsz_w
    im_y = (im_y / im_z) * rsz_h

    # Bilinear interpolation
    im_x = torch.clamp(im_x, 0, fw - 2)
    im_y = torch.clamp(im_y, 0, fh - 2)
    im_x0 = torch.floor(im_x)
    im_x1 = im_x0 + 1
    im_y0 = torch.floor(im_y)
    im_y1 = im_y0 + 1

    wa = (im_x1 - im_x) * (im_y1 - im_y)
    wb = (im_x1 - im_x) * (im_y - im_y0)
    wc = (im_x - im_x0) * (im_y1 - im_y)
    wd = (im_x - im_x0) * (im_y - im_y0)


    img_a = feats[im_x1.long(), im_y1.long()]
    img_b = feats[im_x1.long(), im_y0.long()]
    img_c = feats[im_x0.long(), im_y1.long()]
    img_d = feats[im_x0.long(), im_y0.long()]

    bilinear = wa.unsqueeze(1) * img_a + wb.unsqueeze(1) * img_b + wc.unsqueeze(1) * img_c + wd.unsqueeze(1) * img_d

    return bilinear
