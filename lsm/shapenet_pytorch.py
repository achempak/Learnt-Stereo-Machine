from shapenet import ShapeNet
import torch
import numpy as np

class ShapeNetDataset(torch.utils.data.Dataset):

    def __init__(self, im_dir, vox_dir, nviews, nvox, split_file, train=True, categories=None):
        self.data = ShapeNet(
            im_dir=im_dir,
            split_file=split_file,
            rng_seed=42,
            vox_dir=vox_dir)

        class_to_id = {}
        for shape, shape_id in zip(self.data.shape_cls, list(self.data.shape_ids)):
            class_to_id[shape] = shape_id

        if train:
            self.smids = self.data.get_smids('train')
        else:
            self.smids = self.data.get_smids('test')

        if categories is not None:
            cat_smids = []
            class_ids = [class_to_id[c] for c in categories]
            for i in range(len(self.smids)):
                if self.smids[i][0] in class_ids:
                    cat_smids.append(self.smids[i])
            self.smids = cat_smids



        self.nviews = nviews

    def __len__(self):
        return len(self.smids)

    def __getitem__(self, index):
        sid, mid = self.smids[index]
        view_idx = np.random.choice(20, size=(self.nviews, ), replace=False)

        imgs = self.data.get_im(sid, mid, view_idx)
        vol = self.data.get_vol(sid, mid)
        K = self.data.get_K(sid, mid, view_idx)
        R = self.data.get_R(sid, mid, view_idx)

        imgs = np.moveaxis(imgs, -1, 1)
        vol = np.moveaxis(vol, -1, 0)

        return imgs, vol, K, R
