from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import cv2
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def sar32bit2RGB(img):
    nimg = np.array(img, dtype=np.float32)
    nimg = nimg / nimg.max() * 255
    nimg_8 = nimg.astype(np.uint8)
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(cv_img)
    return pil_img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []

        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        if train is not None:
            num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train is not None:
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False):
        self.dataset = dataset
        self.transform = transform

        self.pair = pair


    def __len__(self):
        return len(self.dataset)


    def get_image(self, img_path):
        if img_path.endswith('SAR.tif'):
            img = read_image(img_path)
            img = sar32bit2RGB(img)
            img_size = img.size
        else:
            img = read_image(img_path).convert('RGB')
            img_size = img.size
            img_size = [img_size[0] * 0.75, img_size[1] * 0.75]
            # img_size =((img_size[0] / 130 - 0.33238) / 0.01773, (img_size[1] / 678-0.37445) / 0.02000, img_size[1] / img_size[0])
        # img_size = ((img_size[0] - 43.209059) / 299.55908, (img_size[1] - 253.87630662020905) / 9193.529996, img_size[1] / img_size[0])
        # img_size = ((img_size[0] - 43.209059), (img_size[1] - 253.87630662020905),
        #             img_size[1] / img_size[0])
        img_size = ((img_size[0] / 93 - 0.434) / 0.031, (img_size[1] / 427-0.461) / 0.031, img_size[1] / img_size[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, img_size


    def __getitem__(self, index):
        if self.pair:
            imgs = []
            for img in self.dataset[index]:
                img_path, pid, camid = img
                im, img_size = self.get_image(img_path)
                imgs.append((im, pid, camid, img_path.split('/')[-1], img_size))
            return imgs
        else:
            img_path, pid, camid = self.dataset[index]
            img, img_size = self.get_image(img_path)
            return img, pid, camid, img_path.split('/')[-1], img_size
