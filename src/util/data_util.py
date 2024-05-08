import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import json


def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


# =========== ModelNet40 =================
class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition  # Here the new given partition will cover the 'train'

    def __getitem__(self, item):  # indice of the pts or label
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        #if self.partition == 'train':
            # pointcloud = pc_normalize(pointcloud)  # you can try to add it or not to train our model
            #pointcloud = translate_pointcloud(pointcloud)
            #np.random.shuffle(pointcloud)  # shuffle the order of pts
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# =========== ShapeNet Part =================
class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False):
        self.npoints = npoints
        self.root ='C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset'
        self.catfile = os.path.join(self.root, 'formerclasses.txt')
        self.cat = {}
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
    

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'train.json'), 'r') as f:
            train_ids = set([str(d.split('\\')[-1]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'validation.json'), 'r') as f:
            val_ids = set([str(d.split('\\')[-1]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'test.json'), 'r') as f:
            test_ids = set([str(d.split('\\')[-1]) for d in json.load(f)])

      
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            
          
            #print(fns,'fns')
            #print(train_ids,'ids')
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn in train_ids) or (fn in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            #print(test_ids)
            
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                #print(token,'token')
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        #print(self.cat,'cat')
        #print(self.meta,'meta')

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
           
        #print(self.datapath,'datapath')
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'former': [0,1,2,3,4,5]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        pass
        #print(data.shape)
        #print(label.shape)
