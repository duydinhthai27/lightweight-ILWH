import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import requests
import tarfile
from tqdm import tqdm
from imbalanceddl.dataset.dataset_base import BaseDataset

class IMBALANCECINIC10(torchvision.datasets.ImageFolder, BaseDataset):
    """Imbalance CINIC-10 Dataset"""
    cls_num = 10
    base_folder = 'cinic-10'
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        
        self.root = root
        self.train = train
        # check download
        if download:    
            if not os.path.exists(self.root):
                os.makedirs(self.root)
            self.download()

            
        split = 'train' if train else 'val'
        self.root = os.path.join(root, self.base_folder, split)
        if not os.path.exists(self.root):
            raise RuntimeError(
                'Dataset not found at {}. '
                'You can use download=True to download it. '
                'If you have already downloaded the dataset manually, '
                'please make sure it is in the correct location.'.format(self.root))

        super(IMBALANCECINIC10, self).__init__(self.root, transform=transform,
                                              target_transform=target_transform)
        np.random.seed(rand_number)
        self.data = np.array(self.samples)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                               imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def download(self):
        if os.path.exists(self.root + '/' + self.base_folder):
            print('Files already downloaded and verified')
            return

        file_path = None
        try:
            print('Downloading CINIC-10 dataset...')
            session = requests.Session()
            session.verify = False
            response = session.get(self.url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            file_path = os.path.join(self.root, self.filename)
            
            # Download with progress bar
            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print('Extracting files...')
            with tarfile.open(file_path, 'r:gz') as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    members_list = tar.getmembers()
                    with tqdm(total=len(members_list), desc='Extracting') as pbar:
                        for member in members_list:
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                            tar.extract(member, path, numeric_owner=numeric_owner)
                            pbar.update(1)

                safe_extract(tar, path=self.root)

            os.remove(file_path)
            print('Dataset downloaded and extracted successfully!')
            
        except Exception as e:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            raise RuntimeError('Error downloading dataset. Please check network connection or download manually.')

    def __getitem__(self, index):
        path, target_n = self.data[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.data)

