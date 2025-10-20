import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import requests
import zipfile
from tqdm import tqdm
from imbalanceddl.dataset.dataset_base import BaseDataset


class IMBALANCETINY(torchvision.datasets.ImageFolder, BaseDataset):
    """Imbalanced Tiny-Imagenet Dataset

    Code for creating Imbalanced Tiny-Imagenet dataset
    """
    cls_num = 200
    base_folder = 'tiny-imagenet-200'
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        
        print(f"[DEBUG] __init__ called with root={root}, train={train}, download={download}")
        self.root = root
        self.train = train
        
        # check download
        if download:
            print(f"[DEBUG] Download requested. Checking directory: {self.root}")
            if not os.path.exists(self.root):
                print(f"[DEBUG] Creating directory: {self.root}")
                os.makedirs(self.root)
            self.download()
            
        # In imbalance_dataset.py, root is already the full path to train or val/images
        if not os.path.exists(self.root):
            print(f"[ERROR] Dataset not found at {self.root}")
            raise RuntimeError(
                'Dataset not found at {}. '
                'You can use download=True to download it. '
                'If you have already downloaded the dataset manually, '
                'please make sure it is in the correct location.'.format(self.root))

        print(f"[DEBUG] Calling super().__init__ with root_path={self.root}")
        super(IMBALANCETINY, self).__init__(self.root, transform=transform,
                                           target_transform=target_transform)
        print(f"[DEBUG] Samples loaded: {len(self.samples)}")
        print("=> Generating Imbalanced Tiny-ImageNet with Type: {} | Ratio: {}".
              format(imb_type, imb_factor))
        np.random.seed(rand_number)
        self.data = np.array(self.samples)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                               imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def download(self):
        # We need to determine the base directory to download the data
        # In imbalance_dataset.py, root is the full path to train or val/images
        # We need to go up levels to find the base directory
        base_dir = self.root
        # If we're in 'train' or 'val/images', go up to the parent directory
        if os.path.basename(self.root) == 'train' or os.path.basename(self.root) == 'images':
            base_dir = os.path.dirname(self.root)
            if os.path.basename(base_dir) == 'val':
                base_dir = os.path.dirname(base_dir)
        
        print(f"[DEBUG] download() called. Base directory determined as: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"[DEBUG] Creating base directory: {base_dir}")
            os.makedirs(base_dir)

        file_path = os.path.join(base_dir, self.filename)
        print(f'[DEBUG] Downloading {self.filename}...')
        try:
            session = requests.Session()
            session.verify = False
            response = session.get(self.url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            print(f'[DEBUG] Saving to {file_path}...')
            with open(file_path, 'wb') as f, tqdm(
                desc=f"Downloading {self.filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f'[DEBUG] Extracting to {base_dir}...')
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Get list of files for progress tracking
                file_list = zip_ref.infolist()
                
                # Safety check and extract with progress bar
                with tqdm(desc="Extracting files", total=len(file_list), unit='file') as pbar:
                    for zipinfo in file_list:
                        if zipinfo.filename.startswith('/') or '..' in zipinfo.filename:
                            print(f"[WARNING] Skipping potentially unsafe path: {zipinfo.filename}")
                            pbar.update(1)
                            continue
                        zip_ref.extract(zipinfo, base_dir)
                        pbar.update(1)

            # Make sure the validation data is in the right format
            # Tiny-ImageNet validation images are in a different structure
            val_dir = os.path.join(base_dir, self.base_folder, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            if os.path.exists(val_dir) and not os.path.exists(os.path.join(val_images_dir, 'n01443537')):
                print('[DEBUG] Restructuring validation directory...')
                # Read validation annotations
                val_anno_path = os.path.join(val_dir, 'val_annotations.txt')
                if os.path.exists(val_anno_path):
                    with open(val_anno_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                img_name, class_dir = parts[0], parts[1]
                                # Create class directory if it doesn't exist
                                class_path = os.path.join(val_images_dir, class_dir)
                                if not os.path.exists(class_path):
                                    os.makedirs(class_path)
                                # Move image to class directory
                                img_src = os.path.join(val_images_dir, img_name)
                                img_dst = os.path.join(class_path, img_name)
                                if os.path.exists(img_src) and not os.path.exists(img_dst):
                                    try:
                                        os.rename(img_src, img_dst)
                                    except Exception as e:
                                        print(f"[ERROR] Could not move {img_name} to {class_dir}: {e}")

            print('[DEBUG] Extraction done!')
        except Exception as e:
            print(f'[ERROR] Error downloading/extracting dataset: {e}')
            if os.path.exists(file_path):
                os.remove(file_path)
            raise RuntimeError('Error downloading dataset. Please check network connection or download manually.')
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of
            the target class.
        """
        path, target_n = self.data[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """
        Original Code is self.samples, we change to self.data
        Thus override this method
        """
        return len(self.data)
        
        



