import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset # Added import

# This is the Cutout version from imbalanceddl.utils.cutout
# It operates on Tensors.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img): # img is a Tensor
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def imshow(img_tensor, title=None):
    """Helper function to show a tensor as an image"""
    # The Normalize transform needs to be reversed for proper display
    # CIFAR-10 Normalize parameters: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    inv_normalize = transforms.Normalize(
        mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
        std=[1/0.2023, 1/0.1994, 1/0.2010]
    )
    img_tensor = inv_normalize(img_tensor)
    npimg = img_tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')

def main():
    # --- Mock Configuration for ImbalancedDataset ---
    class MockConfig:
        dataset = 'cifar10'
        imb_type = 'exp'      # Example imbalance type
        imb_factor = 1.0      # Using 1.0 for no imbalance to ensure easy access to first class
        rand_number = 0
        # Attributes for ImbalancedDataset's default CIFAR-10 transforms
        # These transforms are on the dataset object, but we'll extract raw PIL for visualization
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        image_size = 32
        padding = 4
        cifar_root = './data'
        # cls_num_list will be set by ImbalancedDataset
        # Add any other attributes if ImbalancedDataset init or methods require them

    config = MockConfig()

    # --- Load dataset using ImbalancedDataset ---
    print("=> Instantiating ImbalancedDataset for CIFAR-10")
    imbalance_data_wrapper = ImbalancedDataset(config, dataset_name=config.dataset)
    
    # Get the underlying training dataset object (e.g., IMBALANCECIFAR10 instance)
    # This object will have default transforms set by ImbalancedDataset
    train_dataset_obj, _ = imbalance_data_wrapper.train_val_sets 
    print(f"=> Successfully loaded: {type(train_dataset_obj)}")

    # --- Get the first original PIL image and label from the dataset object ---
    # Access raw data (numpy array) and convert to PIL to bypass pre-applied transforms
    if hasattr(train_dataset_obj, 'data') and hasattr(train_dataset_obj, 'targets'):
        if len(train_dataset_obj.data) > 0:
            img_data_numpy = train_dataset_obj.data[0]  # Numpy array (H, W, C)
            original_pil_image = Image.fromarray(img_data_numpy)
            label = int(train_dataset_obj.targets[0])
            print(f"=> Extracted first PIL image and label ({label}) from ImbalancedDataset's train_dataset_obj")
        else:
            print("Error: train_dataset_obj.data is empty.")
            return
    else:
        print("Error: train_dataset_obj does not have 'data' or 'targets' attributes.")
        return

    # --- Define normalization and ToTensor (applied at the end of each augmentation pipeline) ---
    to_tensor_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # --- Original Image (just ToTensor and Normalize) ---
    original_tensor_image = to_tensor_normalize(original_pil_image)

    # --- Define Augmentation Pipelines ---
    # Each pipeline starts with transforms that expect PIL images, then ToTensor and Normalize

    # 1. Basic Augmentations: RandomCrop and RandomHorizontalFlip
    basic_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        to_tensor_normalize
    ])
    augmented_basic = basic_augment(original_pil_image)

    # 2. Basic + Cutout
    # Cutout is applied AFTER ToTensor, as it operates on Tensors.
    # This matches the implementation in imbalanceddl.utils.cutout.py
    basic_plus_cutout_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Operates on PIL
        transforms.RandomHorizontalFlip(),          # Operates on PIL
        transforms.ToTensor(),                      # Converts PIL to Tensor
        Cutout(n_holes=1, length=16),             # Operates on Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Operates on Tensor
    ])
    augmented_cutout = basic_plus_cutout_transform(original_pil_image)


    # 3. Basic + AutoAugment (CIFAR-10 Policy)
    basic_plus_autoaugment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10), # Operates on PIL
        to_tensor_normalize
    ])
    augmented_autoaugment = basic_plus_autoaugment(original_pil_image)

    # 4. Basic + RandAugment
    basic_plus_randaugment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(), # Operates on PIL
        to_tensor_normalize
    ])
    augmented_randaugment = basic_plus_randaugment(original_pil_image)
    
    # --- Display the images ---
    images_to_show = [
        (original_tensor_image, "Original"),
        (augmented_basic, "Basic Aug (Crop, Flip)"),
        (augmented_cutout, "Basic + Cutout"),
        (augmented_autoaugment, "Basic + AutoAugment"),
        (augmented_randaugment, "Basic + RandAugment")
    ]

    plt.figure(figsize=(15, 5))
    for i, (img_tensor, title) in enumerate(images_to_show):
        plt.subplot(1, len(images_to_show), i + 1)
        # imshow(img_tensor, title=title)
        imshow(img_tensor)
    
    # CIFAR-10 specific class names for the title
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # plt.suptitle(f"CIFAR-10 Augmentations (Image 0, Class: {cifar10_classes[label]})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # Save or show the plot
    output_path = "/home/hamt/light_weight/imbalanced-DL/example/cifar10_augmentations_visualization.png"
    plt.savefig(output_path)
    print(f"Saved augmentations visualization to {output_path}")
    # plt.show() # Uncomment to display interactively if in a suitable environment

if __name__ == '__main__':
    main()
