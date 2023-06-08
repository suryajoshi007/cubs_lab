import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # get data image list of paths
        self.data = self._get_list_structure(data_path)

    def _get_list_structure(self, data_path):
        # get all jpg image paths from the dataset directory
        jpg_paths = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.jpg'):
                    jpg_paths.append(os.path.join(root, file))
        return jpg_paths

    def __getitem__(self, index):
        # read the image at index in self.data list.
        # apply required transformations and return the result

        image_path = self.data[index]
        image = Image.open(image_path)
        
        # TODO apply required transformation
        transform = transforms.ToTensor()
        tensor_image = transform(image)

        return tensor_image

    def __len__(self):
        return len(self.data)


# export DATASET_PATH="/Users/suryakanthjoshi/cubs_lab/dataset"
data_path: str = os.environ.get('DATASET_PATH')
dataset = CustomDataset(data_path=data_path)

batch_size = 32
shuffle = True
num_workers = 1

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
)

if __name__ == "__main__":
    for batch in data_loader:
        print(batch)
        break
