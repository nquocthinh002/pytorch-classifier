from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor

from PIL import Image
import random
import os
import matplotlib.pyplot as plt

class ClassificationDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.labels = []
        self.image_paths = []
        self.classes = []

        data_path = os.path.join(root, split)
        self.classes = sorted([class_name for class_name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, class_name))])

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_path, class_name)
            for file in os.listdir(class_dir):
                path = os.path.join(class_dir, file)
                if os.path.isfile(path):
                    self.labels.append(label)
                    self.image_paths.append(path)

    def __getitem__(self, index):
        label = self.labels[index]
        image_path = self.image_paths[index]

        with Image.open(image_path) as img:
            image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ROOT_DIR = 'D:/AI/datasets/cifar10'
    SPLIT = 'train'
    IMAGE_SIZE = (32, 32)
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    TRANSFORM = Compose([
        Resize(IMAGE_SIZE),
        ToTensor()
    ])

    train_set = ClassificationDataset(root=ROOT_DIR, split=SPLIT, transform=TRANSFORM)
    # print(train_set.classes)

    idx = random.randint(0, len(train_set)-1)
    print(f'idx:{idx}')
    image, label = train_set[idx]
    # print(image.size)
    # plt.imshow(image)
    print(image.shape)
    plt.imshow(ToPILImage()(image))
    plt.title(f'Class: {train_set.classes[label]}')
    plt.show()

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )

    print(len(train_loader))
    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)