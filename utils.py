
import torch
from models.resnet import ResNet, ResNet101
from models.vgg import VGG16
import torchvision.transforms as transforms


IMAGE_PREPROCESSING = transfor_img = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_model(path: str, device: str):
    if path.find("vgg"):
        model = VGG16()
    else:
        model = ResNet101()
    checkpoint = torch.load(path, map_location=torch.device(device))
    model = ResNet101()
    model.state_dict(checkpoint['net'])
    model.eval()
    return model, checkpoint['epoch'], checkpoint['acc'], checkpoint['loss']  # TODO: Add keyparam 'loss'
