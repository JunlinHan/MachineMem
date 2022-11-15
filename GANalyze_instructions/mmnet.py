import torchvision
import torch
import math
from timm.models import create_model
# Assessor for MachineMem/HumanMem predictor.
class MmNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = create_model('resnet50',num_classes=1)

    def forward(self, x):
        return self.model(x)

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @property
    def input_size(self):
        return [3, 224, 224]

def mmnet(tencrop):
    model = MmNet()
    checkpoint = torch.load("./assessors/machinemem_predictor.tar") #Change this to swap between MachineMem and HumanMem.
    newmodel = {}
    a = checkpoint["state_dict"]
    for k, v in a.items():
        toadd = "model."
        k = ''.join([toadd, k])
        newmodel[k] = v
    model.load_state_dict(newmodel)


    if tencrop:
        input_transform = tencrop_image_transform(model)
        output_transform = tencrop_output_transform_emonet
    else:
        input_transform = image_transform(model)
        output_transform = lambda x: x

    return model, input_transform, output_transform

def tencrop_image_transform(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    normalize = torchvision.transforms.Normalize(mean=model.mean, std=model.std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: tencrop(image.permute(0, 2, 3, 1), cropped_size=224)),
        torchvision.transforms.Lambda(lambda image: torch.stack([torch.stack([normalize(x / 255) for x in crop])
                                                                 for crop in image])),
    ])

def tencrop_output_transform_emonet(output):
    output = output.view(-1, 10).mean(1)
    return output

def image_transform(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    normalize = torchvision.transforms.Normalize(mean=model.mean, std=model.std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
        torchvision.transforms.Lambda(lambda image: torch.stack([normalize(x / 255) for x in image])),
    ])


def tencrop(images, cropped_size=227):
    im_size = 256  # hard coded

    crops = torch.zeros(images.shape[0], 10, 3, cropped_size, cropped_size)
    indices = [0, im_size - cropped_size]  # image size - crop size

    for img_index in range(images.shape[0]):  # looping over the batch dimension
        img = images[img_index, :, :, :]
        curr = 0
        for i in indices:
            for j in indices:
                temp_img = img[i:i + cropped_size, j:j + cropped_size, :]
                crops[img_index, curr, :, :, :] = temp_img.permute(2, 0, 1)
                crops[img_index, curr + 5, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
                curr = curr + 1
        center = int(math.floor(indices[1] / 2) + 1)
        crops[img_index, 4, :, :, :] = img[center:center + cropped_size,
                                           center:center + cropped_size, :].permute(2, 0, 1)
        crops[img_index, 9, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
    return crops
