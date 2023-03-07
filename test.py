import time

import torch

from inference import plot_image_from_output
from torchvision import io
from torchvision.transforms import transforms


def predict_image(model, img_path, device):
    image = io.imread(img_path)
    trs = transforms.ToTensor()
    image = trs(image)
    org_image = image
    my_shape = list(image.shape)
    my_shape.insert(0, 1)
    image = image.reshape(my_shape)
    image = image.to(device)
    with torch.no_grad():
        model.eval()
        start = time.time()
        img_pred = model(image)
        stop = time.time()
        print('Labels: ', img_pred[0]['labels'])
        model.train()
    print(f'Predict time: {stop - start}s \n')
    plot_image_from_output(org_image, img_pred)
