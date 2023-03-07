import time
import torch
from tqdm import tqdm
from model import FasterRCNN, RetinaNet
from dataset import get_loader
from utils import get_config, get_device, save_checkpoint, load_checkpoint
from evaluate import calculate_mAP


def train_one_epoch(model, optimizer, train_loader, device, epoch, NUM_EPOCHS):
    train_loss = 0
    loop = tqdm(train_loader)
    loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    start = time.time()
    for idx, (imgs, annotations) in enumerate(loop):
        if idx == 2:
            break
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)

        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loop.set_postfix(train_loss=train_loss.item(), average_train_loss=train_loss.item() / (idx + 1))
    return train_loss / (idx+1)


def validation(model, val_loader, device, epoch, NUM_EPOCHS):
    print('Validation:')
    val_loss = 0
    with torch.no_grad():
        val_loop = tqdm(val_loader)
        val_loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        for idx, (imgs, annotations) in enumerate(val_loop):
            if idx == 2:
                break
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses
            val_loop.set_postfix(val_loss=val_loss.item(), average_val_loss=val_loss.item() / (idx + 1))
    return val_loss / (idx+1)


if __name__ == "__main__":
    # model = FasterRCNN(num_classes=7, pretrained=False)
    model = RetinaNet(num_classes=7, pretrained=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    device = get_device()
    cfg = get_config("cfg.yaml")
    NUM_EPOCHS = cfg["NUM_EPOCHS"]
    train_loader, val_loader, test_loader = get_loader(params=cfg["loader_params"], cfg=cfg)
    best_loss = 9999
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, NUM_EPOCHS)
        val_loss = validation(model, val_loader, device, epoch, NUM_EPOCHS)
        if bess_loss > val_loss:
            save_checkpoint(model, optimizer, filename=f"checkpoint_{epoch}.pth")
        print(f"Epoch {epoch+1} - train_loss: {train_loss}, val_loss: {val_loss}")
        mAP = calculate_mAP(model, test_loader, device)
