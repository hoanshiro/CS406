import torch
import yaml


def get_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    # save checkpoint
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
