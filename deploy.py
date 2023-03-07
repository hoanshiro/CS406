import torch


def convert2onnx(model, cfg, onnx_path):
    dummy_input = torch.randn(1, 3, cfg["IMAGE_HEIGHT"], cfg["IMAGE_WIDTH"])
    # For the Fastseg model, setting do_constant_folding to False is required
    # for PyTorch>1.5.1
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        do_constant_folding=False,
    )

