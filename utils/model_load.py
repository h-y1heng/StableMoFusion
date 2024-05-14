import torch
from .ema import ExponentialMovingAverage
def load_model_weights(model, ckpt_path, use_ema=True, device='cuda:0'):
    """
    Load weights of a model from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        ckpt_path (str): Path to the checkpoint file.
        use_ema (bool): Whether to use Exponential Moving Average (EMA) weights if available.
    """
    checkpoint = torch.load(ckpt_path,map_location={'cuda:0': str(device)})
    total_iter = checkpoint.get('total_it', 0)

    if "model_ema" in checkpoint and use_ema:
        ema_key = next(iter(checkpoint["model_ema"]))
        if ('module' in ema_key) or ('n_averaged' in ema_key):
            model = ExponentialMovingAverage(model, decay=1.0)
        model.load_state_dict(checkpoint["model_ema"], strict=True)
        if ('module' in ema_key) or ('n_averaged' in ema_key):
            model = model.module
            print(f'\nLoading EMA module model from {ckpt_path} with {total_iter} iterations')
        else:
            print(f'\nLoading EMA model from {ckpt_path} with {total_iter} iterations')
    else:
        model.load_state_dict(checkpoint['encoder'], strict=True)
        print(f'\nLoading model from {ckpt_path} with {total_iter} iterations')

    return total_iter