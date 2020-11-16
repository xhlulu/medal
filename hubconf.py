import torch

from models.electra import Electra

def __build_model(model_class, ckpt, pretrained=True, device='cpu', progress=True, check_hash=True):
    net = model_class(device=device)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            ckpt, 
            map_location=torch.device(device), 
            progress=progress, 
            check_hash=check_hash
        )

        for key, value in state_dict.items():
            new_key = key[len('module.'): ] if key.startswith('module.') else key
            if new_key not in net.state_dict():
                print(new_key, 'not expected')
                continue
            try:
                net.state_dict()[new_key].copy_(value)
            except:
                print(new_key, 'not loaded')
                continue

    return net

def electra(pretrained=True, device='cpu', progress=True):
    return __build_model(
        Electra, 
        ckpt="https://github.com/BruceWen120/medal/releases/download/data/electra.pt",
        pretrained=pretrained, device=device, progress=progress)


# @Bruce TODO: write those functions
def lstm(pretrained=True, device='cpu', progress=True, check_hash=True):
    pass

def lstm_sa(pretrained=True, device='cpu', progress=True, check_hash=True):
    pass