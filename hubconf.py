import torch

from models.electra import Electra

def transfer_state_dict(net, state_dict):
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
    

def electra(pretrained=True, device='cpu', progress=True, check_hash=True):
    net = Electra(device=device)

    if pretrained:
        ckpt = "https://github.com/BruceWen120/medal/releases/download/data/electra.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            ckpt, 
            map_location=torch.device(device), 
            progress=progress, 
            check_hash=check_hash
        )

        transfer_state_dict(net, state_dict)

    return net