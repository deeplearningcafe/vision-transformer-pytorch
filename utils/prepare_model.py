import torch.nn.init as init
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
import torch
import math
import omegaconf
from vit import ViT

# as in the paper the initialization is not mentioned, we will use the same as a normal transformer
def apply_weights_init(model, conf: omegaconf.DictConfig):
    std = math.sqrt(2/(5*conf.vit.hidden_dim))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'gamma' in name.split('.') or 'beta' in name.split('.'):
                continue
            if 'bias' in name.split('.'):
                nn.init.constant_(param.data, 0.0)
            elif 'output_projection' in name.split('.'):
                init.normal_(param.data, mean=0, std=std/math.sqrt(2*conf.vit.num_layers))
            else:
                init.normal_(param.data, mean=0, std=std)

def create_scheduler(optim: torch.optim.Optimizer, conf: omegaconf.DictConfig):
    scheduler_type = conf.train.scheduler_type
    max_epochs = conf.train.max_epochs
    warmup_epochs = conf.train.warmup_epochs
    
    if scheduler_type == "warmup-cosine":
        scheduler_1 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.005, end_factor=1.0, total_iters=warmup_epochs)
        scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs-warmup_epochs, eta_min=conf.train.lr*5e-2)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [scheduler_1, scheduler_2], milestones=[conf.train.warmup_epochs])
    
    elif scheduler_type == "wsd":
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.005, end_factor=1.0, total_iters=conf.train.warmup_epochs)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(conf.train.max_epochs*0.9), eta_min=conf.train.lr*5e-2)
        # we end the cosine at lr*1e-3 so we start the dacay by a factor of that lr, if we max is 2e-3 then we start the decay at 2e-6
        scheduler_decay = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-3, end_factor=2e-4, total_iters=int(conf.train.max_epochs*0.1))
        scheduler =  torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[scheduler_warmup, scheduler_cosine, scheduler_decay], 
                                                           milestones=[conf.train.warmup_epochs, int(conf.train.max_epochs*0.9)])
    else:
        raise "Not implemented optimizer"
        
    return scheduler

def create_optim(model: torch.nn.Module, conf:omegaconf.DictConfig):
    if conf.train.use_bitsandbytes:
        try:
            import bitsandbytes
        except:
            raise ImportError
        
        # the original paper uses sgd, but we can try to use adamw
        if conf.train.optim == 'sgd':
            optim = bitsandbytes.optim.SGD8bit(model.parameters(), lr=conf.train.lr, momentum=0.99)
        else:
            optim = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=conf.train.lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        if conf.train.optim == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=conf.train.lr, momentum=0.99)
        else:
            optim = torch.optim.AdamW(model.parameters(), lr=conf.train.lr, betas=(0.9, 0.999), weight_decay=0.1)
    
    # scheduler
    scheduler = create_scheduler(optim, conf)
    
    return optim, scheduler

def get_update_ratio(model, layers:dict[str, torch.tensor], diffs: dict[str, list]):
    # we have the layers dict that stores the weights and the diffs dict that stores a list with the diff in each step
    # dict are not in place so we need to create a new one
    tmp_dict = {}
    
    for key, layer in layers.items():
        for name, param in model.named_parameters():
            if key in name.split('.'):
                with torch.no_grad():
                    param_clone = param.detach().cpu().clone()
                    change = param_clone-layer
                    diffs[key].append((change.std()/param.std()).log10().item())
                    tmp_dict[key] = param_clone

    return tmp_dict, diffs

def freeze_model(model: torch.nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False 
    
    # now only the last block
    for name, param in model.blocks[-1].named_parameters():
        print(name)
        param.requires_grad = True
    
    # now only the head
    for name, param in model.mlp_head.named_parameters():
        print(name)
        param.requires_grad = True
    
    return model
    
def prepare_training(conf: omegaconf.DictConfig):
    model = ViT(conf)
    model.train()
    apply_weights_init(model, conf)
    
    if conf.train.finetuning:
        load_head = True if conf.vit.num_classes == 1000 else False
        model.from_pretrained(conf.train.checkpoint, load_head)        
    
    if conf.train.freeze_model:
        model = freeze_model(model)
    model = model.to(conf.train.device)
    
    optim, scheduler = create_optim(model, conf)
    
    loss_fn = nn.CrossEntropyLoss()
    
    if conf.train.from_checkpoint:
        model, optim, scheduler = training_from_checkpoint(conf, model, optim, scheduler)
    
    return model, optim, scheduler, loss_fn
    
def training_from_checkpoint(conf: omegaconf.DictConfig, 
                             model: nn.Module,
                             optim: torch.optim.Optimizer,
                             scheduler: torch.optim.lr_scheduler.LRScheduler,
                            ):
    checkpoint = torch.load(conf.inference.checkpoint)

    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optim, scheduler

def prepare_model(conf: omegaconf.DictConfig):
    model = ViT(conf)
    
    checkpoint = torch.load(r''.join(conf.inference.checkpoint),
                         map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(conf.train.device)
    model.eval()
    return model
    