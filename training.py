import numpy as np
import torch
import torch.utils
import torch.utils.data
import omegaconf
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import hydra
from utils.prepare_model import prepare_training
from utils.prepare_data import create_dataset
import os
from datetime import datetime
import gc

np.random.seed(46)
torch.manual_seed(46)

torch.backends.cudnn.benchmark = True

def training(model: nn.Module,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             optim: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.LRScheduler,
             loss_fn: torch.nn.Module,
             conf: omegaconf.DictConfig):
    current_epoch = 0
    running_epochs = 0
    train_losses = 0.0
    val_losses = 0.0
    grad_norms = []
    logs = []

    log_path = os.path.join(conf.train.log_path, f"log_output_{datetime.now().strftime(r'%Y%m%d-%H%M%S')}.csv")
    pbar = tqdm(total=conf.train.max_epochs)
    print("Start training!")
    
    while current_epoch < conf.train.max_epochs:
        for img, label in train_loader:
            output = model(img)
            
            loss = loss_fn(output, label)
            train_losses += loss.item()
            
            optim.zero_grad()
            loss.backward()
            
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            grad_norms.append(norm.item())
            
            # clip grad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optim.step()
            
        running_epochs += 1

        # we will do validation after the epoch ends
        if current_epoch % conf.train.eval_epoch == 0:
            for img, label in val_loader:
                output = model(img)                
                
                loss = loss_fn(output, label)
                val_losses += loss.item()
                
        if current_epoch % conf.train.log_epoch == 0:
            train_losses /= (len(train_loader) * running_epochs)
            val_losses /= len(val_loader)
            max_norm = max(grad_norms)
            mean_norm = sum(grad_norms)/ (len(train_loader) * running_epochs)

            print(f"Epoch {current_epoch}  || Train Loss : {train_losses} || Validation Loss : {val_losses} || Learning rate: {scheduler.state_dict()['_last_lr'][0]} || Mean Norm: {mean_norm} || Max Norm: {max_norm}" # || Trained Tokens: {total_tokens}"
                    )

            log_epoch = {'epoch': current_epoch+1, 'train_loss': train_losses, 'val_loss': val_losses,
                            "mean_norm": mean_norm, "max_norm": max_norm,
                            "learning_rate": scheduler.state_dict()['_last_lr'][0]}
            
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv(log_path, index=False)
            train_losses = 0
            val_losses = 0
            del grad_norms
            grad_norms = []
            running_epochs = 0
            torch.cuda.empty_cache()
            gc.collect()
        
        if current_epoch % conf.train.save_epoch == 0:
            print("Saving")
            torch.save(model.state_dict(), conf.train.save_path + '/unet_' + str(current_epoch+1) + '.pth')

            # when the epoch ends we call the scheduler
        scheduler.step()
        current_epoch += 1
        pbar.update(1)
        
    print("Finished Training!")
    return logs

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: omegaconf.DictConfig):
    model, optim, scheduler, loss_fn = prepare_training(conf)
    
    train_loader, val_loader = create_dataset(conf)
    
    conf.train.save_path = os.path.join(conf.train.save_path, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}")
    if os.path.isdir(conf.train.save_path) == False:
        os.makedirs(conf.train.save_path)
    if os.path.isdir(conf.train.log_path) == False:
        os.makedirs(conf.train.log_path)

    training(model, train_loader, val_loader, optim, scheduler, loss_fn, conf)

if __name__ == "__main__":
    main()