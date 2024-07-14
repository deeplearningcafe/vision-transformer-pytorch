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
import random

random.seed(46)
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
    train_acc = 0.0
    val_acc = 0.0
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
            
            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == label.data) / preds.shape[0]
            train_acc += acc.item()

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5, norm_type=2)
            optim.step()
            
        running_epochs += 1

        # we will do validation after the epoch ends
        if current_epoch % conf.train.eval_epoch == 0:
            for img, label in val_loader:
                output = model(img)                
                
                loss = loss_fn(output, label)
                val_losses += loss.item()
                
                preds = torch.argmax(output, dim=1)
                acc = torch.sum(preds == label.data) / preds.shape[0]
                val_acc += acc.item()
                
        if current_epoch % conf.train.log_epoch == 0:
            train_losses /= (len(train_loader) * running_epochs)
            val_losses /= len(val_loader)
            train_acc /= (len(train_loader) * running_epochs)
            val_acc /= len(val_loader)
            
            max_norm = max(grad_norms)
            mean_norm = sum(grad_norms)/ (len(train_loader) * running_epochs)

            print(f"Epoch {current_epoch}  || Train Loss : {train_losses} || Validation Loss : {val_losses} || Learning rate: {scheduler.state_dict()['_last_lr'][0]} || Mean Norm: {mean_norm} || Max Norm: {max_norm} || Train Accuracy: {train_acc} || Val Accuracy: {val_acc}" # || Trained Tokens: {total_tokens}"
                    )

            log_epoch = {'epoch': current_epoch+1, 'train_loss': train_losses, 'val_loss': val_losses,
                            "mean_norm": mean_norm, "max_norm": max_norm, "train_accuracy": train_acc,
                            "val_accuracy": val_acc, "learning_rate": scheduler.state_dict()['_last_lr'][0]}
            
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv(log_path, index=False)
            train_losses = 0
            val_losses = 0
            train_acc = 0
            val_acc = 0
            del grad_norms
            grad_norms = []
            running_epochs = 0
            torch.cuda.empty_cache()
            gc.collect()
        
        if current_epoch % conf.train.save_epoch == 0:
            print("Saving")
            torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'seed': 46,
            }, conf.train.save_path + '/unet_' + str(current_epoch+1) + '.pth')

            # when the epoch ends we call the scheduler
        scheduler.step()
        current_epoch += 1
        pbar.update(1)
        
    # after ending training save training state and model
    
    print("Finished Training!")
    return logs

def overfit_one_batch(model: nn.Module,
             batch: tuple[torch.tensor],
             optim: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.LRScheduler,
             loss_fn: torch.nn.Module,
             conf: omegaconf.DictConfig,
             output_log:bool=True,
             save_update_ratio:bool=False):
    
    current_step = 0
    img = batch[0]
    label = batch[1]
    pbar = tqdm(total=conf.overfit_one_batch.max_steps)
    if save_update_ratio:
        diffs = {"predictor": []}
        layers = {"predictor": model.mlp_head.predictor.weight.detach().cpu().clone()}
    
    # logs
    logs = {}
    losses = []
    grad_norms = []
    lrs = []
    accs = []
    
    log_path = os.path.join(conf.train.log_path, f"overfit_batch_{datetime.now().strftime(r'%Y%m%d-%H%M%S')}.csv")

    while current_step < conf.overfit_one_batch.max_steps:
        output = model(img)
        loss = loss_fn(output, label)
        
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == label.data) / preds.shape[0]
        accs.append(acc.item())
        losses.append(loss.item())
        
        optim.zero_grad()
        loss.backward()
        
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        grad_norms.append(norm.item())
        lrs.append(scheduler.state_dict()['_last_lr'][0])
        
        # clip grad
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5, norm_type=2)
        optim.step()
        scheduler.step()
        
        current_step += 1
        pbar.update(1)
        if current_step > 1 and abs(losses[-2]-losses[-1]) < conf.overfit_one_batch.tolerance:
            break

        if current_step % conf.overfit_one_batch.logging_steps == 0 or current_step == conf.train.max_epochs-1:
            print(f"Step {current_step}  || Loss : {losses[-1]} || Learning rate: {lrs[-1]} || Norm: {grad_norms[-1]} || Acc: {accs[-1]}")
        
    logs = {'losses': losses, "gradient_norm": grad_norms, "learning_rate": lrs, "accuracy": accs}
    if output_log:
        df = pd.DataFrame(logs)
        df.to_csv(log_path, index=False)
    if save_update_ratio:
        return logs, diffs
    return logs

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: omegaconf.DictConfig):
    model, optim, scheduler, loss_fn = prepare_training(conf)
    
    train_loader, val_loader = create_dataset(conf, True)
    
    if os.path.isdir(conf.train.log_path) == False:
        os.makedirs(conf.train.log_path)

    if conf.overfit_one_batch.overfit:
        conf.vit.num_layers = conf.vit.num_layers//2

        batch = next(iter(train_loader))
        overfit_one_batch(model, batch, optim, scheduler, loss_fn, conf, True, True)
    
    else:
        conf.train.save_path = os.path.join(conf.train.save_path, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}")
        if os.path.isdir(conf.train.save_path) == False:
            os.makedirs(conf.train.save_path)

        training(model, train_loader, val_loader, optim, scheduler, loss_fn, conf)

if __name__ == "__main__":
    main()