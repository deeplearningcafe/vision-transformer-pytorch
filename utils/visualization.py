import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_logs(logs_path:str,):
    df = pd.read_csv(r"".join(logs_path))
    print(f"Max norm: {df['max_norm'].max()}")
    print(f"Min train loss: {df['train_loss'].min()}")
    print(f"Min val loss: {df[df['val_loss']!=0.0]['val_loss'].min()}")
    
    x = np.arange(len(df["train_loss"]))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, df["train_loss"],  label='train_loss')
    ax.scatter(x, df["val_loss"], label='val_loss')
    ax.plot(x, df["mean_norm"],  label='mean_norm')
    ax.plot(x, df["max_norm"],  label='max_norm')

    ax.plot(x, df["learning_rate"]*5000, label='lr')
    ax.plot(x, df["train_accuracy"]*10, label='train_accuracy')
    ax.plot(x, df["val_accuracy"]*10, label='val_accuracy')


    ax.set(xlabel='steps', ylabel='loss and gradients',)
    ax.grid()
    ax.set_ylim(ymin=0, ymax=10)
    ax.legend()
    plt.show()

def plot_gradients(layers_list:list[str], model:torch.nn.Module):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []

    for name, param in model.named_parameters():
        t = param.grad
        if name in layers_list:
            print('%5s %10s | mean %+f | std %e | grad:data ratio %e' % (name, tuple(param.shape), t.mean(), t.std(), t.std() / param.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{tuple(param.shape)} {name}')
    plt.legend(legends)
    plt.title('weights gradient distribution')
    plt.show()
    plt.close()
