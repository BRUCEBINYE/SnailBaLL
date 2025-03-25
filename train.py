
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

#from datasetsContact import *
from datasetsMinus import *
from models import Net
from utils import EarlyStopping, LRScheduler
# from tqdm import tqdm

# training function
def fit(model, trainloader, optimizer, criterion, device):
    # print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    # prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    # for i, data in prog_bar:
    for data in trainloader:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy


# validation function
def validate(model, valloader, criterion, device):
    # print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    # prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        # for i, data in prog_bar:
        for data in valloader:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy
    

def train(model, optimizer, datasets, epochs,
          lr_scheduler=None, early_stopping=None, device='cuda'):
    
    model = model.to(device)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # lists to store per-epoch loss and accuracy values
    # train_loss, train_accuracy = [], []
    # val_loss, val_accuracy = [], []
    start = time.time()
    for epoch in range(epochs):
        train_data_running_loss = 0
        train_data_running_acc = 0
        for trainloader, _ in datasets:
            
            train_data_loss, train_data_accuracy = fit(
                model, trainloader, optimizer, criterion, device
            )
            train_data_running_loss += train_data_loss
            train_data_running_acc += train_data_accuracy
        train_epoch_loss = train_data_running_loss/len(datasets)
        train_epoch_accuracy = train_data_running_acc/len(datasets)
        
        val_data_running_loss = 0
        val_data_running_acc = 0
        for _, testloader in datasets:
            val_data_loss, val_data_accuracy = validate(
                model, testloader, criterion, device
            )
            val_data_running_loss += val_data_loss
            val_data_running_acc += val_data_accuracy
        val_epoch_loss = val_data_running_loss/len(datasets)
        val_epoch_accuracy = val_data_running_acc/len(datasets)

        # train_loss.append(train_epoch_loss)
        # train_accuracy.append(train_epoch_accuracy)
        # val_loss.append(val_epoch_loss)
        # val_accuracy.append(val_epoch_accuracy)
        if lr_scheduler:       
            lr_scheduler(val_epoch_loss)
        if early_stopping:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break
        print(f"Epoch {epoch+1}, "
              f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}")
    end = time.time()
    print(f"Training time: {(end-start)/60:.3f} minutes")


def run(datasets, d_model, ffn_hidden, n_head, 
        n_layers, n_hidden, 
        drop_prob, n_input, epochs,
        lr, device, save_path, save_model=False):
    '''
    Args:
        n_input: [length, dim], # n_input = [130, 1024]
        drop_prob: # drop_prob = 0.05
    '''
    
    #drop_prob = 0.05
    #n_input = [130, 1024]
    #epochs = 100

    os.makedirs(save_path, exist_ok=True)

    model = Net(d_model, ffn_hidden, n_head, n_layers, 
                n_hidden, drop_prob, n_input, n_out=2, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = LRScheduler(optimizer)  
    early_stopping = EarlyStopping() 
    
    train(model, optimizer, datasets, epochs, lr_scheduler, 
          early_stopping, device)
    
    if save_model:
        checkpoint = {'d_model': d_model, 
                      'ffn_hidden': ffn_hidden, 
                      'n_head': n_head, 
                      'n_layers': n_layers, 
                      'drop_prob': drop_prob, 
                      'n_hidden': n_hidden,
                      'n_input': n_input, 
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, save_path + f'checkpoint_{d_model}_{ffn_hidden}_{n_head}_{n_layers}_{n_hidden}.pth')






