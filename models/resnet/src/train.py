import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn


def setlr(optimizer, lr):
    """
    Set learning rate
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def lr_decay(optimizer, epoch, learning_rate):
    """
    Learning rate decay
    """
    if epoch % 10 == 0:
        new_lr = learning_rate / (10 ** (epoch // 10))
        optimizer = setlr(optimizer, new_lr)
        print(f"Changed learning rate to {new_lr}")
    return optimizer


def train_classification(
    model,
    train_loader,
    valid_loader,
    epochs,
    optimizer,
    train_losses,
    valid_losses,
    device,
    learning_rate,
    loss_fn,
    change_lr=None,
):
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        batch_losses = []
        if change_lr:
            optimizer = change_lr(optimizer, epoch, learning_rate)

        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x).squeeze()
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)

        print(f"Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}")
        model.eval()
        batch_losses = []
        trace_y = []
        trace_yhat = []
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)  # long
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)

        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        print(
            f"Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}"
        )

def train_regression(
    model,
    train_loader,
    valid_loader,
    epochs,
    optimizer,
    train_losses,
    valid_losses,
    device,
    learning_rate,
    loss_fn,
    change_lr=None,
):
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        batch_losses = []
        if change_lr:
            optimizer = change_lr(optimizer, epoch, learning_rate)

        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).squeeze()
            y_hat = model(x).squeeze()
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)

        print(f"Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}")
        model.eval()
        batch_losses = []
        trace_y = []
        trace_yhat = []
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).squeeze()  # long
            y_hat = model(x).squeeze()
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        error = np.mean(np.abs(trace_yhat - trace_y))
        print(
            f"Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-MAE : {error}"
        )