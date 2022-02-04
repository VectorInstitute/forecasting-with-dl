import os
import torch 
import numpy as np
from utils.metrics import metric


def train_step(af_model, train_loader, model_optim, criterion, device, pred_len, start_token_len, features="M"):
    train_loss_list = []

    af_model.train()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

        model_optim.zero_grad()
        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :start_token_len, :], dec_inp], dim=1).float().to(device)

        outputs = af_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

        f_dim = -1 if features == 'MS' else 0
        batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
        loss = criterion(outputs, batch_y)
        train_loss_list.append(loss.item())

        loss.backward()
        model_optim.step()

        train_loss = np.average(loss.detach().cpu().numpy())
        train_loss_list.append(train_loss)

    avg_train_loss = np.mean(train_loss_list)

    return avg_train_loss

def val_step(af_model, val_loader, criterion, device, pred_len, start_token_len, features="M"):
    val_loss_list = []
    af_model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :start_token_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder

            outputs = af_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if features == 'MS' else 0
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            val_loss_list.append(loss)
            
    avg_val_loss = np.average(val_loss_list)

    return avg_val_loss

def test(af_model, test_loader, run_name, device, pred_len, start_token_len, features="M"):
    af_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + run_name, 'checkpoint.pth')))

    preds = []
    trues = []
    inputs = []
    folder_path = './test_results/' + run_name + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    af_model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :start_token_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            outputs = af_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if features == 'MS' else 0
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            inp = batch_x.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            inputs.append(inp)
            
    preds = np.array(preds)
    trues = np.array(trues)
    inputs = np.array(inputs)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

    return inputs, preds, trues