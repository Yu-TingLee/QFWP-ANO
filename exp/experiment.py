import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from models import qfwp_ano
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

MODELS = ("QFWP-ANO-R", "QFWP-ANO-O", "QFWP-ANO-RO")


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = qfwp_ano.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _predict(self, batch_x):
        """Forecast for the last pred_len steps and the target channels."""
        outputs = self.model(batch_x)
        f_dim = -1 if self.args.features == "MS" else 0
        return outputs[:, -self.args.pred_len :, f_dim:]

    def _target(self, batch_y):
        f_dim = -1 if self.args.features == "MS" else 0
        return batch_y[:, -self.args.pred_len :, f_dim:]

    def vali(self, vali_loader, criterion):
        self.model.eval()
        losses, maes = [], []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                outputs = self._predict(batch_x).cpu()
                true = self._target(batch_y)
                losses.append(criterion(outputs, true).item())
                maes.append((outputs - true).abs().mean().item())
        self.model.train()
        return float(np.average(losses)), float(np.average(maes))

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")
        _, test_loader = self._get_data("test")

        no_checkpoint = bool(self.args.no_checkpoint)
        path = os.path.join(self.args.checkpoints, setting)
        if not no_checkpoint:
            os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, save_to_disk=not no_checkpoint
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            epoch_start = time.time()
            for batch_x, batch_y, _, _ in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                loss = criterion(self._predict(batch_x), self._target(batch_y))
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        optimizer, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()

            train_loss = float(np.average(train_losses))
            vali_loss, vali_mae = self.vali(vali_loader, criterion)
            test_loss, test_mae = self.vali(test_loader, criterion)
            print(
                f"Epoch {epoch + 1} | time {time.time() - epoch_start:.1f}s | train {train_loss:.6f} | val MSE {vali_loss:.6f} MAE {vali_mae:.6f} | "
                f"test MSE {test_loss:.6f} MAE {test_mae:.6f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "TST":
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args)

        if no_checkpoint:
            if early_stopping.best_state is not None:
                self.model.load_state_dict(early_stopping.best_state)
        else:
            self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth")))
        return self.model

    def test(self, setting, load_checkpoint=False):
        _, test_loader = self._get_data("test")
        if load_checkpoint:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
                )
            )

        folder = os.path.join(
            "results", "mtsf", self.args.model + self.args.save_suffix, setting
        )
        plots = os.path.join(folder, "test_plots")
        os.makedirs(plots, exist_ok=True)

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                outputs = self._predict(batch_x).cpu().numpy()
                true = self._target(batch_y).numpy()
                preds.append(outputs)
                trues.append(true)
                if i % 20 == 0:
                    inp = batch_x.cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pred = np.concatenate((inp[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pred, os.path.join(plots, f"{i}.pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}, rse:{rse}")

        np.savez(
            os.path.join(folder, "metrics.npz"),
            mae=np.asarray(mae),
            mse=np.asarray(mse),
            rmse=np.asarray(rmse),
            mape=np.asarray(mape),
            mspe=np.asarray(mspe),
            rse=np.asarray(rse),
            corr=np.asarray(corr),
        )
        np.save(os.path.join(folder, "pred.npy"), preds)
        np.save(os.path.join(folder, "true.npy"), trues)
