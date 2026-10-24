import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.learning_rate
            if epoch < 3
            else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "step":
        step_size = getattr(args, "lr_step_size", 10)
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // step_size))}
    elif args.lradj == "cos":
        T = max(1, args.train_epochs)
        eta_min = getattr(args, "lr_eta_min", 0.0)
        t = min(epoch, T)
        lr_adjust = {
            epoch: eta_min
            + 0.5 * (args.learning_rate - eta_min) * (1 + np.cos(np.pi * t / T))
        }
    elif args.lradj == "cos_warmup":
        warmup = getattr(args, "lr_warmup", 5)
        T = max(1, args.train_epochs)
        eta_min = getattr(args, "lr_eta_min", 0.0)
        if epoch <= warmup:
            lr = args.learning_rate * epoch / max(1, warmup)
        else:
            prog = (epoch - warmup) / max(1, T - warmup)
            prog = min(1.0, prog)
            lr = eta_min + 0.5 * (args.learning_rate - eta_min) * (
                1 + np.cos(np.pi * prog)
            )
        lr_adjust = {epoch: lr}
    elif args.lradj == "3":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1
        }
    elif args.lradj == "4":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1
        }
    elif args.lradj == "5":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1
        }
    elif args.lradj == "6":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1
        }
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_to_disk=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        # Without a checkpoint file the best weights are kept in best_state.
        self.save_to_disk = save_to_disk
        self.best_state = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        if self.save_to_disk:
            torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        else:
            self.best_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def visual(true, preds=None, name="./pic/test.pdf"):
    """Plot ground truth vs prediction to `name`."""
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")
