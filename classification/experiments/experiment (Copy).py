# %%
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from fc_fp_dataset import FCFPDataset
from fc_model_factory import create_model


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    num_folds = cfg.num_folds
    fp_ids = list(range(2000))
    random.shuffle(fp_ids)
    splits = np.array(np.array_split(fp_ids, num_folds))

    for fold in range(num_folds):
        model = create_model(cfg)
        model = model.to(device)

        opt_func = getattr(torch.optim, cfg.train.optimizer.name)
        optimizer = opt_func(model.parameters(),
                             lr=cfg.train.optimizer.learning_rate,
                             momentum=cfg.train.optimizer.momentum,
                             weight_decay=cfg.train.optimizer.weight_decay)

        fp_ids_val = splits[fold]
        fc_fp_ds_val = FCFPDataset(cfg.train.dataset.root, fp_ids_val)
        # fc_fp_ds_val.set_resize()
        DataLoader = torch.utils.data.DataLoader
        fc_fp_dl_val = DataLoader(fc_fp_ds_val,
                                  batch_size=cfg.train.batch_size,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=True, pin_memory=use_gpu)

        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])
        fc_fp_ds_tra = FCFPDataset(cfg.train.dataset.root, fp_ids_tra)
        # fc_fp_ds_tra.set_hflip()
        fc_fp_ds_tra.set_rotate()
        # fc_fp_ds_tra.set_resize()
        fc_fp_dl_tra = DataLoader(fc_fp_ds_tra,
                                  batch_size=cfg.train.batch_size,
                                  num_workers=cfg.train.num_workers,
                                  shuffle=True, pin_memory=use_gpu)
        for e in range(cfg.train.num_epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            for xi, yi, fp_class_2, gender, index in fc_fp_dl_tra:
                x = xi.to(device)
                y = yi.to(device)
                optimizer.zero_grad()
                y_out = model(x)
                criterion = getattr(torch.nn, cfg.train.loss_function)
                loss = criterion(y_out, y)
                _, labels = torch.max(y, 1)
                _, preds = torch.max(y_out, 1)
                total_correct += torch.sum(preds == labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(e, total_loss / len(fc_fp_dl_tra),
                  total_correct/len(fc_fp_ds_tra))

            # Perform validation every 10 epochs
            if e % 10 == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                with torch.no_grad():
                    for xi, yi, fp_class_2, gender, index in fc_fp_dl_val:
                        x = xi.to(device)
                        y = yi.to(device)
                        y_out = model(x)
                        loss = criterion(y_out, y)
                        _, labels = torch.max(y, 1)
                        _, preds = torch.max(y_out, 1)
                        val_correct += torch.sum(preds == labels)
                        val_loss += loss.item()
                print(f"Validation Epoch {e} - Loss: "
                      f"{val_loss / len(fc_fp_dl_val)}, Accuracy: "
                      f"{val_correct / len(fc_fp_ds_val)}")

# %%
