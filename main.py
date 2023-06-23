import sys

from fairlib.src.networks import get_main_model
from fairlib.src.utils.utils import seed_everything
from fairlib.src import base_options
import os
import sys
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':

    dataset = sys.argv[1]
    demon = sys.argv[2]

    back_up_repo = 'result_' + dataset + '_demon_' + demon

    if dataset == 'Bios':
        args = {
            "dataset": "Bios_gender",
            "emb_size": 768,
            "num_classes": 28,
            "batch_size": 128,
            "lr": 0.0001,
            "same_batch": True,
            "same_mlp": False,
            "eo_optimization": False,
            "data_dir": "./data/bios",
            "device_id": 0,
            "exp_id": back_up_repo,
            "normalize": False,
            "adv_level": 'output',
            "beta": 1,
            "alpha": 1,
            "w_gp": 0,
            "beta_auto": "none",
            "return_first_hidden": False,
            "dropout_w": 1,
            "nj": 5,
            "ni": 20,
            "KL": False,
            "lr_critic": 5e-5,
            "clamp": 0.01,
        }

        debias_options = base_options.BaseOptions()
        debias_state = debias_options.get_state(args=args, silence=True)

        seed_everything(2022)

        debias_model = get_main_model(debias_state)
        # number of epoch fixed in utils > function train_self()
        # debias_model.train_self()

        if not args["same_mlp"]:
            args = {
                "dataset": "Bios_gender",
                "emb_size": 768,
                "num_classes": 2,
                "batch_size": 8,
                "lr": 0.00001,
                "data_dir": "./data/bios",
                "device_id": 0,
                "adv_level": 'output',
                "return_first_hidden": False,
            }
            debias_options = base_options.BaseOptions()
            debias_state = debias_options.get_state(args=args, silence=True)
            mlp_critic = get_main_model(debias_state)
        else:
            mlp_critic = None

        if demon == 'marked_personas':
            path_critic = "demon_MLP_bios_dv2_story.pt"
        elif demon == 'EEC':
            path_critic = "demon_MLP_bios_EEC.pt"
        else:
            path_critic = "demon_MLP_bios.pt"

        debias_model.train_fair_wasserstein(mlp_critic, path_critic)
    elif dataset == "Moji":
        args = {
            "dataset": "Moji",
            "emb_size": 2304,
            "num_classes": 2,
            "batch_size": 128,
            "lr": 0.00001,
            "same_batch": True,
            "same_mlp": False,
            "eo_optimization": False,
            "data_dir": "./data/moji",
            "normalize": False,
            "device_id": 0,
            "w_gp": 0,
            "exp_id": back_up_repo,
            "adv_level": "last_hidden",
            "beta": 1,
            "alpha": 1,
            "nj": 5,
            "ni": 5,
            "KL": False,
            "beta_auto": False,
            "lr_critic": 5e-5,
            "clamp": 0.01,
            "dropout_w": 1,
            "return_first_hidden": False,
        }

        debias_options = base_options.BaseOptions()
        debias_state = debias_options.get_state(args=args, silence=True)

        seed_everything(2022)

        debias_model = get_main_model(debias_state)
        # number of epoch fixed in utils > function train_self()
        # debias_model.train_self()

        if not args["same_mlp"]:
            args = {
                "dataset": "Moji",
                "emb_size": 2304,
                "num_classes": 2,
                "batch_size": 128,
                "lr": 0.00001,
                "n_hidden": 2,
                "data_dir": "./data/moji",
                "device_id": 0,
                "adv_level": "last_hidden",
            }
            debias_options = base_options.BaseOptions()
            debias_state = debias_options.get_state(args=args, silence=True)
            mlp_critic = get_main_model(debias_state)
        else:
            mlp_critic = None

        debias_model.train_fair_wasserstein_mi(mlp_critic, "demon_MLP_moji.pt")
    else:
        print("Error wrong dataset name, expect Bios or Moji")