import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import argparse

sys.path.append("../../")
from MDP_Env import MDP_Env
from Earthquake_GFN_MIA_Runner import GFNRunner as Runner
from config import get_config


def make_train_env(all_args):
    def init_env():
        if all_args.env_name == "fairdiffusion":
            env_args = {"scenario": "earthquake",
                        "msg_limit": 20,
                        "feats_num": 2,
                        "seeds_num": 5,
                        "neighbourhood_depth": 2,
                        "episodes": 1059,
                        "rela_dir": "../Earthquake/",
                        "data_path_suffix": ".txt",
                        "graph": "/DiG_Tokens",
                        "seeds": "None",
                        "event": "/Event_Tokens"}
            env = MDP_Env(env_args=env_args)
            print("70%Training")
        else:
            print("Can not support the " + all_args.env_name + " environment.")
            raise NotImplementedError
        return env
    return init_env


def make_eval_env(all_args):
    def init_env():
        if all_args.env_name == "fairdiffusion":
            env_args = {"scenario": "earthquake",
                        "msg_limit": 10,
                        "feats_num": 768,
                        "seeds_num": 10,
                        "neighbourhood_depth": 2,
                        "episodes": 1059,
                        "training_epi": 741,
                        "rela_dir": "../Earthquake/",
                        "data_path_suffix": ".txt",
                        "graph": "/G",
                        "subgraph":  "/Interest_Region/ir{}",
                        "node_labels": "/Evaluation/irlabels/ir{}_labels",
                        "seeds": "/RIS/{}/ir{}_seeds",
                        "SE_mat_eval": "/Evaluation/SE_Tensor",
                        "SE_mat_training": "/Training/SE_Tensor",
                        "msg_matrix": "/G_Tensor"}
            env = MDP_Env(env_args=env_args)
            print("Eval")
        else:
            print("Can not support the " + all_args.env_name + " environment.")
            raise NotImplementedError
        return env
    return init_env


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='flood')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_episodes', type=int, default=10000)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("mumu config: ", all_args)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    all_args.use_wandb = True
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity='newonepointfive',
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         # job_type = "evaluating",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    env = make_train_env(all_args)()
    # env = make_eval_env(all_args)()

    config = {
        "all_args": all_args,
        "env": env,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    env.close()

    if all_args.use_wandb:
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
