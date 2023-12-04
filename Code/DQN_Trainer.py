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
from Fair_Diffusion_Model import MDPEnv
from Runner import DQNRunner as Runner
from config import get_config


def make_train_env(all_args):
    def init_env():
        if all_args.env_name == "fairdiffusion":
            env_args = {"scenario": all_args.scenario,
                        "msg_limit": 10,
                        "obs_limit": 1000,
                        "feats_num": 50,
                        "neighbours_num": [5, 5],
                        "rela_dir": "../Dataset/Flood_input_data/",
                        "data_path_suffix": ".txt",
                        "graph": "/Flood512to521SVD",
                        "node_labels": "/node_labels",
                        "seed_sets": "/random_seeds",
                        "SE_matrix": "/SE_matrix",
                        "msg_matrix": "/msg_matrix"}

            env = MDPEnv(env_args=env_args)
        else:
            print("Can not support the " + all_args.env_name + " environment.")
            raise NotImplementedError
        return env
    return init_env


def make_eval_env(all_args):
    def init_env():
        if all_args.env_name == "fairdiffusion":
            env_args = {"scenario": all_args.scenario,
                        "msg_limit": 10,
                        "obs_limit": 1000,
                        "feats_num": 50,
                        "neighbours_num": [5, 5],
                        "rela_dir": "../Dataset/Flood_input_data/",
                        "data_path_suffix": ".txt",
                        "graph": "/Flood512to521SVD",
                        "node_labels": "/node_labels",
                        "seed_sets": "/random_seeds",
                        "SE_matrix": "/SE_matrix",
                        "msg_matrix": "/msg_matrix"}

            env = MDPEnv(env_args=env_args)
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
        device = torch.device("cuda:1")
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
    eval_env = make_eval_env(all_args)() if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "env": env,
        "eval_env": eval_env,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    env.close()
    if all_args.use_eval and eval_env is not env:
        eval_env.close()

    if all_args.use_wandb:
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])