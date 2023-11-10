import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='DQN', choices=["DQN"])
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--num_env_steps", type=int, default=300000,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='xxx',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=False, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='fairdiffusionmodel', help="specify the name of environment")
    
    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=3000, help="Max length for any episode")

    # network parameters
    parser.add_argument("--hidden_dim", type=int, default=256)

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')

    # DQN parameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discounter factor of future rewards')
    parser.add_argument("--eps_start", type=float, default=0.9,
                        help='the starting value of epsilon')
    parser.add_argument("--eps_end", type=float, default=0.9,
                        help='the final value of epsilon')
    parser.add_argument("--eps_decay", type=int, default=1000,
                        help='the final value of epsilon')
    parser.add_argument("--tau", type=float, default=0.005,
                        help='the update rate of the target network')
    parser.add_argument("--memory_size", type=int, default=10000)

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=10, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=10000, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="number of episodes of a single evaluation.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    return parser