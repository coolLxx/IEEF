import os
import argparse

from utils.utils import load_json, save_json, get_or_create_logger


CONFIGURATION_FILE_NAME = "run_config.json"

logger = get_or_create_logger(__name__)


def add_config(parser):
    """ define arguments """
    group = parser.add_argument_group("Construction")
    group.add_argument("-backbone", type=str, default="t5-large",  # t5-small
                       choices=["t5-small", "t5-base", "t5-large"])
    group.add_argument("-version", type=str, default="2.0",
                       choices=["2.0", "2.1"])
    group.add_argument("-data_version", type=str, default="3.0", choices=['1.0', '2.0', '3.0'], help="1.0: 原始版本 2.0: 续写版本")
    group.add_argument("-task", type=str, default="e2e",
                       choices=["dst", "e2e"])
    group.add_argument("-add_auxiliary_task", action="store_true")
    group.add_argument("-context_size", type=int, default=-1)
    group.add_argument("-ururu", action="store_true")

    group = parser.add_argument_group("Training")
    group.add_argument("-agent_type", type=str, default="us", choices=['ds', 'us'],  # ds
                        help="us: user simulator; ds: dialog system")
    group.add_argument("-batch_size", type=int, default=4)  # 8
    group.add_argument("-epochs", type=int, default=2)  # 20
    group.add_argument("-aug_cutoff_ratio", type=float, default=0.02)  # <MOD> <ADD>
    group.add_argument("-warmup_steps", type=int, default=-1)
    group.add_argument("-warmup_ratio", type=float, default=0.2)
    group.add_argument("-learning_rate", type=float, default=5e-4)  # 5e-4
    group.add_argument("-weight_decay", type=float, default=0.0)
    group.add_argument("-grad_accum_steps", type=int, default=1)
    group.add_argument("-max_grad_norm", type=float, default=1.0)
    group.add_argument("-aux_loss_coeff", type=float, default=0.5)
    group.add_argument("-resp_loss_coeff", type=float, default=1.0)
    group.add_argument("-bspn_loss_coeff", type=float, default=1.0)
    group.add_argument("-num_train_dialogs", type=int, default=-1)  # -1
    group.add_argument("-train_from", type=str, default=None)
    group.add_argument("-no_validation", action="store_true")
    group.add_argument("-no_learning_rate_decay", action="store_true")

    group = parser.add_argument_group("Prediction")
    group.add_argument("-pred_data_type", type=str, default="dev",
                       choices=["dev", "test"])
    group.add_argument("-overwrite_with_span", action="store_true")
    group.add_argument("-beam_size", type=int, default=1)
    group.add_argument("-do_sample", action="store_true")
    group.add_argument("-top_k", type=int, default=0)
    group.add_argument("-top_p", type=float, default=0.7)
    group.add_argument("-temperature", type=float, default=1.0)
    group.add_argument("-use_true_dbpn", action="store_true")
    group.add_argument("-use_true_curr_aspn", action="store_true")
    group.add_argument("-use_true_prev_bspn", action="store_true")
    group.add_argument("-use_true_prev_aspn", action="store_true")
    group.add_argument("-use_true_prev_resp", action="store_true")
    group.add_argument("-top_n", type=int, default=5)
    group.add_argument("-output", type=str, default=None)  # None
    group.add_argument("-predict_agent_type", type=str, default="ds", choices=['ds', 'us'], 
                        help="us: user simulator; ds: dialog system")

    group = parser.add_argument_group("Misc")
    # group.add_argument("-run_type", type=str, required=True,
    #                    choices=["train", "predict"])
    group.add_argument("-run_type", type=str, default="train", choices=["train", "predict"])  # <MOD>
    group.add_argument("-excluded_domains", type=str, nargs="+")
    group.add_argument("-model_dir", type=str, default="./simulator_t5_large_data3.0_interact")  # default="checkpoints")
    group.add_argument("-seed", type=int, default=42)
    group.add_argument("-ckpt", type=str, default=None)
    group.add_argument("-log_frequency", type=int, default=100)  # -1
    group.add_argument("-max_to_keep_ckpt", type=int, default=10)
    group.add_argument("-num_gpus", type=int, default=1)
    group.add_argument("-model_name", type=str, default='mttod', help = 'mttod, pptod, ubar, galaxy')

    return parser.parse_args()  # <MOD> <ADD>


def check_config(parser):
    """ parse arguments and check configuration """
    cfg = parser.parse_args()

    if cfg.run_type == "predict" and cfg.ckpt is None:
        raise ValueError("To predict output, set ckpt to run.")

    # update arguments for consistency with checkpoint model
    ckpt_path = cfg.ckpt or cfg.train_from
    if ckpt_path is not None:
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))

        if cfg.ckpt is not None:
            setattr(cfg, "model_dir", ckpt_dir)

        ckpt_cfg = load_json(os.path.join(ckpt_dir, CONFIGURATION_FILE_NAME))

        for group in parser._action_groups:
            if group.title == "Construction":
                for argument in group._group_actions:
                    key = argument.dest

                    new_value = getattr(cfg, key)
                    old_value = ckpt_cfg.get(key)

                    if old_value is not None and new_value != old_value:
                        logger.warning(
                            "Update argument for consistency (%s: %s -> %s)",
                            key, str(new_value), str(old_value))

                        setattr(cfg, key, old_value)

    # check invalid combinations of configurations
    if cfg.task == "dst" and cfg.run_type == "predict":
        setattr(cfg, "use_true_dbpn", True)
        setattr(cfg, "use_true_prev_aspn", True)
        setattr(cfg, "use_true_prev_resp", True)

    if cfg.task == "dst" and cfg.use_true_prev_bspn:
        raise ValueError(
            "To evaluate dialog state tracking, generated belief state should be used")

    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)
        save_json(vars(cfg), os.path.join(
            cfg.model_dir, CONFIGURATION_FILE_NAME))

    return cfg


def get_config():
    """ return ArgumentParser Instance """
    parser = argparse.ArgumentParser(
        description="Configuration of task-oriented dialogue model with multi-task learning.")

    args = add_config(parser)  # add_config(parser)  # <MOD>

    # args = check_config(parser)
    args.max_to_keep_ckpt = args.epochs

    return args


if __name__ == "__main__":
    configs = get_config()
