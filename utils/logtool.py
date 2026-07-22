import os

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, ListConfig, OmegaConf


def print_header(text):
    print("-" * len(text))
    print(text)
    print("-" * len(text))


def print_train(args):
    print_header("*** Training Configurations ***")
    print(
        f"|- Input/Look Back/Prediction/Slide Horizon: "
        f"{args.input_len,args.look_back,args.pred_len,args.slide_step}"
    )
    print(
        f"|- Dims: input={args.enc_in}, mark={args.enc_mark_in}, "
        f"model_dim={args.d_model}, output={args.c_out}"
    )
    print(f"|- Number of trainable parameters: {args.num_params}")
    print(f"|- Experiment name: {args.exp_id}")
    print(f"|- Checkpoints save to: {args.best_checkpoint_path}")
    print(f"|- Predictions save to: {args.best_prediction_path}")
    print(f"|- Metrics save to: {args.best_metrics_path}")


def set_wandb(args):
    os.makedirs(args.log_dir, exist_ok=True)
    if not args.wandb:
        return None

    import wandb

    project_name = (
        f"{args.project_name}-{args.dloader_name}-"
        f"{args.input_len}|{args.look_back}|{args.pred_len}|{args.slide_step}"
    )
    wandb.require("core")
    wandb.init(
        config={},
        entity=args.wandb_entity,
        name=args.exp_id,
        project=project_name,
        dir=args.log_dir,
    )
    wandb.config.update(args)
    return wandb


def print_config(config, resolve=True, name="CONFIG"):
    tree = rich.tree.Tree(name, style="bold red")
    for field in config.keys():
        branch = tree.add(field, style="bold black")
        section = config.get(field)
        if isinstance(section, (DictConfig, ListConfig)):
            section = OmegaConf.to_yaml(section, resolve=resolve)
        branch.add(rich.syntax.Syntax(str(section), "yaml"), style="red")
    rich.print(tree)
