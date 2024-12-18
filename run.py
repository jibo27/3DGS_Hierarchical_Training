# Our code is based on https://github.com/NVlabs/CF-3DGS
import sys
from argparse import ArgumentParser

from trainer.ht3dgs_trainer import HTGaussianTrainer
from arguments import ModelParams, PipelineParams, OptimizationParams

from datetime import datetime
import yaml

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    model_cfg = lp.extract(args)
    pipe_cfg = pp.extract(args)
    optim_cfg = op.extract(args)
    
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

        params = {"OptimizationParams": optim_cfg, 
                   "ModelParams": model_cfg, "PipelineParams": pipe_cfg}
        for param in params.keys():
            if param in config.keys():
                cfg = params[param]
                for k, v in config[param].items():
                    setattr(cfg, k, v)


    if model_cfg.mode == "train":
        model_cfg.data_path = model_cfg.data_path_train
        model_cfg.data_type = model_cfg.data_type_train
    else:
        model_cfg.data_path = model_cfg.data_path_eval
        model_cfg.data_type = model_cfg.data_type_eval
    data_path = model_cfg.data_path


    trainer = HTGaussianTrainer(data_path, model_cfg, pipe_cfg, optim_cfg)
    start_time = datetime.now()
    if model_cfg.mode == "train":
        if pipe_cfg.train_mode == "progressive_training":
            trainer.progressive_training()
        elif pipe_cfg.train_mode == "hierarchical_training":
            trainer.hierarchical_training()
        elif pipe_cfg.train_mode == 'pose_only':
            trainer.train_pose_only()
        else:
            raise ValueError
    elif model_cfg.mode == "render":
        trainer.render_nvs(traj_opt=model_cfg.traj_opt)
    elif model_cfg.mode == "eval_nvs":
        trainer.eval_nvs()
    elif model_cfg.mode == "eval_pose":
        trainer.eval_pose()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
