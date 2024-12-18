"""
    From "https://github.com/hustvl/4DGaussians/blob/master/utils/params_utils.py#L1"
"""
def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    print(f"set args.{key} as {value}")
                    setattr(args, key, value)

    return args