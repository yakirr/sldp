import json

def add_default_params(args):
    if args.config is not None:
        # read in config file and replace '-' with '_' to match argparse behavior
        with open(args.config, 'r') as f:
            config = { k.replace('-','_') : v for k,v in json.load(f).items()}
        # overwrite with any non-None entries, resolving conflicts in favor of args
        config.update({
            k:v for k,v in args.__dict__.items()
            if (v is not None and v != []) or k not in config.keys()
            })
        # replace information in args
        args.__dict__ = config
