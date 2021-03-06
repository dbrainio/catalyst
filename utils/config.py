import os
import json
import yaml
import copy
from collections import OrderedDict
from catalyst.utils.misc import merge_dicts


def load_ordered_yaml(
    stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict
):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def save_config(config, logdir):
    os.makedirs(logdir, exist_ok=True)
    with open("{}/config.json".format(logdir), "w") as fout:
        json.dump(config, fout, indent=2, ensure_ascii=False)


def parse_config_args(*, config, args, unknown_args):
    for arg in unknown_args:
        arg_name, value = arg.split("=")
        arg_name = arg_name[2:]
        value_content, value_type = value.rsplit(":", 1)

        if "/" in arg_name:
            arg_names = arg_name.split("/")
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))

            config_ = config
            for arg_name in arg_names[:-1]:
                if arg_name not in config_:
                    config_[arg_name] = {}

                config_ = config_[arg_name]

            config_[arg_names[-1]] = arg_value
        else:
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))
            args.__setattr__(arg_name, arg_value)
    return config, args


def parse_args_uargs(args, unknown_args, dump_config=False):
    args_ = copy.deepcopy(args)

    # load params
    config = {}
    for config_path in args_.config.split(","):
        with open(config_path, "r") as fin:
            if config_path.endswith("json"):
                config_ = json.load(fin, object_pairs_hook=OrderedDict)
            elif config_path.endswith("yml"):
                config_ = load_ordered_yaml(fin)
            else:
                raise Exception("Unknown file format")
        config = merge_dicts(config, config_)

    config, args_ = parse_config_args(
        config=config, args=args_, unknown_args=unknown_args
    )

    # hack with argparse in config
    config_args = config.get("args", None)
    if config_args is not None:
        for key, value in config_args.items():
            arg_value = getattr(args_, key, None)
            if arg_value is None:
                arg_value = value
            setattr(args_, key, arg_value)

    logdir = getattr(args_, "logdir", None)
    if dump_config and logdir is not None:
        save_config(config=config, logdir=logdir)

    return args_, config
