import hashlib
import os
import socket
from disstl.utils.munging import flatten_dict
from datetime import datetime
from disstl.standards import STD_DATETIME_FORMAT


def get_execution_context():
    ec = {}
    ########################################################################################
    # populate some important values from the runtime context:
    ##
    # things about the machine/node/host
    hostname = os.getenv("HOSTNAME")
    if hostname is None:
        hostname = socket.getfqdn()
    ec["hostname"] = hostname
    ec["user"] = os.getenv("USER")
    # ec['gpu'] = [{'model': torch.cuda.get_device_name(device=f"cuda:{config['gpu']}"), 'ram': 0}]

    # things about the run (info from scheduler)
    # ec['pod_name']  = os.getenv("KUBERNETES_POD_NAME")

    # CI variables
    ec["ci_registry_image"] = os.getenv("CI_REGISTRY_IMAGE")
    ec["ci_commit_ref_name"] = os.getenv("CI_COMMIT_REF_NAME")
    ci_commit_sha = os.getenv("CI_COMMIT_SHA")
    ec["ci_commit_sha"] = ci_commit_sha if ci_commit_sha is not None else "unknown"

    # simple time stamp
    ec["timestamp"] = datetime.utcnow().strftime(STD_DATETIME_FORMAT)
    ##
    ########################################################################################
    return ec


def get_hashes(config):
    """
    This hashes a configuration dictionary by flattening it, sorting keys, and then
    hashing what is important to distinguish experiments from each other. It returns a tuple:
    word, hash, where word is simply the word in a list at the position corresponding to the hash as hexidecimal integer
    modulo 16. This helps readability of a name built from the output of this function. word--hash is easier to read
    than hash.

    There are two types of config keys to avoid paying attention to with this hash:
    1. config keys that will not impact the results of training, such as paths to where the output will be stored
    2. config keys that are so important that we want to know when two experiments are the same up to that key,val pair,
        * For example, the random seed used to sample data and instantiate networks. If the seed is used
        to compute the hash, then two experiments that differ only by seed will be hashed and named like this:
           - Wonderful--1d4h--seed__1 and Trite--23dg--seed__2
        If the seed is not used, then two experiments that differ only by seed will be hashed like this:
           - Wonderful--1d4h--seed__1 and Wonderful--1d4h--seed__2

    config_keys_to_avoid is the tuple of strings which will be used to filter out config keys before hashing
    """
    config = flatten_dict(config)
    config_keys_to_avoid = (
        "dirs.output_root_dir",
        "seed_everything",
        "logger",
        "experiment.name",
        "out_path",
        "hash_config",
    )
    sorted_config = dict(sorted(config.items()))
    hashable_keys = []
    for k, v in sorted_config.items():
        if k.startswith(config_keys_to_avoid) or k.endswith(config_keys_to_avoid):
            continue
        hashable_keys.append(k)
    hashable_config = "".join([str(v) for k, v in sorted_config.items() if k in hashable_keys])
    hashed_config = hashlib.md5(hashable_config.encode("utf-8")).hexdigest()
    adjectives = get_adjectives()
    adjective = adjectives[int(hashed_config, 16) % len(adjectives)]
    return adjective, hashed_config[:4]


def get_adjectives():
    dirname, _ = os.path.split(os.path.abspath(__file__))
    adjectives_fname = f"{dirname}/adjectives.txt"
    if not os.path.isfile(adjectives_fname):
        print("Error missing file: ", adjectives_fname)
        return ["no_config_hash"]
    with open(adjectives_fname) as f:
        adjectives = [line.strip() for line in f]
    return adjectives


def get_colors():
    dirname, _ = os.path.split(os.path.abspath(__file__))
    colors_fname = f"{dirname}/colors.txt"
    if not os.path.isfile(colors_fname):
        print("Error missing file: ", colors_fname)
        return ["no_color"]
    with open(colors_fname) as f:
        colors = [line.strip() for line in f]
    return colors
