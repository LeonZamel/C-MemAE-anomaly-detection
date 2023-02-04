import glob
import os
import random
import re
from datetime import datetime

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from IPython.display import display
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

from systems.memae_autoencoder_system import MemaeSystem

target_regex = re.compile(r"target_([^/|\\]+)")


def extract_classes_from_path(ckpt_path):
    if not ckpt_path:
        return None, False
    targets_str = target_regex.search(ckpt_path).group(1)
    if targets_str is None:
        raise ValueError(f"Targets cannot be determined from checkpoint {ckpt_path}")
    dataset = None
    if "mnist" in ckpt_path:
        dataset = "mnist"
    elif "cifar" in ckpt_path:
        dataset = "cifar"
    else:
        raise ValueError(f"Data set cannot be determined from checkpoint {ckpt_path}")

    all_targets = targets_str.split(",")

    return ",\n".join(map(NICE_NAMING_CLASSES[dataset].__getitem__, all_targets)), len(all_targets) > 1


NICE_NAMING_VALUES = {
    "model.model_type": {
        "mnist-memae": "MemAE",
        "mnist-memae-flat": "MemAE flat",
        "mnist-ae": "AE",
        "mnist-ae-flat": "AE flat",
        "cifar-memae": "MemAE",
        "cifar-memae-flat": "MemAE flat",
        "cifar-ae": "AE",
        "cifar-ae-flat": "AE flat",
    },
    "from_checkpoint": lambda p: extract_classes_from_path(p)[0],
    "model.model_transformers": lambda val: ", ".join(
        [
            {
                "copy_samples": "Copy Samples",
                "delete_memory": "Delete Memory",
                "conditional_to_deleted": "Delete Memory",
                "conditional_copy_best_fitting": "Copy Best Fitting from Conditional Memory",
            }[v]
            for v in val
        ]
    )
    if val
    else "Transfer Learning",
}
NICE_NAMING_LEVELS = {
    "epoch": "Epoch",
    "model.model_type": "Model Type",
    "seed": "Seed",
    "model.memory_size": "Memory Size",
    "model.shrink_threshold": "Shrink Threshold",
    "model.model_transformers": "Few-Shot Method",
    "from_checkpoint": "Previous Class",
    "model.learning_rate": "Learning Rate",
}
NICE_NAMING_CLASSES = {
    "mnist": {str(i): str(i) for i in range(10)},
    "cifar": {
        "0": "Plane",
        "1": "Car",
        "2": "Bird",
        "3": "Cat",
        "4": "Deer",
        "5": "Dog",
        "6": "Frog",
        "7": "Horse",
        "8": "Ship",
        "9": "Truck",
    },
}


def hydra_all_seeds_multirun_subdirs(directory):
    seeds = glob.glob(f"{directory}/seed=*")
    return [glob.glob(f"{dir}/*/run*") for dir in seeds]


def hydra_all_multirun_subdirs(directory):
    return glob.glob(f"{directory}/*/run*")


def hydra_all_run_subdirs(directory):
    return glob.glob(f"{directory}/run*")


def hydra_run_subdir(directory):
    dirs = glob.glob(f"{directory}/run*")
    assert len(dirs) == 1
    return dirs[0]


def all_subdirs(directory):
    return glob.glob(f"{directory}/*")


def parse_runs(
    directory, must_include_hyperparams=[], ignore_hyperparams=[], ignore_if_na=True, nice_names=True
):
    """
    Takes a directory or list of directories and parses all runs with their hyperparameters.
    Creates a dataframe with a multiindex corresponding to the hyperparameters which are different for the runs
    along with the path to the runs
    """
    if not isinstance(directory, list):
        directory = [directory]

    all_config_files = []
    for dir in directory:
        all_config_files.extend(glob.glob(f"{dir}/**/.hydra/config.yaml", recursive=True))

    configs = []
    for config_file in all_config_files:
        with open(config_file, "r") as f:
            configs.append(yaml.safe_load(f))

    # Double split the last part from the path, removing "/.hydra/config.yaml"
    # so we get the top level directory of that run
    paths = [os.path.split(os.path.split(path)[0])[0] for path in all_config_files]

    # Flatten the configs
    # Nested dictionaries are turned into a dictionary with just one level
    hyperparams_flat = pd.json_normalize(configs, sep=".")

    for k, mappings in NICE_NAMING_VALUES.items():
        if isinstance(mappings, dict):
            for to_replace, value in mappings.items():
                hyperparams_flat[k].replace(to_replace, value, inplace=True)
        else:
            hyperparams_flat[k] = hyperparams_flat[k].apply(mappings)

    hyperparams_flat = hyperparams_flat.applymap(lambda x: str(x) if isinstance(x, (list, tuple)) else x)

    hyperparams_flat.rename(columns=NICE_NAMING_LEVELS, inplace=True)

    if ignore_if_na:
        hyperparams_flat.dropna(inplace=True, axis="columns")
    hyperparams_flat.fillna("NA", inplace=True)

    unique = hyperparams_flat.nunique(dropna=False)
    unique_no_na = hyperparams_flat.nunique(dropna=True)
    # assert (unique[unique + 1 == unique_no_na] == 1).all()

    diff_hyperparams = hyperparams_flat.columns[unique > 1].to_list()

    # We might want some hyperparams to always be present even if there is no difference in values
    for hp in must_include_hyperparams:
        if hp not in diff_hyperparams:
            diff_hyperparams.append(hp)

    # There might be some hyperparams we want to ignore
    diff_hyperparams = list(filter(lambda x: x not in ignore_hyperparams, diff_hyperparams))

    # Only take the hyperparams we are interested in
    hyperparams_flat: pd.DataFrame = hyperparams_flat[diff_hyperparams]

    hyperparams_flat = hyperparams_flat.to_dict(orient="list")

    # Create a list of tuples of the hyperparameters as indexing values
    index_keys = [tuple(hyperparams_flat[k][i] for k in diff_hyperparams) for i, _ in enumerate(configs)]

    # Create index from the tuples
    multiindex = pd.MultiIndex.from_tuples(index_keys, names=diff_hyperparams)

    # Convert the values to a pandas dataframe
    df = pd.DataFrame(zip(paths, configs), index=multiindex, columns=["path", "config"])

    df.sort_index(inplace=True)
    return df


def load_scores_from_run(path, conditional, dataset, metric, must_be_same_length=True):
    """
    Load the auroc scores from tfevents files and combine to a dataframe for one run
    """
    # A run has multiple targets, get all
    file_list = glob.glob(f"{path}/**/*tfevents*", recursive=True)

    data = {}
    num_epochs = None
    for file_name in file_list:
        condition_name, multiple = extract_classes_from_path(file_name)

        # Load the tfevents file
        ea = event_accumulator.EventAccumulator(file_name)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if len(tags) == 0:
            print("Warning, no scalars found in TF events")
        for tag in tags:
            if tag.startswith(metric) and not tag.endswith("mean"):
                vals = np.array(list(map(lambda se: se.value, ea.Scalars(tag))))
                target = NICE_NAMING_CLASSES[dataset][tag.split("_")[-1]]
                name = f"{target}"
                if conditional:
                    name = f"{condition_name}\n- " + name
                data[name] = vals
                num_epochs = len(vals)
    if must_be_same_length:
        data = pd.DataFrame(data, index=range(1, num_epochs + 1))
    else:
        data = pd.DataFrame.from_dict(data, orient="index")
        data = data.transpose()

    data.columns.set_names("Class" + ("es - Condition" if conditional else ""), inplace=True)
    data.index.set_names("Epoch", inplace=True)
    return data


def load_scores_from_run_df(paths, metric, must_be_same_length=True):
    """
    Given a dataframe with a path column and a multiindex, loads the scores for every path and creates a new mutliindex dataframe
    """
    idxs = []
    dfs = []
    for idx, val in paths.iterrows():
        path = val["path"]
        config = val["config"]
        conditional = config["conditional"] is not None
        dataset = config["datamodule"]["name"]
        idxs.append(idx)
        dfs.append(load_scores_from_run(path, conditional, dataset, metric, must_be_same_length))
    df = pd.concat(dfs, keys=idxs, names=paths.index.names)
    return df


def load_model_change_from_run_df(paths, must_be_same_length=True):
    """ """

    idxs = []
    dfs = []
    for idx, val in tqdm(paths.iterrows(), total=len(paths)):
        path = val["path"]
        config = val["config"]
        conditional = config["conditional"] is not None
        idxs.append(idx)

        target_dirs = get_target_dirs(path)

        data = {}
        for target_dir in target_dirs:
            target_name = target_regex.search(target_dir).group(1)
            name_list, checkpoint_list = get_sorted_checkpoints(target_dir)
            # Load the memory from all the checkpoints
            mems = load_memories_from_checkpoints(MemaeSystem, checkpoint_list, False)
            # epochs, memory_size, feature_size = mems.shape
            # Align the memories such that we a[i] and b[i] contain two sequential epoch memories
            a = torch.tensor(mems[1:])
            b = torch.tensor(mems[:-1])
            # Then we take the cosine similarity between two epochs
            mems_similarity = torch.cosine_similarity(a, b, dim=2)
            # The values can be negative, add 1 and divide by two to adjust
            mems_similarity += 1
            mems_similarity /= 2
            # Take mean over all memory entries (over the memory size dim)
            mems_similarity = mems_similarity.mean(dim=1)
            # We interpret the memory difference as 1 - similarity
            data[target_name] = 1 - mems_similarity

            # The outermost array layer is converted to python list instead
            # data[target_name] = [mem for mem in mems]

        df = pd.DataFrame(data, index=name_list[1:])
        df.columns.set_names(("Condition - " if conditional else "") + "Class", inplace=True)
        df.index.set_names("Epoch", inplace=True)
        dfs.append(df)

    df = pd.concat(dfs, keys=idxs, names=paths.index.names)
    return df


def get_sorted_checkpoints(target_dir):
    checkpoint_files = glob.glob(f"{target_dir}/checkpoints/*.ckpt")
    p = re.compile(r"epoch=(\d+)")

    full_list = []
    extras = []
    for file in checkpoint_files:
        found = p.search(file)
        try:
            # Try to parse the epoch as a number, if it doesn't work
            # the name is something special, like "epoch=init"
            # we put those to the front
            epoch = int(found.group(1))
            full_list.append((epoch, file))
        except:
            extras.append((os.path.basename(file), file))

    # We must sort the list of integer named epochs as
    # e.g. epoch 9 comes after 20 lexicographically
    full_list = extras + sorted(full_list)
    name_list, file_list = zip(*full_list)
    return name_list, file_list


def load_memories_from_checkpoints(system_constructor, checkpoint_list, normalize=False):
    memories = []

    for checkpoint in checkpoint_list:
        system = system_constructor.load_from_checkpoint(checkpoint)
        system.eval()

        mems = []
        for w in system.model.mem_rep.memory.weight:
            mems.append(w.detach().numpy())
        memories.append(mems)

    memories = np.array(memories)
    if normalize:
        memories -= memories.min()
        memories = memories / memories.max()
    # output-shape: (epochs, memory_size, feature_dims)
    return memories


def show_normalized_img(img, save=False, filename=None):
    plt.figure()
    img = img[0].permute(1, 2, 0).detach().numpy()
    img = img - img.min()
    img = img / img.max()
    img = plt.imshow(img)
    plt.axis("off")
    if save:
        if not filename:
            filename = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')}{random.randint(0, 100000)}"

        plt.savefig(
            f"{filename}.pdf",
            bbox_inches="tight",
        )


def get_target_dirs(run_dir):
    return glob.glob(f"{run_dir}/*/target_*")


def plot_multi(
    df: pd.DataFrame,
    y_label,
    margin_top=0.05,
    margin_bottom=0.05,
    top=None,
    bottom=None,
    save=False,
    filename=None,
    legend_loc="lower right",
    figsize=(7, 5),
):
    """
    Takes a dataframe with multiindex and for each column creates a line plot:
    Each line plot contains a line for each level of the multiindex except the last one,
    it is used for the x axis
    """

    min_score = df.min().min() - margin_bottom
    max_score = df.max().max() + margin_top

    if top is not None:
        max_score = top
    if bottom is not None:
        min_score = bottom

    # Plot every condition/class over the epochs
    for col in df.columns:
        # Get data for every hyperparameter combination and plot it
        # groupby every level except the last one. Levels are (hyperparam1, hyperparam2, ..., hyperparamN, epochs)
        if df.index.nlevels > 1:
            grouped = df[col].groupby(level=(df.index.names[: df.index.nlevels - 1]))
        else:
            grouped = df[col]
        plt.figure(figsize)
        ax = plt.gca()
        ax.set_ylabel(y_label)
        ax.set_ylim([min_score, max_score])
        for (idx, data) in grouped:
            data.loc[idx].plot()

        ax.set_title(f"{df.columns.names[0]}: {col}")

        # Map the keys to lists, in case there is just one level per key, we can still use the .join function later
        list_of_keys = [
            keys if isinstance(keys, (list, tuple)) else [keys] for keys in grouped.indices.keys()
        ]

        list_of_keys = list(map(lambda keys: list(map(str, keys)), list_of_keys))

        ax.legend(
            list(map(" - ".join, list_of_keys)),
            title=" - ".join(df.index.names[:-1]),
            loc=legend_loc,
        )
        if save:
            if not filename:
                filename = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')}{random.randint(0, 100000)}"

            plt.savefig(
                f"{filename}{col}.pdf",
                bbox_inches="tight",
            )
        plt.show()


def plot_vector_as_bar(vec, y_label, x_label, save=False, filename=None):
    plt.bar(range(len(vec)), vec)
    ax = plt.gca()
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if save:
        if not filename:
            filename = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')}{random.randint(0, 100000)}"

        plt.savefig(
            f"{filename}.pdf",
            bbox_inches="tight",
        )


# Widgets
def select_checkpoint_from_run_df(run_df):
    checkpoints = []

    idx_level_names = run_df.index.names

    for idx, val in run_df.iterrows():
        path = val["path"]
        config = val["config"]
        conditional = config["conditional"] is not None

        idx_name = "; ".join(map(lambda x: f"{x[0]}: {x[1]}", zip(idx_level_names, idx)))

        target_dirs = get_target_dirs(path)

        data = {}
        for target_dir in target_dirs:
            target_name = target_regex.search(target_dir).group(1)
            name_checkpoint_list = zip(*get_sorted_checkpoints(target_dir))

            for name, ckpt in name_checkpoint_list:
                checkpoints.append((f"{idx_name}; Class: {target_name}; Epoch: {name}", ckpt))

    select_checkpoint = widgets.Dropdown(
        options=checkpoints,
        value=checkpoints[0][1],
        description="Checkpoint:",
        disabled=False,
    )
    display(select_checkpoint)
    return select_checkpoint


def select_target(run_dir):
    target_files = get_target_dirs(run_dir)
    targets = [(target_regex.search(target).group(1), target) for target in target_files]

    select_target = widgets.Dropdown(
        options=targets,
        value=targets[0][1],
        description="Target:",
        disabled=False,
    )
    display(select_target)
    return select_target


def select_checkpoint(target_dir):
    checkpoints = get_sorted_checkpoints(target_dir, with_names=True)

    select_checkpoint = widgets.Dropdown(
        options=checkpoints,
        value=checkpoints[0][1],
        description="Checkpoint:",
        disabled=False,
    )
    display(select_checkpoint)
    return select_checkpoint
