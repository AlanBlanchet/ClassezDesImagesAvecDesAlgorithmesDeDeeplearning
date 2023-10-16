import copy
import itertools

import torch


def update_list_by_dict(target: list, value: dict):
    """
    Updates a list using keys from dictionary
    May be helpful if you want a fuzzy update of a list
    If key greater than list size, list will be extended
    If list value is dict, it's deep updated
    """
    for k, v in value.items():
        idx = int(k)
        if idx >= len(target):
            target.extend(itertools.repeat(None, idx - len(target) + 1))
        if isinstance(target[idx], dict) and isinstance(v, dict):
            deepupdate(target[idx], v)
        else:
            target[idx] = v


def deepupdate(target: dict, *addons: dict):
    """Deep update target dict with update
    For each k,v in update: if k doesn't exist in target, it is deep copied from
    update to target.
    Nested lists are extend if you update by the list and updated if you update by dictionary
    """
    addon = addons[0]
    if len(addons) > 1:
        addon = deepupdate(addon, *addons[1:])

    for k, v in addon.items():
        curr_val = None
        if isinstance(target, dict):
            if k not in target:
                target[k] = copy.deepcopy(v)

            curr_val = target[k]

        if isinstance(curr_val, list):
            if isinstance(v, list):
                target[k] = copy.deepcopy(v)
                # curr_val.extend(copy.deepcopy(v))
            elif isinstance(v, dict):
                update_list_by_dict(curr_val, v)
            else:
                curr_val.extend(v)
        elif isinstance(curr_val, dict):
            deepupdate(target[k], v)
        elif isinstance(curr_val, set):
            if not k in target:
                target[k] = v.copy()
            else:
                target[k].update(v.copy())
        else:
            target[k] = copy.copy(v)
    return target


def deep_dict_parse_tensor(d: dict):
    return {k: deep_dict_parse_propagate(v) for k, v in d.items()}


def deep_dict_parse_propagate(value):
    if type(value) == torch.Tensor:
        return value.item()
    elif type(value) == dict:
        return deep_dict_parse_tensor(value)
