import ray.tune as tune


def resolve_config(ray_config: dict):
    ray_params: dict = ray_config.pop("ray")

    return {**ray_config, **ray_resolve(ray_params)}


def ray_resolve(ray_config: dict):
    params = {}
    for key, value in ray_config.items():
        if type(value) == list:
            print(value)
            if len(value) == 2:
                params[key] = tune.uniform(*value)
            else:
                params[key] = tune.grid_search(value)
        elif type(value) == dict:
            params[key] = ray_resolve(value)
        else:
            params[key] = value
    return params


def keep_tunes(config: dict):
    tunes = {}

    for key, value in config.items():
        if type(value) == dict:
            sub_val = value.get("grid_search", None)

            if sub_val is not None:
                tunes[key] = sub_val

    return tunes
