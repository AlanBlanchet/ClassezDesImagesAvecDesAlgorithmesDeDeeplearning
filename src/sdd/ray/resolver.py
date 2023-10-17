import ray.tune as tune

RAY_CHOICES = ["uniform", "loguniform", "grid_search"]


def resolve_config(ray_config: dict):
    ray_params: dict = ray_config.pop("ray")

    return {**ray_config, **ray_resolve(ray_params), "ray": True}


def ray_resolve(ray_config: dict):
    params = {}
    for key, value in ray_config.items():
        if type(value) == dict:
            if all([x in value.keys() for x in ["type", "value"]]):
                t = value["type"]
                v = value["value"]

                if t == "uniform":
                    params[key] = tune.uniform(*v)
                elif t == "loguniform":
                    params[key] = tune.loguniform(*v)
                elif t == "grid_search":
                    params[key] = tune.grid_search(v)
                elif t == "choice":
                    params[key] = tune.choice(v)
                else:
                    raise ValueError(
                        f"Type must be one of [{','.join(RAY_CHOICES)}], received {t}"
                    )
            else:
                params[key] = ray_resolve(value)
        else:
            raise ValueError("Value should be a dict with 'type' and 'value' keys")
    return params


def keep_tunes(config: dict):
    valids = [dict, list, int, float, str, bool]
    tunes = {}

    for key, value in config.items():
        if type(value) not in valids:
            tunes[key] = value

    return tunes
