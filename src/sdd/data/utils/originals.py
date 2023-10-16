def keep_originals(output: dict, input: dict, to_keep: list[str]):
    for k in input.keys():
        if k in to_keep:
            output["original_" + k] = input[k].clone()
        output[k] = input[k]
    return output
