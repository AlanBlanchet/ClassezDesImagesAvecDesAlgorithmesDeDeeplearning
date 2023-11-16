def return_if_task_bb(obj, task: str):
    if is_task_bb(task):
        return obj

    obj_type = type(obj)
    if obj_type == dict:
        return {}
    elif obj_type == list:
        return []
    return None


def chose_if_task_bb(obj1, obj2, task: str):
    return obj1 if is_task_bb(task) else obj2


def is_task_bb(task: str):
    return task == "detection"
