"""
    Module:
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/23
    Introduce: Used to store and load models

    Update time:
        2020/12/28: TODO: update the introduce of functions
"""
__all__ = ["save_model", "load_model"]

import joblib


def save_model(model, path):
    """
    # TODO: save the model
    :param model:
        the net of all kinds of models
    :param path:
        the path of file (including file name)
    :return:
        return None
    """
    joblib.dump(model, filename=path)


def load_model(path):
    """
    # TODO: load the model
    :param path:
        the path of file (including file name)
    :return:
        return the model(object which can be called directly)
    """
    return joblib.load(path)
