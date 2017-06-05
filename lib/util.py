# -*- coding: utf-8 -*-
import os
import pickle


def dump_to_pickle(path, name, data):
    with open(os.path.join(path, '{}.pickle'.format(name)), 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(path, name):
    with open(os.path.join(path, '{}.pickle'.format(name)), 'rb') as f:
        return pickle.load(f)
