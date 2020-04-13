import os
import time
from datetime import datetime
import json
from collections import OrderedDict

## json for config
def read_json(fname):
    with open(fname, 'rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def parse_config(config_path):
    return read_json(config_path)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ConfigParser():
    def __init__(self, config_path, logger=None):
        assert config_path is not None, "Error: config path is not provided" 
        print (config_path)      
        self._config = read_json(config_path)
        self._ConfigIsNoneError()

        exp_name = os.path.splitext(os.path.basename(config_path))[0]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._config["ckpt_folder"] = os.path.join(self._config["saved"], exp_name, timestamp, "ckpt")
        self._config["log_folder"] = os.path.join(self._config["saved"], exp_name, timestamp, "log")
        self._config["result_folder"] = os.path.join(self._config["saved"], exp_name, timestamp, "result")

        board_name = exp_name+timestamp
        self._config["board_name"] = os.path.join(self._config["board"], board_name)
        print(self._config["board_name"])

        ensure_dir(self._config["ckpt_folder"])
        ensure_dir(self._config["log_folder"])
        ensure_dir(self._config["result_folder"])
        

    def get_config_parameters(self):
        self._ConfigIsNoneError()
        return self._config

    def update_config(self, config):
        self._config = config
    
    def get_logger(self):
        return self._logger

    def set_logger(self, logger):
        self._logger = logger
    
    def get_content(self, key, default_value=None):
        return self._config.get(key, default_value)

    def set_content(self, key, value):
        assert isinstance(key, str), "Error: Key type must be str"
        self._config[key] = value

    def set_deeper_content(self, keys, value):
        raise NotImplementedError

    def reset(self):
        self._config = None

    ## as-private functions
    def _ConfigIsNoneError(self):
        assert self._config is not None, "Error: config is None"