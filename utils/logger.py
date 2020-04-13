from collections import OrderedDict
import os
import numpy as np
import torch
###########################################################

class SimpleLogger():
    verbosity_lvl = ['debug', 'runtime']
    key_val_fmt = "{}: {}\n"
    str_fmt = "{}\n"

    def __init__(self,filename,verbosity="debug"):
        self.handle = open(filename, "w+")
        assert self.handle
        assert verbosity in self.verbosity_lvl
        self.verbosity = verbosity
        self.drawer=None
        
    def close(self):
        self.handle.close()

    def info(self, content, auto_flush=True, auto_endline=False):
        if isinstance(content, OrderedDict):
            self.log_ordered_dict(content, auto_flush=auto_flush, auto_endline=auto_endline)
        elif isinstance(content, str):
            self._write_str(content, auto_flush=auto_flush, auto_endline=auto_endline )
        else:
            self.log_iterable(content, auto_flush=auto_flush, auto_endline=auto_endline)


    def _write_str(self, str, auto_flush=True, auto_endline=False):
        self.handle.write(str)
        if auto_endline:
            self.handle.write("\n")
        if auto_flush:
            self.handle.flush()
        if self.verbosity == "debug":
            print(str)

    def log_ordered_dict(self, ordered_dict, auto_flush=True, auto_endline=False):
        for k, v in ordered_dict.items():
            self._write_str( self.key_val_fmt.format(k, v), auto_flush=auto_flush, auto_endline=auto_endline )

    def log_iterable(self, iterable, auto_flush=True, auto_endline=False):
        for i in iterable:
            self._write_str( self.str_fmt.format(i), auto_flush=auto_flush, auto_endline=auto_endline )

    def flush(self):
        self.handle.flush()

    def add_drawer(self, draw_path):
        self.drawer = DrawerInLogger(draw_path=draw_path)

    def get_drawer(self):
        if self.drawer is None:
            print("Warning: no drawer available")
        return self.drawer

###########################################################

## average meter
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

