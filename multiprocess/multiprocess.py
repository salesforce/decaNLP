import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


class Multiprocess():

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.world_size = args.world_size

        if os.path.isfile(args.dist_sync_file):
            os.remove(args.dist_sync_file)

    def run(self, runtime_args):
        self.start(runtime_args)
        self.join()

    def start(self, runtime_args):
        self.processes = []
        for rank in range(self.world_size):
            self.processes.append(Process(target=self.init_process, args=(rank, self.fn, self.args, runtime_args)))
            self.processes[-1].start()

    def init_process(self, rank, fn, args, runtime_args):
        torch.distributed.init_process_group(world_size=self.world_size, 
                                             init_method='file://'+args.dist_sync_file, 
                                             backend=args.backend, 
                                             rank=rank)
        fn(args, runtime_args, rank, self.world_size)
  
    def join(self):
        for p in self.processes:
            p.join()



