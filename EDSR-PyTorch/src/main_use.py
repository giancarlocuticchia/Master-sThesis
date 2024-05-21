import os
import torch
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import imageio
import torch
from tqdm import tqdm

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)

def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None: break
            imageio.imwrite(filename, tensor.numpy())

class checkpoint_minimal():
    def __init__(self, args):
        self.args = args
        self.dir = args.dir_demo
        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)

        self.n_processes = 8
    
    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)
    
    def save_results(self, dataset, filename, sr_image, scale):
        if self.args.save_results:
            export_folder = self.get_path("results")
            if not os.path.exists(export_folder):
                os.makedirs(export_folder, exist_ok=True)
            
            filename = os.path.join(export_folder,
                                    '{}_x{}'.format(filename, scale)
            )

            normalized = sr_image.mul(255 / self.args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            self.queue.put(('{}.png'.format(filename), tensor_cpu))
    
    def begin_background(self):
        self.queue = Queue()

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

checkpoint = checkpoint_minimal(args)

class MainUse():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = my_model

    def main_use(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    lr = lr.squeeze()
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], sr, scale)
        
        if self.args.save_results: self.ckp.end_background()

        torch.set_grad_enabled(True)
    
    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

def main():
    global model
    print(f"Placing images inside {args.dir_demo} into dataloader...")
    loader = data.Data(args)
    print("Loading model...")
    _model = model.Model(args, checkpoint)
    print("Processing images...")
    u = MainUse(args, loader, _model, checkpoint)
    u.main_use()
    print("All images were successfully processed.")

if __name__ == '__main__':
    main()