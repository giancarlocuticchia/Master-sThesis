## Content of this folder

We are providing the following files:

* config-test.txt
* config-train.txt
* EDSRx4-modelnamedparameters.txt
* log-test.txt
* log-train.txt
* printmodel-EDSRx4t2-full.txt
* torchinfo-EDSRx4t2-depth5-full.txt

The explanations for those files are shown below.


## config-test.txt and log-test.txt

If we run the main.py script on a terminal with a command like the following:

```bash
python main.py --data_test Demo --scale 4 --save edsr_x4_test --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train "../pre-train/edsr_x4-4f62e9ef.pt" --test_only --save_results
```

We would expect to be created a folder "edsr_x4_test" inside of a folder "experiment" inside the "EDSR-PyTorch" directory. Inside of "edsr_x4_test" we would have a config.txt file and a log.txt file similar to [config-test.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/config-test.txt) and [log-test.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/log-test.txt) respectively.


## config-train.txt and log-train.txt

If we run the main.py script on a terminal with a command like the following:

```bash
python main.py --template EDSR_custom --save_models --chop
```

where EDSR_custom is a template defined in [template.py](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/EDSR-PyTorch/src/template.py) like:

```python 
    if args.template.find('EDSR_custom') >= 0:
        args.dir_data =  "../image-data"
        args.data_train = "Custom"
        args.data_test = "Custom"
        args.data_range = "1-2400/2401-2500"
        args.ext = "sep"
        args.scale = "4"
        args.model = "EDSR"
        args.pre_train = "../pre-train/edsr_x4-4f62e9ef.pt"
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.test_every = 100
        args.epochs = 11
        args.batch_size = 16
        args.save = "edsr_x4_train"
```

We would expect to be created a folder "edsr_x4_train" inside of a folder "experiment" inside the "EDSR-PyTorch" directory. Inside of "edsr_x4_test" we would have a config.txt file and a log.txt file similar to [config-train.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/config-train.txt) and [log-train.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/log-train.txt) respectively.


## printmodel-EDSRx4t2-full.txt

If we load the model the EDSR x4 model like the following:

```python
import torch

import utility
import model

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

# Loading the model
model = model.Model(args, checkpoint)
```

where we prepared the arguments "args" like explained in section "1.1 Setting up the args" of our Jupyter Notebook [0_The_EDSRx4_model](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Notebooks-scripts/Notebooks/0_The_EDSRx4_model.ipynb).

If we do a "print(model)", we would expect a print like the one showed in the file [printmodel-EDSRx4t2-full.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/printmodel-EDSRx4t2-full.txt).

## EDSRx4-modelnamedparameters.txt

As shown in section "2.2 Model parameters" of our Jupyter Notebook [2_Freezing_layers](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Notebooks-scripts/Notebooks/2_Freezing_layers.ipynb), if we load the EDSR x4 model (as shown above), if we do:

```python
for name,value in model.named_parameters():
  print(name)
```

we then we would expect to be printed something like shown in file [EDSRx4-modelnamedparameters.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/EDSRx4-modelnamedparameters.txt).

## torchinfo-EDSRx4t2-depth5-full.txt

As shown in section "2.4 Visualizing it with torchinfo" of our Jupyter Notebook [2_Freezing_layers](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Notebooks-scripts/Notebooks/2_Freezing_layers.ipynb), we can see the model structure and if a layer of the model is trainable or not, by using the "summary" function of the torchinfo library.

If we load the model as shown above, and we use the summary function of the torchinfo library like:

```python
from torchinfo import summary

batch_size = 16
image_dim = (3, 500, 500)

summary(model=model,
        input_size=(batch_size, *image_dim),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        cache_forward_pass=True,
        depth=5,
        mode="train",
        row_settings=["var_names"],
        idx_scale=0
)
```
we would expect a print of the model structure like the one shown in file [torchinfo-EDSRx4t2-depth5-full.txt](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Output-files/torchinfo-EDSRx4t2-depth5-full.txt).