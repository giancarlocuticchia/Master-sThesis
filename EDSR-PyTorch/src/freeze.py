# Function to freeze parameters in the model for Transfer Learning
"""
There are 5 named parameters we can freeze:

    ["sub_mean", "add_mean", "head", "body", "tail"]

The "body" contains 33 layers composed of:
    0 to 31: ResBlock layer (each composed of 2 Conv layers)
    32: Convolutional layer

The "tail" contains 2 layers composed of:
    0: Upsampler (composed of 2 Conv layers)
    1: Convolutional layer

Each layer has 2 parameters: weight and bias. Both are assumed to be frozen if
the layer is selected.

Example of input:
param_to_freeze = "sub_mean+add_mean+head+body+tail"
body_to_freeze = "1-10+21-30+32"        # valid from 0 to 32
tail_to_freeze = "0-1"                  # valid from 0 to 1

Which will be read as:
['sub_mean', 'add_mean', 'head', 'body', 'tail']
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32]
[0, 1]

For printing summary with torchinfo:
torchinfo_inputsize = "510,339"     # Default value

Means "width,height" of the test image to use to print the summary.
"""

def freeze_model(model, args):
    # Get the arguments from args
    param_to_freeze = args.param_to_freeze
    body_to_freeze = args.body_to_freeze
    tail_to_freeze = args.tail_to_freeze
    print_frozen_param = args.print_frozen_param    # Default is False
    torchinfo_summary = args.torchinfo_summary      # Default is False
    input_size = args.torchinfo_inputsize           # Default is "510,339" as the example test image
    
    # Parse the input parameters
    param_to_freeze = get_param_to_freeze_list(param_to_freeze)
    body_to_freeze = get_body_to_freeze_list(body_to_freeze)
    tail_to_freeze = get_tail_to_freeze_list(tail_to_freeze)
    input_size = get_input_size(input_size, batch_size=args.batch_size, n_colors=args.n_colors)   # Default is (16, 3, 510, 339) as the example test image

    # Iterate over the named parameters
    for name,value in model.named_parameters():
        name_split = name.split(".")
        # Freeze sub_mean
        if name_split[1] == "sub_mean" and "sub_mean" in param_to_freeze:
            value.requires_grad = False

        # Freeze add_mean
        if name_split[1] == "add_mean" and "add_mean" in param_to_freeze:
            value.requires_grad = False

        # Freeze head
        if name_split[1] == "head" and "head" in param_to_freeze:
            value.requires_grad = False

        # Freeze body
        if name_split[1] == "body" and "body" in param_to_freeze:
            if int(name_split[2]) in body_to_freeze:
                value.requires_grad = False
        
        # Freeze tail
        if name_split[1] == "tail" and "tail" in param_to_freeze:
            if int(name_split[2]) in tail_to_freeze:
                value.requires_grad = False

    if print_frozen_param :
        # Counting the frozen layers
        param_total = 0
        param_frozen = 0
        for name,value in model.named_parameters():
            #print(name, value.requires_grad)
            param_total+=1
            if value.requires_grad == False : param_frozen+=1

        print(f"Total named parameters: {param_total}\nFrozen named parameters: {param_frozen}")
    
    if torchinfo_summary :
        show_torchinfo_summary(model, input_size, idx_scale=0)

    return model

def get_param_to_freeze_list(param_to_freeze):
    # Function to parse the input for the param_to_freeze and return a list with the named parameters for the layers to freeze.
    elements_list = param_to_freeze.split("+")
    param_to_freeze_list = []

    if param_to_freeze != "":
      for element in elements_list:
          param_to_freeze_list.append(element)

    return param_to_freeze_list

def get_body_to_freeze_list(body_to_freeze):
    # Function to parse the input for the body_to_freeze and return a list with the number of the layers in the body to freeze.
    ranges_list = [ranges.split("-") for ranges in body_to_freeze.split("+")]
    body_to_freeze_list = []

    if body_to_freeze != "":
      for element in ranges_list:
          if len(element) > 1:
              for border in range(int(element[0]),int(element[1])+1):
                body_to_freeze_list.append(int(border))
          else:
              body_to_freeze_list.append(int(element[0]))

    return body_to_freeze_list

def get_tail_to_freeze_list(tail_to_freeze):
    # Function to parse the input for the tail_to_freeze and return a list with the number of the layers in the tail to freeze.
    elements_list = tail_to_freeze.split("-")
    tail_to_freeze_list = []

    if tail_to_freeze != "":
      for element in elements_list:
          tail_to_freeze_list.append(int(element))

    return tail_to_freeze_list

def show_torchinfo_summary(model, input_size=(16, 3, 510, 339), idx_scale=0):
    # Checking for torchinfo
    print("Importing torchinfo\n")
    try:
        from torchinfo import summary
    except:
        import subprocess
        print("Installing torchinfo...")
        subprocess.call(["pip", "install", "-q", "torchinfo"])
        from torchinfo import summary
    print("torchinfo successfully imported.\n")

    # Print a summary using torchinfo
    print("Printing summary of the model with torchinfo...")
    print(summary(model=model,
            input_size=input_size,    # make sure this is "input_size", not "input_shape"
            idx_scale=idx_scale,                    # it was asking for this parameter, otherwise it would arise an error
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    ))

    return

def get_input_size(input_size, batch_size=16, n_colors=3):
    # Get input image dimensions "width,height"
    width,height = input_size.split(",")
    width,height = int(width),int(height)
    # Get input image number of color channels
    n_colors = int(n_colors)
    return (batch_size, n_colors, width, height)