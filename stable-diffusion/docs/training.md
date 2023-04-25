# Stable Diffusion Training Documentation

This document provides guidance on how to use the training feature of the Stable Diffusion repository. The training feature is executed using the `main.py` file and accepts various command-line arguments to configure the training process.

## Prerequisites

Before you begin training, ensure that you have the following:

1. Python 3.7 or later installed.
2. Required packages installed (PyTorch, PyTorch Lightning, OmegaConf, etc.).
3. A configuration file (in YAML format) containing the model, data, and training parameters.

## Configuration File

To use the training feature, you need to provide a configuration file (in YAML format) that specifies the model, data, and various training parameters. A sample config file structure is shown below:

```yaml
model:
  type: diffusion_model
  ...
  
data:
  type: diffusion_dataset
  ...

training:
  gpus: 1
  max_epochs: 100
  ...

logging:
  ...
```

Make sure to replace the ellipses (`...`) with appropriate values for your use case.

## Training Process

To train a model using the Stable Diffusion repository, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory containing the `main.py` file.
3. Execute the following command to start the training process:

```bash
python main.py --base <config.yaml> --name <name> --train
```

Replace `<config.yaml>` with the path to your configuration file and `<name>` with the desired name for the training run.

### Additional Command-line Arguments

You can customize the training process by providing additional command-line arguments. Some of the most common arguments are:

- `--resume`: Resume training from a checkpoint.
- `-d/--debug`: Run in debug mode.
- `--no_test`: Skip testing.
- `--scale_lr`: Scale learning rate based on batch size.
- `--postfix`: Append a postfix to the log directory.
- `--seed`: Set the random seed.
- `--logdir`: Set the log directory.

For a complete list of arguments, refer to the `main.py` documentation or execute the command `python main.py -h`.

## Monitoring Training

During the training process, the script logs various metrics and visualizations (e.g., images). By default, these logs are saved in the `logs/` directory, although you can specify a different directory using the `--logdir` argument.

To visualize training progress, you can use a tool like TensorBoard. To do this, execute the following command in a separate terminal:

```bash
tensorboard --logdir logs/
```

Then, open a web browser and navigate to the URL provided by TensorBoard.

This document should provide a basic understanding of how to use the training feature in the Stable Diffusion repository. If you have any further questions or need more information, please feel free to ask.