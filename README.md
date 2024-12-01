DL-NLP project: LMUformer 
========


# Document Classification

## Data preparation
* For setting up the dataset:
* setup kaggle profile locally
* run load_Data.ipynb to setup data

## LMUformer
run LMUformer_classifier.ipynb

## S4former
run S4_classifier.ipynb

## LMUformer
run Transformer_decoder_model.ipynb



# Image generation

This repository implements [VQVAE](https://arxiv.org/abs/1711.00937) for mnist and colored version of mnist and follows up with a simple LSTM for generating numbers.

## Data preparation
For setting up the dataset:

Verify the data directory has the following structure:
```
VQVAE-Pytorch/data/train/images/{0/1/.../9}
	*.png
VQVAE-Pytorch/data/test/images/{0/1/.../9}
	*.png
```

# Quickstart
* ```cd Image_generation``` change directory
* ```python -m tools.train_vqvae``` for training vqvae
* ```python -m tools.infer_vqvae``` for generating reconstructions and encoder outputs for LSTM training
* ```python -m tools.train_lstm``` for training minimal LSTM 
* ```python -m tools.generate_images``` for using the trained LSTM to generate some numbers

## Configurations
* ```config/vqvae_mnist.yaml``` - VQVAE for training on black and white mnist images
* ```config/vqvae_colored_mnist.yaml``` - VQVAE with more embedding vectors for training colored mnist images 

## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created and ```output_train_dir``` will be created inside it.

During training of VQVAE the following output will be saved 
* Best Model checkpoints(VQVAE and LSTM) in ```task_name``` directory

During inference the following output will be saved
* Reconstructions for sample of test set in ```task_name/output_train_dir/reconstruction.png``` 
* Encoder outputs on train set for LSTM training in ```task_name/output_train_dir/mnist_encodings.pkl```
* LSTM generation output in ```task_name/output_train_dir/generation_results.png```

## Evaluation

Evaluations is done using FID score.
run experiment.ipynb

## Sample Output for VQVAE

Running default config VQVAE for mnist should give you below reconstructions for both versions

Sample Generation Output after just 10 epochs
Training the vqvae and lstm longer and more parameters(codebook size, codebook dimension, channels , lstm hidden dimension e.t.c) will give better results 

## Citations
```
@misc{oord2018neural,
      title={Neural Discrete Representation Learning}, 
      author={Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
      year={2018},
      eprint={1711.00937},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


