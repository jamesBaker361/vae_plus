
# vae_plus

This project is based off of [Liu et al 2017](https://arxiv.org/abs/1703.00848), and [Elgammal et al 2017](https://arxiv.org/abs/1706.07068). The goal is to translate images from one domain (such as real human face photos) to another domain (such as cartoon faces). This is done in three steps:
1. Train a classifier to categorize images as belonging to one of the domains; I found fine-tuning a pretrained classifier to be most effective.  Use the `yvae/yvae_classification_loop.py` for this
2. Train a variational autoencoder to reconstruct images from all domains with an additional creative loss: the classifier should not be able to distinguish which domain the reconstructed images are from. Use `yvae/yvae_creativity_loop.py` for this
3. Train a set of autoencoders that use the pretrained encoder layers from step 2. Some of the layers in the encoder will be shared. Each autoencoder will correspond to a different domain, and have its own decoder. Use `yvae/yvae_unit_loop.py` for this.

Command line args are described in `yvae/yvae_parser_setup.py`

The rationale behind this is that the latent space for the images in all domains should be shared, and ideally, pretraining to minimize creative loss should encourage the encoder layers to generalize as much as possible.