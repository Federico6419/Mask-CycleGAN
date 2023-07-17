# Mask CycleGAN

Mask CycleGAN is an improvment to the original CycleGAN, indeed it deals with an Image-To-Image Translation between two domains.
It uses a different architecture and a masking scheme to solve two main problems of CycleGAN, that are the unimodality in image translation and the lack of interpretability of latent variables.

## Installation

To execute the Main Notebook you have to open the file "Main-Notebook" on Google Colab and execute it: indeed the git clone command is already inserted in the code, and you have not to perform it. This contains a brief description of the Network, an example of a training of 10 epochs and the summary of the results we obtained.

Notice that both training and testing we performed were executed on the Notebook called "Mask-CycleGAN", indeed it could be used in different modalities according to the settings we put in the configuration file.
