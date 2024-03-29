Author: Marius Tresoldi, 2024

The following project was carried out in the context of the capstone project during a degree of advanced study in data science 
at ETH Zurich.

The core ideas are based on the following paper:

https://arxiv.org/abs/1710.11469

The paper introduces conditonal variance regularization for domain shift robust learning in computer vision.
The paper primarily applies the regularization on the predicted logits, called conditional variance of prediction (CVP). 
In this project, the method is modified by directly applying the regularization to the output of the convolutional layers, 
called conditional variance of representation (CVR).

The script "mnist_cvr.py" applies CVP and CVR on the MNIST data set and compares the performance of the two methods.

The script "celeba_cvr.py" applies CVR on the CelebA data set and shows that CVR allows to learn domain shift robust 
representations, which when transferred yield domain shift robust predictors without any need for regularization.

The script "train_utily.py" contains various jax based utilities that implement CVP/CVR.

For details see the comments in the respective scripts.

In order to run the code install the packages in the requirements.txt file.
Further, ImageMagick needs to be installed in the environment and callable from the 
command line in order to create CelebA images with degraded quality:

https://www.imagemagick.org/
