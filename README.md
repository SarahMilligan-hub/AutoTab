
# AutoTab

AutoTab is a flexible R package for training Variational Autoencoders
(VAEs) on heterogeneous tabular data. It supports continuous, binary,
and categorical variables within a single model and includes a custom
loss function and decoder architecture that automatically handles each
distribution type.

In the decoder, the final activation layer slices the output tensor by
distribution, preserving a single amortized decoder with shared hidden
layers that jointly map the latent space to all observed variables. The
reconstruction loss is computed as the sum of distribution-specific
likelihoods, consistent with the VAE evidence lower bound (ELBO).

AutoTab extends beyond the standard (vanilla) VAE framework and allows
users to integrate several well-known extensions from the VAE
literature. Refer to the function-level R documentation for all
available options.

## Installation

When loading the package via library(AutoTab), AutoTab will remind you
to activate your reticulate/conda environment before running any model
functions. AutoTab requires TensorFlow 2.10.0 within the active Python
environment.

AutoTab was developed under Python 3.10.8 and numpy 1.26.4. If
compatibility issues arise, recreate your environment with this version.

``` r
# Install from GitHub
remotes::install_github("SarahMilligan-hub/AutoTab")

# Load the package
library(AutoTab)
 
```

## Example

Below is a basic example of running a VAE training iteration. The
example is not a VAE that has been tuned. It is merely an example to
show how each function works. The user will execute hyperparameter
tuning of their VAE and input dataset using the AutoTab package.
Hyperparameters (the options within the AutoTab package) are not a one
size fits all scenario.

For an example of using a Mixture of Gaussian Prior run ?mog_prior

Note: When running AutoTab the user will receive the following warning
from tensorflow:

WARNING:tensorflow:The following Variables were used in a Lambda layer’s
call (tf.math.multiply_3), but are not present in its tracked objects:
\<tf.Variable ‘beta:0’ shape=() dtype=float32\>. This is a strong
indication that the Lambda layer should be rewritten as a subclassed
Layer.

This is merely a warning and should not effect the computation of
AutoTab. This occurs because tensorflow does not see beta, (the weight
on the regularization part of the ELBO) until after the first iteration
of training and the first computation of the loss is initiated.
Therefore it is not an internally tracked object. However, it is being
tracked and updated outside of the model graph which can be seen in the
KL loss plots and in the training printout in the R console.

``` r
#Before executing the example Initiate your Conda / Reticulate Environment 

library(AutoTab)
library(dplyr)
library(keras)
library(caret)

#Testing my example in the readme documentation
set.seed(123)
age        <- rnorm(100, mean = 45, sd = 12)
income     <- rnorm(100, mean = 60000, sd = 15000)
bmi        <- rnorm(100, mean = 25, sd = 4)
smoker     <- rbinom(100, 1, 0.25)
exercise   <- rbinom(100, 1, 0.6)
diabetic   <- rbinom(100, 1, 0.15)
education  <- sample(c("HighSchool", "College", "Graduate"), 100, replace = TRUE, prob = c(0.4, 0.4, 0.2))
marital    <- sample(c("Single", "Married", "Divorced"), 100, replace = TRUE)
occupation <- sample(c("Clerical", "Technical", "Professional", "Other"), 100, replace = TRUE)
data_final <- data.frame(
  age, income, bmi,
  smoker, exercise, diabetic,
  education, marital, occupation
)

encoded_data = dummyVars(~ education + marital +occupation, data = data_final)
one_hot_coded = as.data.frame(predict(encoded_data, newdata = data_final))
data_cont = subset(data_final, select = c(age, income,bmi ))
Continuous_MinMaxScaled = as.data.frame(lapply(data_cont, min_max_scale)) #min_max_scale is a function in AutoTab
data_bin = subset(data_final, select = c(smoker, exercise, diabetic ))
#Bind all data together 
data = cbind(Continuous_MinMaxScaled, data_bin, one_hot_coded)

# Step 1: Extract and set feature distributions
feat_dist = feat_reorder(extracting_distribution(data_final),data)
rownames(feat_dist) = NULL
set_feat_dist(feat_dist)

# Step 2: Define encoder and decoder architectures

encoder_info <- list(
  list("dense", 25, "relu"),
  list("dense", 50, "relu")
)

decoder_info <- list(
  list("dense", 50, "relu"),
  list("dense", 25, "relu")
)

reset_seeds(1234)
training <- VAE_train(
  data = data,
  encoder_info = encoder_info,
  decoder_info = decoder_info,
  Lip_en = 0,      # spectral normalization off
  pi_enc = 0,
  lip_dec = 0,
  pi_dec = 0,
  latent_dim = 5,
  epoch = 200,
  beta = 0.01,     # β-VAE regularization weight
  kl_warm = TRUE,
  beta_epoch = 20, # warm-up epochs
  temperature = 0.5,
  batchsize = 16,
  wait = 20,
  lr = 0.001, 
)

# Step 4: Extract encoder and decoder for sampling

weights_encoder <- Encoder_weights(
  encoder_layers = 2,
  trained_model = training$trained_model,
  lip_enc = 0,
  pi_enc = 0,
  BNenc_layers = 0,
  learn_BN = 0
)

latent_encoder <- encoder_latent(
  encoder_input = data,
  encoder_info = encoder_info,
  latent_dim = 5,
  Lip_en = 0,
  power_iterations=0
)

latent_encoder %>% keras::set_weights(weights_encoder)
input_data <- as.matrix(data)
latent_space <- predict(latent_encoder, as.matrix(input_data))

# Rebuild and apply decoder

weights_decoder <- Decoder_weights(
  encoder_layers = 2,
  trained_model = training$trained_model,
  lip_enc = 0,
  pi_enc = 0,
  prior_learn = "fixed",
  BNenc_layers = 0,
  learn_BN = 0
)

decoder <- decoder_model(
  decoder_input = NULL,
  decoder_info = decoder_info,
  latent_dim = 5,
  feat_dist = feat_dist,
  lip_dec = 0,
  pi_dec = 0
)

decoder %>% keras::set_weights(weights_decoder)

# Sample from latent space

z_mean <- latent_space[[1]]
z_log_var <- latent_space[[2]]
sample_latent <- Latent_sample(z_mean, z_log_var)

decoder_sample <- predict(decoder, as.matrix(sample_latent))
decoder_sample <- as.data.frame(decoder_sample)
}
```

# Citing AutoTab

If you use AutoTab in your research, please cite:

Milligan, S. (2025). AutoTab: Variational Autoencoder for Heterogeneous
Tabular Data. GitHub. <https://github.com/SarahMilligan-hub/AutoTab>
