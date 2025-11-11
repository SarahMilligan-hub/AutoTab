
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

Below is a basic example of running a VAE training iteration.

``` r
library(AutoTab)

#The data used in this example is simulated data that contains 1 contiuous, 1 binary, and one categorical variable. 
set.seed(123)
cont <- rnorm(100, mean = 50, sd = 10)
bin <- rbinom(100, 1, 0.4)
cat <- sample(c("A", "B", "C"), size = 100, replace = TRUE)
data_final = data.frame(  continuous = cont,  binary = bin,  category = factor(cat))

#Before using AutoTab, like in many machine learning contexts, the data must be pre-processed. The recommended pre processing for continuous variables in AutoTab is min-max scaling. 
library(caret)
encoded_data = dummyVars(~ category, data = data_final)
one_hot_coded = as.data.frame(predict(encoded_data, newdata = data_final))
Continuous_MinMaxScaled = as.data.frame(lapply(cont, min_max_scale)) #min_max_scale is a function in AutoTab
#Bind all data together 
data = cbind(Continuous_MinMaxScaled, bin, one_hot_coded)

# Step 1: Extract and set feature distributions
feat_dist = feat_reorder(extracting_distribution(data_final),data)
rownames(feat_dist) = NULL
set_feat_dist(feat_dist)

# Step 2: Define encoder and decoder architectures

encoder_info <- list(
list("dense", 100, "relu"),
list("dense", 80, "relu")
)

decoder_info <- list(
list("dense", 80, "relu"),
list("dense", 100, "relu")
)

# Step 3: Train the VAE
\dontrun{
training <- VAE_train(
data = data,
encoder_info = encoder_info,
decoder_info = decoder_info,
Lip_en = 0,      # spectral normalization off
pi_enc = 0,
lip_dec = 0,
pi_dec = 0,
latent_dim = 5,
epoch = 100,
beta = 0.01,     # β-VAE regularization weight
kl_warm = TRUE,
beta_epoch = 20, # warm-up epochs
temperature = 0.5,
batchsize = 64,
wait = 20,
lr = 1e-3
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
Lip_en = 0
)

latent_encoder %>% keras::set_weights(weights_encoder)
input_data <- as.matrix(data)
latent_space <- keras::predict(latent_encoder, as.matrix(input_data))

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

decoder_sample <- keras::predict(decoder, as.matrix(sample_latent))
decoder_sample <- as.data.frame(decoder_sample)
}
```

# Citing AutoTab

If you use AutoTab in your research, please cite:

Milligan, S. (2025). AutoTab: Variational Autoencoder for Heterogeneous
Tabular Data. Boston University, Department of Biostatistics.

\#License MIT © 2025 Sarah Milligan
