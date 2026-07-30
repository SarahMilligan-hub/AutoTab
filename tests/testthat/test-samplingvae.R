# Unit Tests for sampling_VAE, only ran in github where thre is a tensorflow enviorment

make_small_vae_inputs <- function() {
  data_example <- data.frame(
    cont = c(1, 2, 3, 4, 5, 6, 7, 8),
    bin  = c(0, 1, 0, 1, 0, 1, 0, 1),
    cat  = factor(c("A", "B", "C", "A", "B", "C", "A", "B"))
  )

  feat_dist <- extracting_distribution(data_example)

  cont_scaled <- as.data.frame(lapply(data_example["cont"], min_max_scale))
  bin_col     <- data_example["bin"]
  cat_dummy   <- as.data.frame(model.matrix(~ cat - 1, data = data_example))

  preprocessed <- cbind(cont_scaled, bin_col, cat_dummy)
  feat_dist_reordered <- feat_reorder(feat_dist, preprocessed)

  list(data = preprocessed, feat_dist = feat_dist_reordered)
}

test_that("encoder_latent builds a model with z_mean and z_log_var of the correct latent_dim", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  encoder_input_data <- matrix(rnorm(4 * 3), nrow = 4, ncol = 3)

  model <- encoder_latent(
    encoder_input = encoder_input_data,
    encoder_info = list(list("dense", 5, "tanh")),
    latent_dim = 2,
    Lip_en = 0,
    power_iterations = 0
  )

  preds = predict(model, encoder_input_data)

  expect_length(preds, 2)
  expect_equal(dim(preds[[1]]), c(4, 2))
  expect_equal(dim(preds[[2]]), c(4, 2))
})

test_that("Encoder_weights extracts the correct number of weight tensors (no BN/SN)", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  encoder_input_data <- matrix(rnorm(4 * 3), nrow = 4, ncol = 3)

  model <- encoder_latent(
    encoder_input = encoder_input_data,
    encoder_info = list(list("dense", 5, "tanh")),
    latent_dim = 2,
    Lip_en = 0,
    power_iterations = 0
  )

  weights <- Encoder_weights(
    encoder_layers = 1,
    trained_model = model,
    lip_enc = 0,
    pi_enc = 0,
    BNenc_layers = 0,
    learn_BN = 0
  )

  # 1 hidden dense layer (2 tensors) + z_mean + z_log_var (2 tensors each) = 6
  expect_length(weights, 6)
  expect_equal(weights, keras::get_weights(model)[1:6])
})

test_that("Decoder_weights extracts only the decoder tensors from a combined VAE model", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  inputs <- make_small_vae_inputs()

  vae <- model_VAE(
    data = inputs$data,
    encoder_info = list(list("dense", 8, "tanh")),
    decoder_info = list(list("dense", 8, "tanh")),
    Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0,
    latent_dim = 2,
    feat_dist = inputs$feat_dist,
    lr = 0.001, beta = 0.1,
    max_std = 10.0, min_val = 1e-3,
    temperature = 1.0,
    weighted = 0, seperate = 0,
    prior = "single_gaussian"
  )

  all_weights <- keras::get_weights(vae)

  # encoder: 1 hidden dense layer + z_mean + z_log_var = 6 tensors (see test above)
  decoder_weights <- Decoder_weights(
    encoder_layers = 1,
    trained_model = vae,
    lip_enc = 0,
    pi_enc = 0,
    prior_learn = "fixed",
    BNenc_layers = 0,
    learn_BN = 0
  )

  expect_true(length(decoder_weights) > 0)
  expect_true(length(decoder_weights) < length(all_weights))
  expect_equal(decoder_weights, all_weights[7:length(all_weights)])
})
