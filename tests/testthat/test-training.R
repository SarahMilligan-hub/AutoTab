#Unit tests for training, not ran ocally, ran in the github cloud space that will have tensorflow

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

test_that("VAE_train errors if feat_dist has not been set", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  inputs <- make_small_vae_inputs()

  expect_error(
    VAE_train(
      data = inputs$data,
      encoder_info = list(list("dense", 8, "tanh")),
      decoder_info = list(list("dense", 8, "tanh")),
      Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0,
      latent_dim = 2, epoch = 1, beta = 0.1,
      temperature = 1.0, batchsize = 4, wait = 50, lr = 0.001
    ),
    "feat_dist"
  )
})

test_that("VAE_train errors when weighted = 1 but recon_weights is missing", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  inputs <- make_small_vae_inputs()
  set_feat_dist(inputs$feat_dist)

  expect_error(
    VAE_train(
      data = inputs$data,
      encoder_info = list(list("dense", 8, "tanh")),
      decoder_info = list(list("dense", 8, "tanh")),
      Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0,
      latent_dim = 2, epoch = 1, beta = 0.1,
      temperature = 1.0, batchsize = 4, wait = 50, lr = 0.001,
      weighted = 1
    ),
    "recon_weights"
  )
})

test_that("VAE_train errors when kl_cyclical = TRUE but kl_warm = FALSE", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  inputs <- make_small_vae_inputs()
  set_feat_dist(inputs$feat_dist)

  expect_error(
    VAE_train(
      data = inputs$data,
      encoder_info = list(list("dense", 8, "tanh")),
      decoder_info = list(list("dense", 8, "tanh")),
      Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0,
      latent_dim = 2, epoch = 1, beta = 0.1,
      kl_warm = FALSE, kl_cyclical = TRUE,
      temperature = 1.0, batchsize = 4, wait = 50, lr = 0.001
    ),
    "kl_cyclical"
  )
})

test_that("VAE_train runs to completion on a tiny dataset and returns expected structure", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  inputs <- make_small_vae_inputs()
  set_feat_dist(inputs$feat_dist)

  result <- VAE_train(
    data = inputs$data,
    encoder_info = list(list("dense", 8, "tanh")),
    decoder_info = list(list("dense", 8, "tanh")),
    Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0,
    latent_dim = 2, epoch = 2, beta = 0.1,
    temperature = 1.0, batchsize = 4, wait = 50, lr = 0.001
  )

  expect_type(result, "list")
  expect_named(result, c("trained_model", "loss_history"))
  expect_s3_class(result$trained_model, "keras.engine.training.Model")
  expect_true(length(result$loss_history) >= 1)
  expect_true(length(result$loss_history) <= 2)
})

test_that("reset_seeds runs without error and prints a confirmation message", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  expect_message(reset_seeds(2026), "Random seeds reset")
})
