# Unit tests for decoder_model, only run in github environments with a working TensorFlow install

test_that("decoder_model builds a model with the correct output dimension for mixed feat_dist", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  feat_dist <- data.frame(
    column_name  = c("cont", "bin", "cat"),
    distribution = c("gaussian", "bernoulli", "categorical"),
    num_params   = c(2, 1, 3)
  )

  latent_dim <- 2

  decoder <- decoder_model(
    decoder_input = NULL,
    decoder_info  = list(list("dense", 8, "tanh")),
    latent_dim    = latent_dim,
    feat_dist     = feat_dist,
    lip_dec       = 0,
    pi_dec        = 0
  )

  z <- matrix(rnorm(4 * latent_dim), nrow = 4, ncol = latent_dim)
  output <- predict(decoder, z)

  # total output dim = sum(num_params) for features = 2 + 1 + 3 = 6
  expect_equal(dim(output), c(4, 6))
})

test_that("decoder_model output values respect distribution-specific constraints", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  feat_dist <- data.frame(
    column_name  = c("bin"),
    distribution = c("bernoulli"),
    num_params   = c(1)
  )

  latent_dim <- 2

  decoder <- decoder_model(
    decoder_input = NULL,
    decoder_info  = list(list("dense", 8, "tanh")),
    latent_dim    = latent_dim,
    feat_dist     = feat_dist,
    lip_dec       = 0,
    pi_dec        = 0
  )

  z <- matrix(rnorm(5 * latent_dim), nrow = 5, ncol = latent_dim)
  output <- predict(decoder, z)

  # bernoulli head passes through sigmoid, so output must lie in (0, 1)
  expect_true(all(output > 0 & output < 1))
})

test_that("decoder_model respects max_std/min_val bounds on a gaussian-only feat_dist", {
  skip_if_not(reticulate::py_module_available("tensorflow"))

  feat_dist <- data.frame(
    column_name  = c("cont"),
    distribution = c("gaussian"),
    num_params   = c(2)
  )

  latent_dim <- 2
  max_std <- 3.0
  min_val <- 0.5

  decoder <- decoder_model(
    decoder_input = NULL,
    decoder_info  = list(list("dense", 8, "tanh")),
    latent_dim    = latent_dim,
    feat_dist     = feat_dist,
    lip_dec       = 0,
    pi_dec        = 0,
    max_std       = max_std,
    min_val       = min_val
  )

  z <- matrix(rnorm(5 * latent_dim), nrow = 5, ncol = latent_dim)
  output <- predict(decoder, z)

  sd_column <- output[, 2]
  expect_true(all(sd_column >= min_val & sd_column <= max_std))
})
