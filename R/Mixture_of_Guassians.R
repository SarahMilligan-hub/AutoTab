#' Mixture-of-Gaussians (MoG) prior in AutoTab
#'
#' AutoTab allows the encoder prior to be either a single Gaussian
#' (`prior = "single_gaussian"`) or a mixture of Gaussians
#' (`prior = "mixture_gaussian"`). When using a MoG prior, the user may
#' optionally specify the component means, variances, and mixture weights.
#' The user may also indicate if the means, variances, and mixture weights
#' can be learned or not using learnable_mog with a logical TRUE/FALSE.
#'
#' @section Prior options in \code{VAE_train()}:
#' \itemize{
#'   \item \code{prior}: character, one of \code{"single_gaussian"} or
#'     \code{"mixture_gaussian"}.
#'   \item \code{K}: integer, number of mixture components when
#'     \code{prior = "mixture_gaussian"}.
#'   \item \code{learnable_mog}: logical; if \code{TRUE}, the MoG parameters
#'     (means, log-variances, and mixture weights) are learned during training.
#'   \item \code{mog_means}: optional numeric matrix of size
#'     \code{K x latent_dim}, giving the initial means for each mixture
#'     component in the latent space.
#'   \item \code{mog_log_vars}: optional numeric matrix of size
#'     \code{K x latent_dim}, giving initial log-variances for each component.
#'   \item \code{mog_weights}: optional numeric vector of length \code{K},
#'     giving initial mixture weights that should sum to 1.
#' }
#'
#' @details
#' If \code{prior = "single_gaussian"}, the prior is a standard Normal in the
#' latent space and the MoG-related arguments (\code{K}, \code{mog_means},
#' \code{mog_log_vars}, \code{mog_weights}, \code{learnable_mog}) are ignored.
#'
#' When \code{prior = "mixture_gaussian"}:
#' \itemize{
#'   \item If \code{learnable_mog = FALSE}, then \code{mog_means},
#'     \code{mog_log_vars}, and \code{mog_weights} \strong{must} be supplied
#'     and are treated as fixed.
#'   \item If \code{learnable_mog = TRUE}, any of \code{mog_means},
#'     \code{mog_log_vars}, or \code{mog_weights} that are provided are used
#'     as initial values and are updated during training. If they are omitted,
#'     AutoTab initializes them internally (e.g., Normal or zero-centered
#'     initializations).
#' }
#'
#' @section Shape of \code{mog_means}:
#' For a latent dimension \code{latent_dim} and \code{K} mixture components,
#' \code{mog_means} must be a numeric matrix with:
#' \itemize{
#'   \item \code{nrow(mog_means) == K}
#'   \item \code{ncol(mog_means) == latent_dim}
#' }
#' Each row corresponds to the mean vector of one mixture component in the
#' latent space.
#'
#' @examples
#' \dontrun{
#' #Example of a Mixture of Gaussian prior with learnable_mog = FALSE:
#'
#' library(AutoTab)
#' library(dplyr)
#' library(keras)
#' library(caret)
#' #Testing my example in the readme documentation
#' set.seed(123)
#' age        <- rnorm(100, mean = 45, sd = 12)
#' income     <- rnorm(100, mean = 60000, sd = 15000)
#' bmi        <- rnorm(100, mean = 25, sd = 4)
#' smoker     <- rbinom(100, 1, 0.25)
#' exercise   <- rbinom(100, 1, 0.6)
#' diabetic   <- rbinom(100, 1, 0.15)
#' education  <- sample(
#'   c("HighSchool", "College", "Graduate"),
#'   100,
#'   replace = TRUE,
#'   prob = c(0.4, 0.4, 0.2)
#' )
#' marital    <- sample(c("Single", "Married", "Divorced"), 100, replace = TRUE)
#' occupation <- sample(c("Clerical", "Technical", "Professional", "Other"), 100, replace = TRUE)
#' data_final <- data.frame(
#'  age, income, bmi,
#'   smoker, exercise, diabetic,
#'    education, marital, occupation
#'    )
#'    encoded_data = dummyVars(~ education + marital +occupation, data = data_final)
#'    one_hot_coded = as.data.frame(predict(encoded_data, newdata = data_final))
#'    data_cont = subset(data_final, select = c(age, income,bmi ))
#' Continuous_MinMaxScaled <- as.data.frame(
#'   lapply(data_cont, min_max_scale)
#' )
#' # min_max_scale is a function in AutoTab
#'    data_bin = subset(data_final, select = c(smoker, exercise, diabetic ))
#'    #Bind all data together
#'    data = cbind(Continuous_MinMaxScaled, data_bin, one_hot_coded)
#'
#'    # Step 1: Extract and set feature distributions
#'
#'    feat_dist = feat_reorder(extracting_distribution(data_final),data)
#'    rownames(feat_dist) = NULL
#'    set_feat_dist(feat_dist)
#'
#'    # Step 2: Define encoder / decoder architectures with MOG parameters
#'
#'    encoder_info <- list(
#'     list("dense", 25, "relu"),
#'     list("dense", 50, "relu")
#'     )
#'     decoder_info <- list(
#'      list("dense", 50, "relu"),
#'       list("dense", 25, "relu")
#'       )
#'
#'       mog_means = matrix(c(rep(-5, 5), rep(0, 5), rep(5, 5)), nrow = 3, byrow = TRUE)
#'       mog_log_vars = matrix(log(0.5), nrow = 3, ncol = 5)
#'       mog_weights = c(0.3, 0.4, 0.3)
#'
#'    #Step 3: Run AutoTab
#'
#'       reset_seeds(1234)
#'       training <- VAE_train(
#'        data = data,
#'         encoder_info = encoder_info,
#'          decoder_info = decoder_info,
#'          Lip_en = 0,      # spectral normalization off
#'          pi_enc = 0,
#'          lip_dec = 0,
#'          pi_dec = 0,
#'          latent_dim = 5,
#'          epoch = 20,
#'          beta = 0.01,     # β-VAE regularization weight
#'          kl_warm = TRUE,
#'          beta_epoch = 20, # warm-up epochs
#'          temperature = 0.5,
#'          batchsize = 16,
#'          wait = 20,
#'          lr = 0.001,
#'          K=3,
#'          mog_means = mog_means,
#'          mog_log_vars = mog_log_vars,
#'           mog_weights = mog_weights,
#'            prior = "mixture_gaussian",
#'            learnable_mog = FALSE
#'            )
#'
#' # Step 4: Extract encoder and decoder for sampling
#'            weights_encoder <- Encoder_weights(
#'              encoder_layers = 2,
#'               trained_model = training$trained_model,
#'                lip_enc = 0,
#'                 pi_enc = 0,
#'                  BNenc_layers = 0,
#'                  learn_BN = 0
#'                  )
#'                  latent_encoder <- encoder_latent(
#'                    encoder_input = data,
#'                      encoder_info = encoder_info,
#'                        latent_dim = 5,
#'                          Lip_en = 0,
#'                           power_iterations=0
#'                           )
#' latent_encoder %>% keras::set_weights(weights_encoder)
#' input_data <- as.matrix(data)
#' latent_space <- predict(latent_encoder, as.matrix(input_data))
#'
#' # Rebuild and apply decoder
#'
#' weights_decoder <- Decoder_weights(
#' encoder_layers = 2,
#' trained_model = training$trained_model,
#' lip_enc = 0,
#'  pi_enc = 0,
#'  prior_learn = "fixed",
#'  BNenc_layers = 0,
#'  learn_BN = 0
#'  )
#'
#'  decoder <- decoder_model(
#'   decoder_input = NULL,
#'   decoder_info = decoder_info,
#'    latent_dim = 5,
#'    feat_dist = feat_dist,
#'    lip_dec = 0,
#'    pi_dec = 0
#'    )
#'
#'    decoder %>% keras::set_weights(weights_decoder)
#'
#'    # Sample from latent space
#'    z_mean <- latent_space[[1]]
#'    z_log_var <- latent_space[[2]]
#'    sample_latent <- Latent_sample(z_mean, z_log_var)
#'    decoder_sample <- predict(decoder, as.matrix(sample_latent))
#'    decoder_sample <- as.data.frame(decoder_sample)
#'
#' }
#'
#' \dontrun{
#'  #Example of a Mixture of Gaussian prior with learnable_mog = TRUE
#'   with preset means, variances, and weights:
#'
#'    # Step 2: Define encoder / decoder architectures with MOG parameters
#'
#'    encoder_info <- list(
#'     list("dense", 25, "relu"),
#'     list("dense", 50, "relu")
#'     )
#'     decoder_info <- list(
#'      list("dense", 50, "relu"),
#'       list("dense", 25, "relu")
#'       )
#'
#'       mog_means = matrix(c(rep(-5, 5), rep(0, 5), rep(5, 5)), nrow = 3, byrow = TRUE)
#'       mog_log_vars = matrix(log(0.5), nrow = 3, ncol = 5)
#'       mog_weights = c(0.3, 0.4, 0.3)
#'
#' # Step 3: Run AutoTab
#'
#'       reset_seeds(1234)
#'       training <- VAE_train(
#'        data = data,
#'         encoder_info = encoder_info,
#'          decoder_info = decoder_info,
#'          Lip_en = 0,      # spectral normalization off
#'          pi_enc = 0,
#'          lip_dec = 0,
#'          pi_dec = 0,
#'          latent_dim = 5,
#'          epoch = 20,
#'          beta = 0.01,     # β-VAE regularization weight
#'          kl_warm = TRUE,
#'          beta_epoch = 20, # warm-up epochs
#'          temperature = 0.5,
#'          batchsize = 16,
#'          wait = 20,
#'          lr = 0.001,
#'          K=3,
#'          mog_means = mog_means,
#'          mog_log_vars = mog_log_vars,
#'           mog_weights = mog_weights,
#'            prior = "mixture_gaussian",
#'            learnable_mog = TRUE
#'            )
#' # Step 4: Extract encoder and decoder for sampling
#'            weights_encoder <- Encoder_weights(
#'              encoder_layers = 2,
#'               trained_model = training$trained_model,
#'                lip_enc = 0,
#'                 pi_enc = 0,
#'                  BNenc_layers = 0,
#'                  learn_BN = 0
#'                  )
#'                  latent_encoder <- encoder_latent(
#'                    encoder_input = data,
#'                      encoder_info = encoder_info,
#'                        latent_dim = 5,
#'                          Lip_en = 0,
#'                           power_iterations=0
#'                           )
#' latent_encoder %>% keras::set_weights(weights_encoder)
#' input_data <- as.matrix(data)
#' latent_space <- predict(latent_encoder, as.matrix(input_data))
#' # Rebuild and apply decoder
#'
#' weights_decoder <- Decoder_weights(
#' encoder_layers = 2,
#' trained_model = training$trained_model,
#' lip_enc = 0,
#'  pi_enc = 0,
#'  prior_learn = "learned",
#'  BNenc_layers = 0,
#'  learn_BN = 0
#'  )
#'
#'  decoder <- decoder_model(
#'   decoder_input = NULL,
#'   decoder_info = decoder_info,
#'    latent_dim = 5,
#'    feat_dist = feat_dist,
#'    lip_dec = 0,
#'    pi_dec = 0
#'    )
#'
#'    decoder %>% keras::set_weights(weights_decoder)
#'
#'    # Sample from latent space
#'    z_mean <- latent_space[[1]]
#'    z_log_var <- latent_space[[2]]
#'    sample_latent <- Latent_sample(z_mean, z_log_var)
#'    decoder_sample <- predict(decoder, as.matrix(sample_latent))
#'    decoder_sample <- as.data.frame(decoder_sample)
#'
#' }
#'
#'
#' \dontrun{
#' #Example of a Mixture of Gaussian prior with learnable_mog = TRUE
#' without preset means, variances, and weights:
#'
#'    # Step 2: Define encoder / decoder architectures with MOG parameters
#'
#'    encoder_info <- list(
#'     list("dense", 25, "relu"),
#'     list("dense", 50, "relu")
#'     )
#'     decoder_info <- list(
#'      list("dense", 50, "relu"),
#'       list("dense", 25, "relu")
#'       )
#'
#'  # Step 3: Run AutoTab
#'
#'       reset_seeds(1234)
#'       training <- VAE_train(
#'        data = data,
#'         encoder_info = encoder_info,
#'          decoder_info = decoder_info,
#'          Lip_en = 0,      # spectral normalization off
#'          pi_enc = 0,
#'          lip_dec = 0,
#'          pi_dec = 0,
#'          latent_dim = 5,
#'          epoch = 20,
#'          beta = 0.01,     # β-VAE regularization weight
#'          kl_warm = TRUE,
#'          beta_epoch = 20, # warm-up epochs
#'          temperature = 0.5,
#'          batchsize = 16,
#'          wait = 20,
#'          lr = 0.001,
#'          K=3,
#'          mog_means = NULL,
#'          mog_log_vars = NULL,
#'           mog_weights = NULL,
#'            prior = "mixture_gaussian",
#'            learnable_mog = TRUE
#'            )
#'
#' # Step 4: Extract encoder and decoder for sampling
#'            weights_encoder <- Encoder_weights(
#'              encoder_layers = 2,
#'               trained_model = training$trained_model,
#'                lip_enc = 0,
#'                 pi_enc = 0,
#'                  BNenc_layers = 0,
#'                  learn_BN = 0
#'                  )
#'                  latent_encoder <- encoder_latent(
#'                    encoder_input = data,
#'                      encoder_info = encoder_info,
#'                        latent_dim = 5,
#'                          Lip_en = 0,
#'                           power_iterations=0
#'                           )
#' latent_encoder %>% keras::set_weights(weights_encoder)
#' input_data <- as.matrix(data)
#' latent_space <- predict(latent_encoder, as.matrix(input_data))
#'
#' # Rebuild and apply decoder
#'
#' weights_decoder <- Decoder_weights(
#' encoder_layers = 2,
#' trained_model = training$trained_model,
#' lip_enc = 0,
#'  pi_enc = 0,
#'  prior_learn = "learned",
#'  BNenc_layers = 0,
#'  learn_BN = 0
#'  )
#'
#'  decoder <- decoder_model(
#'   decoder_input = NULL,
#'   decoder_info = decoder_info,
#'    latent_dim = 5,
#'    feat_dist = feat_dist,
#'    lip_dec = 0,
#'    pi_dec = 0
#'    )
#'
#'    decoder %>% keras::set_weights(weights_decoder)
#'
#'    # Sample from latent space
#'    z_mean <- latent_space[[1]]
#'    z_log_var <- latent_space[[2]]
#'    sample_latent <- Latent_sample(z_mean, z_log_var)
#'    decoder_sample <- predict(decoder, as.matrix(sample_latent))
#'    decoder_sample <- as.data.frame(decoder_sample)
#' }
#'
#' @seealso [VAE_train()]
#' @name mog_prior
NULL
