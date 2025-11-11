#######################
#Sampling latent space#
#######################
#' Sample from the latent space using the reparameterization trick
#'
#' @param z_mean,z_log_var The params of the latent space.
#' @return A sample from the latent space.
#' @export
Latent_sample = function(z_mean, z_log_var){
  tf = tensorflow::tf
  z_mean = tf$cast(z_mean, dtype = tf$float32)
  z_log_var = tf$cast(z_log_var, dtype = tf$float32)

  z_log_var_clamped = tf$clip_by_value(z_log_var, clip_value_min = -10.0, clip_value_max = 10.0)
  z_std = tf$maximum(tf$exp(0.5*z_log_var_clamped), 1e-3)
  e = tf$random$normal(shape = tf$shape(z_mean))
  return(z_mean + z_std*e )}
