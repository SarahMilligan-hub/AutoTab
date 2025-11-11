
##############
#VAE Training#
##############
#' This will call your VAE training. Ensure you have your encoder_info and decoder_info.
#'
#' @return your training step
#' @export
VAE_train = function(data,encoder_info, decoder_info,Lip_en, pi_enc=1,lip_dec, pi_dec=1, latent_dim, epoch, beta,kl_warm=FALSE,kl_cyclical = FALSE, n_cycles, ratio, beta_epoch=15, temperature, temp_warm = FALSE,temp_epoch,batchsize, wait, min_delta, lr,max_std=10.0,min_val,weighted=0, recon_weights, seperate = 0,prior="single_gaussian",K =3,learnable_mog=FALSE,mog_means=NULL, mog_log_vars=NULL, mog_weights=NULL){
  EarlyStop = keras::callback_early_stopping(monitor='val_recon_loss', patience=wait, min_delta=min_delta,restore_best_weights = TRUE)

  # Setup beta (fixed or dynamic)
  if (kl_warm) {
    if (kl_cyclical){
      beta_dynamic = cyclical_beta_callback(beta_max = beta, total_epochs = epoch, n_cycles = n_cycles, ratio = ratio)
    } else {
      beta_dynamic = beta_callback(beta_max = beta, warmup_epochs = beta_epoch)
    }
    beta_used = beta_dynamic$beta_var
    beta_callback_list = list(beta_dynamic$callback)
  } else {
    beta_used = keras::k_variable(beta, dtype = "float32", name = "beta_fixed")
    beta_callback_list = list()
  }

  # Setup temperature (fixed or dynamic)
  if (temp_warm) {
    temp_dynamic = temperature_callback(temperature = temperature, warmup_epochs = temp_epoch)
    temp_used = temp_dynamic$temp_var
    temp_callback_list = list(temp_dynamic$callback)
  } else {
    temp_used = keras::k_variable(temperature, dtype = "float32", name = "temp_fixed")
    temp_callback_list = list()
  }

  run_vae = model_VAE(data=data, encoder_info=encoder_info, decoder_info=decoder_info,Lip_en=Lip_en, pi_enc=pi_enc,lip_dec=lip_dec, pi_dec=pi_dec, latent_dim=latent_dim, feat_dist=feat_dist, lr=lr , beta=beta_used,max_std=max_std, min_val=min_val,temperature=temp_used,weighted=weighted, recon_weights=recon_weights, seperate=seperate,prior=prior,K =K,learnable_mog=learnable_mog,mog_means=mog_means, mog_log_vars=mog_log_vars, mog_weights=mog_weights)

  #Tracking loss as we go
  loss_history <<-list()
  loss_tracked = keras::callback_lambda(on_epoch_end = function(epoch,logs){
    print(paste("Epoch", epoch+1, "Loss:", logs$loss, "Recon",logs$recon_loss, "KL_loss",logs$kl_loss  ))
    loss_history[[epoch+1]] <<- logs$loss
  })

  if (seperate == 1){callbacks = c(list(EarlyStop, loss_tracked),beta_callback_list,temp_callback_list,LossPrinterCallback$new())}
  else if (seperate == 0) { callbacks = c(list(EarlyStop, loss_tracked),beta_callback_list,temp_callback_list)}

  input_data <- as.matrix(data)

  run_vae %>% keras::fit(input_data, input_data, epochs = epoch, batch_size = batchsize, validation_split = 0.2, callbacks=callbacks)


  return(list(trained_model = run_vae, loss_history = loss_history))
}


# Function to reset all random seeds in R (and TensorFlow)
reset_seeds <- function(spec_seed) {
  tf = tensorflow::tf
  # Reset TensorFlow/Keras session and clear the graph
  tf$compat$v1$reset_default_graph()
  keras::k_clear_session()  # clears the Keras session
  # Set R random seed
  set.seed(spec_seed)  # seed value is an input option
  # Set TensorFlow random seed
  tf$random$set_seed(spec_seed)
  # Import and set Python's random seed (via reticulate)
  py_random <- reticulate::import("random")  # Import Python's random module
  py_random$seed(spec_seed)  # Set Python's random seed
  cat("Random seeds reset\n")
}
