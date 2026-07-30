#' Install the AutoTab Conda Environment
#'
#' Creates the conda environment containing the exact Python/TensorFlow/Keras
#' stack that AutoTab requires, using the bundled environment specification.
#'
#' @param conda Path to conda binary, or "auto" to let reticulate locate it.
#' @export
install_autotab_env <- function(conda = "auto") {

  envname <- "r-autotab-env"

  if (reticulate::condaenv_exists(envname, conda = conda)) {
    message("Conda environment '", envname, "' already exists. ",
            "Use reticulate::conda_remove('", envname, "') first if you want to rebuild it.")
    return(invisible(envname))
  }

  yml_path <- system.file("conda", "autotab_environment.yml", package = "autotab")

  if (yml_path == "") {
    stop("Could not find bundled environment.yml. Is autotab installed correctly?")
  }

  message("Creating conda environment '", envname, "' from bundled specification...\n",
          "This will download TensorFlow, Keras, and related packages (~1GB). ",
          "This may take several minutes.")

  reticulate::conda_create(environment = yml_path, conda = conda)

  message("Environment created. Activate it in each new R session with:\n",
          "  reticulate::use_condaenv('", envname, "', required = TRUE)")

  invisible(envname)
}
