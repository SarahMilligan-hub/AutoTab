#' @importFrom magrittr %>%
#' @importFrom R6 R6Class

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  options(AutoTab.startup_shown = FALSE)
}

#' @keywords internal
.onAttach <- function(libname, pkgname) {
  if (!interactive() || Sys.getenv("NOT_CRAN") != "true") return(invisible())

  if (isTRUE(getOption("AutoTab.startup_shown"))) return(invisible())
  options(AutoTab.startup_shown = TRUE)

  packageStartupMessage(
    "AutoTab loaded successfully!\n",
    "------------------------------------------------------------\n",
    "Before using AutoTab, ensure your Python environment is active:\n",
    " Install the needed enviorment with install_autotab_env()\n",
    "Then activate the enviorment with \n",
    "reticulate::use_condaenv('r-autotab-env', required = TRUE)\n",
    "------------------------------------------------------------"
  )

  if (!reticulate::py_available(initialize = FALSE)) {
    packageStartupMessage("No active Python environment detected.")
    return(invisible())
  }

}
