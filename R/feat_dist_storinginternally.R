###############################################
# Internal cache for feature distribution     #
###############################################

# Private environment for storing feat_dist
.AutoTab_cache <- new.env(parent = emptyenv())

#' Set the feature distribution for AutoTab
#'
#' This function stores the output of `extracting_distribution()` /
#' `feat_reorder()` inside the package, so subsequent functions (e.g.,
#' `VAE_train()`) can access it safely without relying on the global
#' environment.
#'
#' @param feat_dist A data.frame returned by `extracting_distribution()`
#'   or `feat_reorder()`.
#'
#' @return No output is printed to the console. `feat_dist` is stored
#'   internally so other AutoTab functions (e.g., `VAE_train()`) can
#'   access it.
#'
#' @examples
#' feat_dist <- data.frame(
#'   column_name = c("age", "smoker"),
#'   distribution = c("gaussian", "bernoulli"),
#'   num_params = c(2, 1)
#' )
#' set_feat_dist(feat_dist)
#'
#'
#' @export
set_feat_dist <- function(feat_dist) {
  stopifnot(
    is.data.frame(feat_dist),
    all(c("column_name", "distribution", "num_params") %in% names(feat_dist))
  )
  assign("feat_dist", feat_dist, envir = .AutoTab_cache)
  invisible(TRUE)
}

#' Get the stored feature distribution
#' @keywords internal
get_feat_dist <- function() {
  if (!exists("feat_dist", envir = .AutoTab_cache, inherits = FALSE)) {
    stop("feat_dist not set. Run set_feat_dist(feat_dist) first.")
  }
  get("feat_dist", envir = .AutoTab_cache, inherits = FALSE)
}
