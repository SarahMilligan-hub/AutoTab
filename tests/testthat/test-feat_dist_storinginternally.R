#unit tests for setting feat_dist

test_that("set_feat_dist accepts a valid feat_dist data frame", {
  feat_dist <- data.frame(
    column_name = c("x", "y"),
    distribution = c("gaussian", "bernoulli"),
    num_params = c(2, 1)
  )
  expect_true(set_feat_dist(feat_dist))
})

test_that("set_feat_dist errors when input is not a data frame", {
  expect_error(set_feat_dist(list(column_name = "x")))
})

test_that("set_feat_dist errors when a required column is missing", {
  bad_feat_dist <- data.frame(
    column_name = c("x", "y"),
    distribution = c("gaussian", "bernoulli")
  )
  expect_error(set_feat_dist(bad_feat_dist))
})

test_that("set_feat_dist stores the object so it can be retrieved with get_feat_dist", {
  feat_dist <- data.frame(
    column_name = c("a", "b", "c"),
    distribution = c("gaussian", "bernoulli", "categorical"),
    num_params = c(2, 1, 3)
  )
  set_feat_dist(feat_dist)

  stored <- autotab:::get_feat_dist()
  expect_equal(stored, feat_dist)
})

test_that("get_feat_dist errors if nothing has been set yet", {
  # clear the cache environment directly to simulate a fresh session
  if (exists("feat_dist", envir = autotab:::.AutoTab_cache, inherits = FALSE)) {
    rm("feat_dist", envir = autotab:::.AutoTab_cache)
  }
  expect_error(autotab:::get_feat_dist(), "feat_dist not set")
})
