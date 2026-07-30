#unit tests for min_max_scale

test_that("min_max_scale scales to [0, 1] correctly", {
  x <- c(10, 20, 30)
  result <- min_max_scale(x)
  expect_equal(result, c(0, 0.5, 1))
})

test_that("min_max_scale preserves vector length", {
  x <- c(5, 1, 100, 42, 7)
  result <- min_max_scale(x)
  expect_length(result, length(x))
})

test_that("min_max_scale returns 0 and 1 at the min and max positions", {
  x <- c(3, 8, -2, 15, 0)
  result <- min_max_scale(x)
  expect_equal(result[which.min(x)], 0)
  expect_equal(result[which.max(x)], 1)
})

test_that("min_max_scale works column-wise when applied via lapply to a data frame", {
  data <- data.frame(age = c(20, 40, 60), income = c(3000, 5000, 7000))
  scaled <- as.data.frame(lapply(data, min_max_scale))
  expect_equal(scaled$age, c(0, 0.5, 1))
  expect_equal(scaled$income, c(0, 0.5, 1))
})

test_that("min_max_scale returns NaN for a constant vector", {
  x <- c(5, 5, 5)
  result <- min_max_scale(x)
  expect_true(all(is.nan(result)))
})

test_that("min_max_scale handles a single-element vector", {
  x <- c(7)
  result <- min_max_scale(x)
  expect_true(is.nan(result))
})

test_that("min_max_scale handles negative values", {
  x <- c(-10, 0, 10)
  result <- min_max_scale(x)
  expect_equal(result, c(0, 0.5, 1))
})
