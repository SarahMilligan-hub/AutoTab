#UNit tests for extracting distribution

test_that("extracting_distribution classifies gaussian, bernoulli, and categorical correctly", {
  data_example <- data.frame(
    cont = c(1.5, 2.7, 3.1, 8.4, 5.5),
    bin  = c(0, 1, 0, 1, 1),
    cat  = factor(c("A", "B", "C", "A", "C"))
  )

  feat_dist <- extracting_distribution(data_example)

  expect_equal(feat_dist$column_name, c("cont", "bin", "cat"))
  expect_equal(feat_dist$distribution, c("gaussian", "bernoulli", "categorical"))
  expect_equal(feat_dist$num_params, c(2, 1, 3))
})

test_that("extracting_distribution returns correct data frame structure", {
  data_example <- data.frame(x = rnorm(5))
  feat_dist <- extracting_distribution(data_example)

  expect_s3_class(feat_dist, "data.frame")
  expect_named(feat_dist, c("column_name", "distribution", "num_params"))
  expect_equal(nrow(feat_dist), 1)
})

test_that("extracting_distribution classifies a numeric 2-level column as bernoulli", {
  data_example <- data.frame(x = c(2, 7, 2, 7, 2))
  feat_dist <- extracting_distribution(data_example)

  expect_equal(feat_dist$distribution, "bernoulli")
  expect_equal(feat_dist$num_params, 1)
})

test_that("extracting_distribution classifies a character 2-level column as bernoulli", {
  data_example <- data.frame(x = c("yes", "no", "yes", "no"))
  feat_dist <- extracting_distribution(data_example)

  expect_equal(feat_dist$distribution, "bernoulli")
  expect_equal(feat_dist$num_params, 1)
})

test_that("extracting_distribution flags columns with missing values", {
  data_example <- data.frame(x = c(1, 2, NA, 4))

  expect_message(
    feat_dist <- extracting_distribution(data_example),
    "missing values"
  )
  expect_equal(feat_dist$distribution, "Missing data - cannot use column")
  expect_equal(feat_dist$num_params, 0)
})

test_that("extracting_distribution counts categorical levels correctly", {
  data_example <- data.frame(
    cat = factor(c("A", "B", "C", "D", "A", "B"))
  )
  feat_dist <- extracting_distribution(data_example)

  expect_equal(feat_dist$distribution, "categorical")
  expect_equal(feat_dist$num_params, 4)
})

test_that("feat_reorder reorders feat_dist rows to match reordered data columns", {
  data_example <- data.frame(
    cont = rnorm(5),
    bin  = c(0, 1, 0, 1, 1),
    cat  = factor(c("A", "B", "C", "A", "C"))
  )
  feat_dist <- extracting_distribution(data_example)
  data_reordered <- data_example[, c("cat", "cont", "bin")]

  feat_dist_reordered <- feat_reorder(feat_dist, data_reordered)

  expect_equal(feat_dist_reordered$column_name, c("cat", "cont", "bin"))
})

test_that("feat_reorder preserves all original rows after reordering", {
  data_example <- data.frame(
    cont = rnorm(5),
    bin  = c(0, 1, 0, 1, 1),
    cat  = factor(c("A", "B", "C", "A", "C"))
  )
  feat_dist <- extracting_distribution(data_example)
  data_reordered <- data_example[, c("bin", "cat", "cont")]

  feat_dist_reordered <- feat_reorder(feat_dist, data_reordered)

  expect_equal(nrow(feat_dist_reordered), nrow(feat_dist))
  expect_setequal(feat_dist_reordered$column_name, feat_dist$column_name)
})

test_that("feat_reorder matches dummy-coded columns back to their original variable", {
  feat_dist <- data.frame(
    column_name = c("cont", "cat"),
    distribution = c("gaussian", "categorical"),
    num_params = c(2, 3)
  )
  dummy_data <- data.frame(cat_A = c(1, 0), cat_B = c(0, 1), cont = c(1.2, 3.4))

  feat_dist_reordered <- feat_reorder(feat_dist, dummy_data)

  expect_equal(feat_dist_reordered$column_name, c("cat", "cont"))
})
