###############################
#Getting Feature Distributions#
###############################
#' Extracting_Distribution will create feat_dist which will be used throughout the package.
#' There is one row per variable. Column one, column_name, indicates the names of the variables in the dataset.
#'  Column two, distribution, indicates the type of distribution the variable falls under.
#'  There are three options, Gaussian, Bernoulli, and categorical.
#'  Column three, num_params, indicates how many parameters the VAE should produce for the given variable.
#'
#' @param data Your original input data, prior to pre-processing
#' @return feat_dist which is used in AutoTab to define variable distribution types
#' @export
extracting_distribution = function(data){
  # Create a data set that the following information will fill in
  feat_dist = data.frame(
    column_name = colnames(data), # keep the same column name
    distribution = character(length(colnames(data))), # distribution type as a string
    num_params = integer(length(colnames(data)))  ) # number of parameters as an integer
  for (i in 1:ncol(data)) {
    variable = data[[i]] #This will loop through each column
    name = colnames(data)[i]
    if(is.numeric(variable) && length(unique(variable))>2){ #A numeric column with more than 2 distinct values
      feat_dist$distribution[i] = "gaussian"
      feat_dist$num_params[i] = 2     } #mean and SD
    else if (length(unique(variable))==2){ # character column with only 2 distinct values (binary)
      feat_dist$distribution[i] = "bernoulli"
      feat_dist$num_params[i] = 1     }#just the probability of 1
    else if ((is.character(variable) || is.factor(variable)) && length(unique(variable))>2){ #added option for it to be character or factor
      feat_dist$distribution[i] = "categorical"
      feat_dist$num_params[i] = length(unique(variable))  }}
  return(feat_dist)}


#Make sure order matches the data
#' If feat_dist is misordered, the function feat_reorder can be applied.
#' The user wants the rows of feat_order to match the order of the pre-processed dataset that will be used for training the VAE.
#' Feat_order ensures the row order in feat_dist matches the order in the input dataset.
#'
#' @param feat_dist,data feat_dist is the output of extracting_distribution. Data is your pre-processed data.
#' @return feat_dist that is in the same order as the pre-processed data going into the VAE
#' @export
feat_reorder = function(feat_dist,data){
  #Reorder to match the order in the data
  feat_names <- feat_dist$column_name  # These are the categorical feature names

  # For each dummy column, figure out which original variable it came from
  get_original_var <- function(colname, vars) {
    match <- vars[which(startsWith(colname, vars))]
    if (length(match) > 0) return(match[1]) else return(NA)}

  # Apply to all columns# Extract column names from the dummy-coded data
  dummy_cols <- colnames(data)
  dummy_to_original <- sapply(dummy_cols, get_original_var, vars = feat_names)
  # Get the order in which original variables appear (first time they show up in dummy_cols)
  ordered_original_vars <- unique(dummy_to_original)
  # Reorder feature_dist to match this order
  feat_dist <- feat_dist[match(ordered_original_vars, feat_dist$column_name), ]
  return(feat_dist)}

