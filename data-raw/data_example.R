library(readxl)

data_example <- read_excel("data-raw/data_final.xlsx")

usethis::use_data(data_example, overwrite = TRUE)
