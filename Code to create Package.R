#Code to create the R Package

install.packages(c("usethis","devtools"))

#Create a package project folder
#usethis::create_package("C:/Users/smill/OneDrive/Documents/Boston Univeristy/Dissertation/Code/Autoencoders/VAE from scratch/Final Programs/AutoTab")
#usethis::use_git_config(user.name = "Sarah Milligan", user.email = "slm.milligan@gmail.com")
#usethis::use_mit_license("Sarah Milligan")
usethis::use_readme_rmd()


#######################################################
#Creating Categories of where to save certain function#
#######################################################
#usethis::use_r("extracting_distribution")
#usethis::use_r("latent_sampling")
#usethis::use_r("priors")
#usethis::use_r("encoder")
#usethis::use_r("decoder")
#usethis::use_r("loss")
#usethis::use_r("model")
#usethis::use_r("training")

########################
#Adding Needed Packages#
########################
#usethis::use_package("R6")
#usethis::use_package("keras")
#usethis::use_package("tensorflow")
#usethis::use_package("reticulate")
#usethis::use_package("dplyr")


devtools::document()
