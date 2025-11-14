## Test environments
* Local Windows 11, R 4.4.3
* R-hub (GitHub Actions):
  - windows-latest
  - ubuntu-release
* win-builder (R-devel)

## R CMD check results
0 errors | 0 warnings | 1 note

## Additional information
* The single NOTE from win-builder concerns "possibly misspelled words" in the DESCRIPTION 
  (e.g., VAE, Keras, Autoencoders), which are standard technical terms.
* The package uses TensorFlow (version 2.10) and Keras via the `reticulate` interface.
* Examples and tests that require TensorFlow are conditionally skipped when TensorFlow is not available, ensuring CRAN checks complete successfully on systems without Python/TensorFlow installed.
* All examples have been tested locally using the installed package in a TensorFlow-enabled environment to confirm correct behavior when the backend is available.
* This approach follows standard CRAN practice for optional external dependencies while enabling full VAE functionality for users who configure a TensorFlow environment.
