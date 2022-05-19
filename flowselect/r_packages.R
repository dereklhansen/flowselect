for (pkg in c("ggplot2", "dplyr", "purrr", "knockoff")) {
    if (!require(pkg, quietly=TRUE, character.only=TRUE)) install.packages(pkg)
}

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

if (!require("rhdf5", quietly=TRUE)) BiocManager::install("rhdf5")
