library(tidyverse)
library(rhdf5)
library(stringi)


DATASETS = c(
    "gaussian_ar_mixture_linear_d100_n100000_rho0.98_",
    "gaussian_ar_mixture_sin_cos5_d100_n100000_rho0.98_",
    "rnasq_cor_linear_a40_n10_nv100_nvc5_",
    "rnasq_cor_sin_cos5_a40_n10_nv100_nvc5_"
)

read_asymptotic_dataset <- function(dataset_name) {
    fdrs <- h5read(dataset_name %s+% "_asymptotic_fdr_and_power.h5", "fdrs")
    powers <- h5read(dataset_name %s+% "_asymptotic_fdr_and_power.h5", "powers")

    fdr_tbl <- plyr::adply(fdrs, c(1), .id = "fdr_control_level") %>%
        tibble() %>%
        group_by(fdr_control_level) %>%
        mutate(K=1:n()) %>%
        ungroup %>%
        gather(-fdr_control_level, -K, key="run_id", value="fdr")
    
    power_tbl <- plyr::adply(powers, c(1), .id = "fdr_control_level") %>%
        tibble() %>%
        group_by(fdr_control_level) %>%
        mutate(K=1:n()) %>%
        ungroup %>%
        gather(-fdr_control_level, -K, key="run_id", value="power")

    tbl <- inner_join(fdr_tbl, power_tbl, by=c("fdr_control_level", "K", "run_id"))
    levels(tbl$fdr_control_level) <- c("0.05", "0.10", "0.25")
    tbl <- mutate(tbl, fdr_control_level = as.double(as.character(fdr_control_level)))

    return(tbl)
}

asy_mean_tbl <- map(DATASETS, read_asymptotic_dataset) %>%
    set_names(DATASETS) %>%
    bind_rows(.id="dataset") %>%
    gather(fdr, power, key="metric", value="value") %>%
    mutate(metric = factor(metric, c("fdr", "power"), c("FDR", "Power"))) %>%
    group_by(dataset, fdr_control_level, K, metric) %>%
    summarize(sd_value = sd(value), value=mean(value)) %>%
    arrange(dataset, fdr_control_level, metric, K)

data_types <- c("gaussian", "rnasq")

asy_mean_tbls <- map(data_types, ~filter(asy_mean_tbl, stri_detect_fixed(dataset, .x))) %>%
    set_names(data_types)

iwalk(asy_mean_tbls, function(asy_mean_tbl, data_type) {
    asy_mean_tbl <- mutate(asy_mean_tbl, response=ifelse(stri_detect_fixed(dataset, "linear"), "Linear", "Non-linear"))
    p2 <- ggplot(asy_mean_tbl, aes(x=K, y=value, color=metric)) +
        geom_line() +
        facet_grid(response~fdr_control_level) + 
        theme_bw() + 
        geom_hline(aes(yintercept=fdr_control_level), linetype="dashed") + 
        scale_y_continuous(limits=c(0, 1)) +
        scale_color_discrete(name="Metric") +
        xlab("Number of MCMC samples") +
        ylab("Power and FDR") +
        theme(strip.text.x=element_blank()) +
        theme(
            legend.position="bottom", 
        # legend.justification = "right", 
            legend.background = element_rect(linetype = 1, size = 0.5, colour = 1)
      )

    ggsave("./output/experiments/" %s+% data_type %s+% "_asymptotic_fdr_and_power.png", p2, width=8, height=6, scale=0.65, type="cairo")
})
