library(tidyverse)
library(rhdf5)
library(stringi)
get_all_runs <- function(outdir) {
  model_df <- list.dirs(outdir, recursive=TRUE, full.names=FALSE) %>% 
    {.[!stri_detect_fixed(., "logs")]} %>%
    {.[stri_detect_regex(., ".*/.*/.*/.*/.*")]} %>%
    {tibble(fname=.)} %>%
    tidyr::separate(fname, into=c("datasets", "run", "joint", "knockoff", "var_sel"), sep="/") %>%
    filter(complete.cases(.)) %>%
    dplyr::arrange(datasets, joint, knockoff, var_sel)

  # If available, add pretty names for each method for plotting.
  model_df <- model_df %>%
    left_join(joint_names, by="joint") %>%
    mutate(Joint=ifelse(is.na(Joint), joint, Joint)) %>%
    left_join(knockoff_names, by="knockoff") %>%
    mutate(Knockoff=ifelse(is.na(Knockoff), knockoff, Knockoff)) %>%
    left_join(var_sel_names, by="var_sel") %>%
    mutate(`Variable Selection`=ifelse(is.na(`Variable Selection`), var_sel, `Variable Selection`))


  return(model_df)
}

### HRT
X_hrt_12 <- h5read("./experiments/hrt_null_12.h5", "X_hrt_12")

hrt_dt <- X_hrt_12 %>%
  t %>%
  as_tibble %>%
  mutate(model = "hrt") %>%
  select(model, everything())

run_dir <- "./output/experiments/gaussian_ar_mixture_linear_d100_n100000_rho0.98_/run001/"

knockoff_sample_files <- c(
    ddlk = run_dir %s+% "ddlk/ddlk/knockoff_sample.h5",
    deepknockoff = run_dir %s+% "nojoint/deepknockoff/knockoff_sample.h5",
    gan = run_dir %s+% "nojoint/gan/knockoff_sample.h5",
    mass = run_dir %s+% "nojoint/mass/knockoff_sample.h5"
)

joint_sample_files <- c(
    flowselect = run_dir %s+% "gaussmaf_5/joint_sample.h5"
)

true_sample_file <- c(
    ground_truth = run_dir %s+% "gaussmaf_5/joint_sample.h5"
)

dist_samples <- c(knockoff_sample_files, joint_sample_files) %>%
    map(~h5read(.x, "xTr_tilde"))

gt_samples <- map(true_sample_file, ~h5read(.x, "xTr"))

dist_df <- c(dist_samples, gt_samples) %>%
    map(~as_tibble(t(.x[1:2, ]))) %>%
    bind_rows(.id="model") %>%
    bind_rows(hrt_dt)

dist_df2 <- dist_df %>%
  mutate(model = factor(
    model, 
    c("ground_truth", "deepknockoff", "gan", "mass", "ddlk", "flowselect"),
    c("Ground Truth", "DeepKnockoff", "KnockoffGAN", "MASS", "DDLK", "FlowSelect")
     ))

p <- ggplot(dist_df2, aes(V1, V2)) + geom_hex(bins=30) +
    scale_x_continuous(limits=c(-5, 45)) +
    scale_y_continuous(limits=c(-5, 45)) +
    facet_grid(.~model) +
    xlab("X1") +
    ylab("X2") +
    theme_bw()

ggsave(run_dir %s+% "knockoff_compare.png", p, width=10, height=2, type="cairo")


ps <- map(sort(unique(dist_df2$model)), function(mname) {
  df <- filter(dist_df2, model==mname)
  ggplot(df, aes(V1, V2)) + 
      stat_density_2d(
        aes(fill = ..density..), 
        geom = "raster", 
        contour = FALSE,
        h = c(1.0, 1.0)
      ) +
      scale_x_continuous(limits=c(-5, 45)) +
      scale_y_continuous(limits=c(-5, 45)) +
      xlab(NULL) +
      ylab(NULL) +
      ggtitle(mname) +
      guides(color=FALSE, fill=FALSE)+
      theme_bw()
})

library(gridExtra)
p2 <- arrangeGrob(grobs=ps, nrow=1)

ggsave(run_dir %s+% "knockoff_compare_heatmap.png", p2, width=10.5, height=2, type="cairo")
