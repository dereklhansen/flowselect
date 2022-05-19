library(tidyverse)
library(stringi)
library(ggrepel)
joint_names = tribble(
  ~"Joint", ~"joint", 
  "Ground Truth" , "true_dgp",
  "DDLK", "ddlk",
  "GaussMAF", "gaussmaf_5"
)

knockoff_names = tribble(
  ~"Knockoff", ~"knockoff",
  "Model-X", "modelx",
  "DDLK", "ddlk",
  "FlowSelect", "mcmc",
  "FlowSelect", "mcmc_1000",
  "FlowSelect", "mcmc_n2000_s1000",
  "FlowSelect", "mcmc_n10000_s1000",
  "FlowSelect", "mcmc_n50000_s1000",
  "KnockoffGAN", "gan",
  "DeepKnockoff", "deepknockoff",
  "HRT", "hrt",
  "MASS", "mass"
)

var_sel_names = tribble(
  ~"Variable Selection", ~"var_sel",
  "LASSO", "lasso",
  "HRT-LASSO (n=2000)", "fast_lasso_2000",
  "HRT-LASSO (n=10000)", "fast_lasso_10000",
  "HRT-LASSO (n=2000)", "khrt_2000_fast_lasso",
  "HRT-LASSO (n=10000)", "khrt_10000_fast_lasso",
  "HRT-RF (n=2000)", "fast_rf_2000",
  "HRT-RF (n=10000)", "fast_rf_10000",
  "HRT-RF (n=50000)", "fast_rf_50000",
  "HRT-RF (n=2000)", "khrt_2000_fast_rf",
  "HRT-RF (n=10000)", "khrt_10000_fast_rf",
  "HRT-RF (n=50000)", "khrt_50000_fast_rf"
)



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

get_power_csvs <- function(outdir, model_df) {
  power_df <- mutate(model_df,
    power_file = paste(outdir,
      datasets,
      run,
      joint,
      knockoff,
      var_sel,
      "power.csv",
      sep = "/"
    )
  ) %>%
    filter(file.exists(power_file)) %>%
    mutate(power_csv = map(power_file, read_csv)) %>%
    unnest(cols = power_csv) %>%
    # select(-X1) %>%
    tidyr::gather(power, fdr, key = "metric", value = "value")
  return(power_df)
}

read_file_safe <- function(f) {
  if (file.exists(f)) {
    y <- read_file(f)
  } else {
    y <- NA_character_
  }
  return(y)
}

get_timings <- function(outdir, model_df) {
  time_df <- mutate(model_df,
    joint_time_file = paste(outdir,
      datasets,
      run,
      joint,
      "time.txt",
      sep = "/"
    ),
    time_file = paste(outdir,
      datasets,
      run,
      joint,
      knockoff,
      var_sel,
      "time.txt",
      sep = "/"
    )
  ) %>%
  filter(file.exists(time_file)) %>%
  mutate(time_str = map_chr(time_file, read_file)) %>%
  mutate(time = as.double(time_str)) %>%
  mutate(time_sec = time / 1e9) %>%
  mutate(
    joint_time_str = map_chr(joint_time_file, read_file_safe)
  ) %>%
  mutate(joint_time_sec = as.double(joint_time_str) / 1e9)
  return(time_df)
}

analyze_linear_model <- function(cfg) {
  outdir <- cfg$dir
  model_df <- get_all_runs(outdir)
  print(model_df)
  #print(cfg$cache)
  if (cfg$cache && file.exists(outdir %s+% "/plot_data.rds")) {
    power_df <- read_rds(outdir %s+% "/plot_data.rds")
    time_df <- read_rds(outdir %s+% "/timings.rds")
  } else {
    power_df <- get_power_csvs(outdir, model_df)
    time_df <- get_timings(outdir, model_df)
    write_rds(power_df, outdir %s+% "/plot_data.rds")
    write_rds(time_df, outdir %s+% "/timings.rds")
    write_csv(time_df, outdir %s+% "/timings.csv")
  }
  
  power_csvs <- power_df %>%
    unite("procedure", Joint, Knockoff, `Variable Selection`, sep="/", remove=FALSE) 
  print(power_csvs)
  if (!is.null(cfg$joints)) {
      power_csvs <- filter(power_csvs, joint %in% strsplit(cfg$joints, ",")[[1]])
  }
  if (!is.null(cfg$knockoffs)) {
      print(strsplit(cfg$knockoffs, ","))
      power_csvs <- filter(power_csvs, knockoff %in% strsplit(cfg$knockoffs, ",")[[1]])
  }
  if (!is.null(cfg$varsels)) {
      power_csvs <- filter(power_csvs, var_sel %in% strsplit(cfg$varsels, ",")[[1]])
  }

  for (dataset in c(
    "gaussian_ar_mixture_.*_d100_n1000_rho0.98_",
    "gaussian_ar_mixture_.*_d100_n10000_rho0.98_",
    "rnasq_cor_.*_a40_n10_nv100_nvc5_nt1000_",
    "rnasq_cor_.*_a40_n10_nv100_nvc5_nt10000_"
    )) {
    plot_tbl <- power_csvs %>%
      filter(stri_detect_regex(datasets, dataset)) %>%
      mutate(outcome = ifelse(stri_detect_fixed(datasets, "linear"), "Linear", "Non-linear")) %>%
      filter(!(Joint == "GaussMAF" & Knockoff == "DDLK")) %>%
      filter(!(Joint == "Ground Truth" & Knockoff == "DDLK")) %>%
      # filter((Joint == "Ground Truth" & Knockoff == "DDLK")) %>%
      filter(!(Joint == "Ground Truth" & Knockoff == "FlowSelect")) %>%
      filter(!(outcome == "Linear" & `Variable Selection` == "HRT-RF (n=10000)")) %>%
      group_by(datasets, outcome, n_fdrs, Joint, Knockoff, `Variable Selection`, metric) %>%
      mutate(mean_value = mean(value), sd_value = sd(value, na.rm=TRUE)) %>%
      mutate(run_Knockoff = run %s+% "_" %s+% Knockoff) %>%
      ungroup()

    fdr_plot <- ggplot(
      plot_tbl, aes(x = n_fdrs, y = value, color = Knockoff, fill=Knockoff)
      ) +
      # geom_line(aes(group=run_Knockoff), alpha=0.20) +
      geom_line(aes(y=mean_value)) +
      # geom_ribbon(aes(ymin=mean_value - sd_value, ymax=mean_value+sd_value), alpha=0.1) +
      facet_grid(outcome~metric) +
      geom_function(fun = identity, color="black") +
      ggtitle("Observed FDR and Power at each nominal FDR") + 
      theme_bw()

    
    ggsave(
      outdir %s+% "/" %s+% dataset %s+% "_fdr_and_power.png",
      fdr_plot,
      width = 8,
      height = 5 * length(unique(plot_tbl$`Variable Selection`)),
      type = "cairo"
    )

    fdr_boxplot <- plot_tbl %>%
      filter(n_fdrs %in% c(0.05, 0.10, 0.25)) %>%
      mutate(Knockoff = factor(Knockoff, c("DeepKnockoff", "KnockoffGAN", "MASS", "HRT", "DDLK", "FlowSelect"))) %>%
      mutate(metric = factor(metric, c("fdr", "power"), c("FDR", "Power"))) %>%
      ggplot(aes(x=Knockoff, y=value, color=metric)) +
      geom_boxplot() +
      xlab(NULL) +
      ylab("Power and FDR") +
      geom_hline(aes(yintercept=n_fdrs), linetype="dashed") +
      facet_grid(outcome~n_fdrs) +
      theme_bw() +
      scale_x_discrete(guide=guide_axis(angle=45)) + 
      scale_y_continuous(limits = c(0,1)) +
      scale_color_discrete(name="Metric") +
      theme(axis.text.x = element_text(face = c('plain', 'plain', 'plain', 'plain', 'bold')),
            strip.text.x=element_blank())+
      theme(
        legend.position="bottom", 
        legend.background = element_rect(linetype = 1, size = 0.5, colour = 1)
      )
    dataset_out <- stri_replace_all_fixed(dataset, ".*_", "")
    ggsave(
      outdir %s+% "/" %s+% dataset_out %s+% "_boxplots.png",
      fdr_boxplot,
      width = 8,
      height = 6,
      scale=0.65,
      type = "cairo"
    )
    fdr_meanplot <- plot_tbl %>%
      filter(n_fdrs %in% c(0.05, 0.10, 0.25)) %>%
      mutate(Knockoff = factor(Knockoff, c("DeepKnockoff", "KnockoffGAN", "MASS", "HRT", "DDLK", "FlowSelect"))) %>%
      mutate(metric = factor(metric, c("fdr", "power"), c("FDR", "Power"))) %>%
      ggplot(aes(x=Knockoff, color=metric, group=metric)) +
      # geom_boxplot() +
      # geom_segment(aes(xend=Knockoff, y=mean_value-sd_value, yend=mean_value+sd_value), position=position_dodge(width=0.3))+
      # geom_errorbar(aes(ymin=mean_value - sd_value, ymax=mean_value + sd_value), position=position_dodge(width=0.5))+
      geom_linerange(aes(ymin=mean_value - sd_value, ymax=mean_value + sd_value), position=position_dodge(width=0.5))+
      geom_point(aes(y=mean_value), position=position_dodge(width=0.5)) +
      xlab(NULL) +
      ylab("Power and FDR") +
      geom_hline(aes(yintercept=n_fdrs), linetype="dashed") +
      facet_grid(outcome~n_fdrs) +
      theme_bw() +
      scale_x_discrete(guide=guide_axis(angle=45)) + 
      scale_y_continuous(limits = c(0,1)) +
      scale_color_discrete(name="Metric") +
      theme(axis.text.x = element_text(face = c('plain', 'plain', 'plain', 'plain', 'bold')),
            strip.text.x=element_blank())+
      theme(
        legend.position="bottom", 
        legend.background = element_rect(linetype = 1, size = 0.5, colour = 1)
      )
    dataset_out <- stri_replace_all_fixed(dataset, ".*_", "")
    ggsave(
      outdir %s+% "/" %s+% dataset_out %s+% "_meanplots.png",
      fdr_meanplot,
      width = 8,
      height = 6,
      scale=0.65,
      type = "cairo"
    )






    ##****************************
    ## Power Plot for each method
  ##*********************************
  }
  library(xtable)
  print(time_df)
  joint_timings <- time_df %>%
      filter(`Variable Selection` == "HRT-RF (n=10000)") %>%
      filter(datasets == "rnasq_cor_sin_cos5_a40_n10_nv100_nvc5_") %>%
      select(Joint, joint_time_sec) %>%
      distinct() %>%
      group_by(Joint) %>%
      summarize(
        mean = mean(joint_time_sec),
        median = median(joint_time_sec),
        best = min(joint_time_sec),
        worst = max(joint_time_sec))
  write_csv(joint_timings, outdir %s+% "/joint_timings.csv")
  xtable(joint_timings) %>%
    print(file=outdir %s+% "/joint_timings.tex")
  timings_processed <- time_df %>%
      filter(`Variable Selection` == "HRT-RF (n=10000)") %>%
      filter(datasets == "rnasq_cor_sin_cos5_a40_n10_nv100_nvc5_") %>%
      select(Joint, Knockoff, time_sec) %>%
      group_by(Joint, Knockoff) %>%
      summarize(mean = mean(time_sec), median = median(time_sec), best = min(time_sec), worst = max(time_sec))

  write_csv(timings_processed, outdir %s+% "/timings_processed.csv")
  xtable(timings_processed) %>%
    print(file=outdir %s+% "/timings_processed.tex")
  ##****************************
  ## FDR Plot for each method

  

}
library("optparse")
parser <- OptionParser()
parser <- add_option(parser, c("-d", "--dir"), type="character", default="./output/experiments", help="Directory of output")
parser <- add_option(parser, c("-j", "--joints"), default=NULL, help="List of joint models to use (default is to use all found)")
parser <- add_option(parser, c("-k", "--knockoffs"), default=NULL, help="List of knockoff/CRT to use (default is to use all found)")
parser <- add_option(parser, c("-v", "--varsels"), default=NULL, help="List of variable selection models to use (default is to use all found)")
parser <- add_option(parser, c("-c", "--cache"), action="store_true", dest="cache", default=FALSE)
parsed <- parse_args(parser)
print(parsed)
analyze_linear_model(parsed)
