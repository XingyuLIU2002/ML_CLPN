library(glmnet)
library(qgraph)
library(bootnet)
library(NetworkComparisonTest)
library(EGAnet)
library(mgm)
library(networktools)
library(ggplot2)
library(dplyr)
library(tidyr)
library(doParallel)
library(igraph)
library(RColorBrewer)
library(here)

rm(list = ls())
set.seed(1234)
setwd("D:/1.SJTU/MentalHealthProjects/ML_CLPN_Project")

############################ 1. Load Functions ################################
source("02_Programme/nw01_cleandata_variable.R")
source("02_Programme/nw02_basicnet_variable.R")
source("02_Programme/nw03_compare.R")
source("02_Programme/nw04_CLPN_variable.R")

############################ 2. Variable Definition #############"support"###############
input_dir <- "01_Input/df_all.csv"
output_base_dir <- "03_output/network_analysis_variable_level"
var_extract <- c("anxiety","depression","insomnia","life_satis","loneliness",
               "per_stress","psmu","somatic")
# 协变量
con_vars <- c("gender_T1")
time_points <- c("T1", "T2")
var_mapping <- list(
  anxiety = "ANX",
  depression = "DEP",
  insomnia = "INS",
  life_satis = "SAT",
  loneliness = "LON",
  per_stress = "PST",
  psmu = "PSU",
  somatic = "SOM"
)
na_handling <- "median"

lognames <- c(
  "ANX" = "Anxiety symptoms",
  "DEP" = "Depressive symptoms",
  "INS" = "Insomnia",
  "SAT" = "Life satisfaction",
  "LON" = "Loneliness",
  "PST" = "Perceived stress",
  "PSU" = "Problematic social media use",
  "SOM" = "Somatic symptoms"
)
node_names <- c("ANX","DEP","INS","SAT","LON","PST", "PSU","SOM")
communities <- c(" ","  ","   ",
                 "    ","     ","      ",
                 "       ","        ")
communities_times <- rep(1, 8)
community_colors <- c(
  "Anxiety symptoms" = "#E8B3B3",
  "Depressive symptoms" = "#AFC8E3",
  "Insomnia" = "#C3B7E2",
  "Life satisfaction" = "#B0B0B0",
  "Loneliness" = "#F2C98A",
  "Perceived stress" = "#F4E28C",
  "Problematic social media use" = "#9EC7B5",
  "Somatic symptoms" = "#D1B199"
)
################## Data prepare ###############################
data_prepared <- prepare_data(
  input_dir = input_dir,
  main_vars = var_extract,
  con_vars = c("gender_T1"),
  time_points = c("T1", "T2"),
  var_mapping = var_mapping,
  na_handling = "median"
)
t1_data <- data_prepared$complete_t1
t2_data <- data_prepared$complete_t2

clpn_data <- cbind(
  t1_data %>% setNames(paste0(names(.), "_T1")),
  t2_data %>% setNames(paste0(names(.), "_T2"))
)
# ################## T1/T2 Analysis ###############################
cat("Start the complete network analysis process...\n")
# 1. 分析T1网络
cat("=== Analyze T1 network ===\n")
t1_output_dir <- file.path(output_base_dir, "T1")
# if (!dir.exists(t1_output_dir)) dir.create(t1_output_dir, recursive = TRUE)
t1_results <- analyze_single_timepoint(
  t1_data, "T1", file.path(t1_output_dir),
  communities, community_colors, communities_times, lognames
)
# 2. 分析T2网络
cat("=== Analyze T2 network ===\n")

t2_output_dir <- file.path(output_base_dir, "T2")
# if (!dir.exists(t1_output_dir)) dir.create(t2_output_dir, recursive = TRUE)
t2_results <- analyze_single_timepoint(
  t2_data, "T2", file.path(t2_output_dir),
  communities, community_colors, communities_times, lognames
)


# 3. 比较T1和T2网络
cat("=== Compare T1 and T2 networks ===\n")
compare_output_dir <- file.path(output_base_dir, "NCT_Comparison_T1_T2")
if (!dir.exists(compare_output_dir)) dir.create(compare_output_dir, recursive = TRUE)

nct_results <- compare_two_networks_nct(
  D1 = t1_data,
  D2 = t2_data,
  type_vec = type_vec,
  level_vec = level_vec,
  output_dir = compare_output_dir,
  n_iter = 10  # 正式输出调到1000
)

################## Cross-lagged Panel Network analysis ############################
# 4. T1->T2交叉滞后网络分析
cat("=== Cross-lagged Panel Network analysis of two waves ===\n")
clpn_output_dir <- file.path(output_base_dir, "CLPN")
network_results <- analyze_clpn_network(
  t1_data = t1_data,
  t2_data = t2_data,
  output_dir = clpn_output_dir,
  lognames = lognames,
  var_mapping = var_mapping,
  communities_times = communities_times,
  community_colors = community_colors,
  threshold_value = 0.05,
  iter = 1000 # 正式输出调到1000
)