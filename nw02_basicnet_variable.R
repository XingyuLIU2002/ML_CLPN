################## packages ###########################
library(mgm)
library(bootnet)
library(qgraph)
library(ggplot2)
library(openxlsx)
library(networktools)
################## Network Analysis Functions ############################
analyze_single_timepoint <- function(data, time_point, output_dir, communities, 
                                     community_colors, communities_times, lognames) {
  
  result_file <- file.path(output_dir, paste0("network_analysis_", time_point, ".rds"))
  
  
  # if (file.exists(result_file)) {
  #   cat("Load the existing stability analysis results...\n")
  #   return(readRDS(result_file))
  # }
  
  cat("Analyzing", time_point, "network...\n")
  
  tryCatch({
    # Create output directory
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
    
    # 1. Prepare data
    type_vec <- ifelse(
      sapply(data, function(x) length(unique(x)) <= 5), "c", "g"
    )
    level_vec <- sapply(data, function(x) length(unique(x)))
    level_vec[type_vec == "g"] <- 1
    
    dfmat <- as.matrix(data)
    
    # 2. Build MGM model (Base model for plotting later)
    mgm_model <- mgm(
      data = dfmat,
      type = type_vec,
      level = level_vec,
      lambdaSel = "EBIC",
      lambdaGam = 0.25,
      scale = TRUE,
      verbosity = 0 # 减少mgm的输出信息
    )
    
    # 3. Calculate prediction performance
    pred <- predict(mgm_model, data = data, 
                    error.continuous = "R2", error.categorical = "CC")
    
    errors_mat <- pred$errors
    if ("R2" %in% colnames(errors_mat)) {
      perf_vec <- ifelse(is.na(errors_mat[, "R2"]),
                         errors_mat[, "CC"],
                         errors_mat[, "R2"])
    } else if ("CC" %in% colnames(errors_mat)) {
      perf_vec <- errors_mat[, "CC"]
    } else {
      if (ncol(errors_mat) > 0) {
        perf_vec <- errors_mat[, 1]
      } else {
        perf_vec <- rep(0, ncol(data))
      }
    }
    
    # Ensure values are between 0-1 for pie chart
    perf_vec <- pmin(pmax(perf_vec, 0), 1)
    
    # 4. Basic network plot
    node_names <- colnames(dfmat)
    full_names <- lognames[node_names]
    # === 定义 community 分组（关键修复）===
    item <- factor(communities, levels = communities)

    # Define variable groups
    #item_nolabel <- factor(item, levels = unique(item), labels = rep("", length(unique(item))))
    # Create basic network plot
    basic_net <- qgraph(
      mgm_model$pairwise$wadj,
      layout = "spring",
      groups = item,
      color = community_colors,
      labels = node_names,
      nodeNames = full_names,
      pie = perf_vec,
      vsize = 5,
      asize = 3,
      border.width = 1.2,
      legend = TRUE,
      legend.cex = 0.5,
      label.cex = 0.8, 
      edge.width = 0.5
      # ,title = paste("Network -", time_point)
    )
    
    # Save basic network plot
    file_base <- file.path(output_dir, paste0("plot_basic_network_", time_point))
    save_plot_formats(basic_net, file_base, width = 12, height = 8)
    # print(basic_net)
    
    # pdf_file <- file.path(output_dir, paste0("plot_basic_network_", time_point, ".pdf"))
    # pdf(pdf_file, width = 10, height = 8)
    # plot(basic_net)  # 显示图表
    # dev.off()
    
    # 5. Centrality analysis
    
    centrality_results <- centralityTable(basic_net)
    
    plot_centrality <- centralityPlot(basic_net, 
                                      labels = paste0(lognames, " (", node_names, ")"), 
                                      scale = "z-scores",
                                      # include = c("Strength", "OutStrength", "InStrength"), 
                                      include = c('Strength', 'Closeness', 'Betweenness'),#, "ExpectedInfluence"
                                      orderBy = "Strength", decreasing = FALSE)
    
    # centrality_pdf_file <- file.path(output_dir, paste0("plot_centrality_", time_point, ".pdf"))
    # cat("Displaying centrality plot...\n")
    # 
    # pdf(centrality_pdf_file, width = 10, height = 8)
    # plot(plot_centrality)
    # dev.off()

    file_base <- file.path(output_dir, paste0("plot_centrality_", time_point))
    save_plot_formats(plot_centrality, file_base, width = 8, height = 6)
    print(plot_centrality)

    
    # 7. Network stability analysis using bootnet
    cat("Performing MGM network stability analysis using bootnet nonparameter...\n")
    
    # --- 运行 bootnet 分析 ---
    bootnet_nonpara <- bootnet(
      data = dfmat,             
      default = "mgm",           
      scale = TRUE,
      nBoots = 1000,
      nCores = parallel::detectCores() - 1,                
      computeStability = TRUE,   
      labels = node_names,
      verbose = TRUE,
      type = "nonparametric", 
      statistics = "all",
      arguments = list(
        level = level_vec,
        lambdaSel = "EBIC",
        lambdaGam = 0.25,
        pbar = 0
      )
    )
    
    # 准确性检验图绘制
    # bootnon_pdf_file <- file.path(output_dir, paste0("plot_bootnet_nonpara_", time_point, ".pdf"))
    # pdf(bootnon_pdf_file, width = 10, height = 8)
    # plot(bootnet_nonpara, labels=FALSE, order="sample", meanColor = NA)  # labels=FALSE 表示不显示节点标签，order="sample" 表示按样本排序绘图
    # dev.off()

    plot_bootnon_nonpara <- plot(bootnet_nonpara, labels=FALSE, order="sample", meanColor = NA)
    file_base <- file.path(output_dir, paste0("plot_bootnet_nonpara_", time_point))
    save_plot_formats(plot_bootnon_nonpara, file_base, width = 10, height = 8)

    
    # 绘制边的差异变化图，仅显示非零边，按样本排序
    # bootnon_edge_pdf_file <- file.path(output_dir, paste0("plot_bootnet_nonpara_edge_", time_point, ".pdf"))
    # pdf(bootnon_edge_pdf_file, width = 10, height = 8)
    # plot(bootnet_nonpara, "edge", plot = "difference", onlyNonZero = TRUE, order = "sample")
    # dev.off()

    plot_bootnon_edge <- plot(bootnet_nonpara, "edge", plot = "difference", onlyNonZero = TRUE, order = "sample")
    file_base <- file.path(output_dir, paste0("plot_bootnet_nonpara_edge_", time_point))
    save_plot_formats(plot_bootnon_edge, file_base, width = 10, height = 8)
    # "edge" 指定绘制边，plot = "difference" 表示绘制边的变化，onlyNonZero = TRUE 表示仅显示非零边
    
    
    # 强度差异检验
    # bootnon_strength_pdf_file <- file.path(output_dir, paste0("plot_bootnet_nonpara_strength_", time_point, ".pdf"))
    # pdf(bootnon_strength_pdf_file, width = 10, height = 8)
    # plot(bootnet_nonpara, statistics="Strength", plot="difference", order = "mean")  
    # dev.off()

    plot_bootnon_strength <- plot(bootnet_nonpara, statistics="Strength", plot="difference", order = "mean")
    file_base <- file.path(output_dir, paste0("plot_bootnet_nonpara_strength_", time_point))
    save_plot_formats(plot_bootnon_strength, file_base, width = 10, height = 8)
    
    
    cat("Performing MGM network stability analysis using bootnet case...\n")
    bootnet_case <- bootnet(
      data = dfmat,             
      default = "mgm",           
      scale = TRUE,
      nBoots = 1000,              # 实际分析建议设置为 nBoot = 1000
      nCores = parallel::detectCores() - 1,                
      computeStability = TRUE,   
      labels = node_names,
      verbose = TRUE,
      type = "case", 
      statistics = "all",
      arguments = list(
        level = level_vec,
        lambdaSel = "EBIC",
        lambdaGam = 0.25,
        pbar = 0
      )
    )
    
    
    # --- 提取并保存稳定性数据 ---
    cs_coef_summary <- corStability(bootnet_case) # 获得 CS 系数
    bootstrap_summary_df <- summary(bootnet_case) # 获得置信区间数据框
    
    # --- 保存稳定性结果图像 ---
    # bootnet_case_pdf <- file.path(output_dir, paste0("plot_bootnet_case_", time_point, ".pdf"))
    # pdf(bootnet_case_pdf, width = 10, height = 8)
    # plot(bootnet_case, statistics = c("Strength","Closeness","Betweenness","ExpectedInfluence"))
    # dev.off()
    
    plot_bootnet_case <- plot(bootnet_case, statistics = c("Strength","Closeness","Betweenness"))#,"ExpectedInfluence"
    file_base <- file.path(output_dir, paste0("plot_bootnet_case_", time_point))
    save_plot_formats(plot_bootnet_case, file_base, width = 10, height = 8)
    
    # 将稳定性结果汇总到一个列表中，方便保存到 Excel
    stability_results_list <- list(
      CS_Coefficients = as.data.frame(cs_coef_summary),
      Centrality_CIs = bootstrap_summary_df$centRes,
      Edge_CIs = bootstrap_summary_df$edgeRes
    )
    
    stability_excel_file <- file.path(output_dir, paste0("bootnet_stability_results_", time_point, ".xlsx"))
    write.xlsx(stability_results_list, file = stability_excel_file)
    cat("Stability data saved to Excel:", stability_excel_file, "\n")
    

    write.csv(mgm_model$pairwise$wadj,
              file.path(output_dir, paste0("adjacency_matrix_", time_point, ".csv")))
    
    write.csv(centrality_results,
              file.path(output_dir, paste0("centrality_measures_", time_point, ".csv")))
    
    
    # Prepare return results
    results <- list(
      mgm_model = mgm_model,
      performance = perf_vec,
      centrality = centrality_results,
      network_plot = basic_net,
      stability = list(
        bootnet_case = bootnet_case, # 保存完整的 bootnet 对象
        bootnet_nonpara = bootnet_nonpara,
        cs_summary = cs_coef_summary,
        summary_df = bootstrap_summary_df,
        method_used = "mgm_bootnet"
      )
    )
    
    # Save results RDS
    saveRDS(results, result_file)
    cat(time_point, "network analysis completed and results saved in RDS format.\n")
    
    return(results)
    
  }, error = function(e) {
    cat("Error in", time_point, "network analysis:", e$message, "\n")
    return(NULL)
  })
}

save_plot_formats <- function(plot_obj, file_path_base, width = 10, height = 8) {
  # Save PDF
  pdf_file <- paste0(file_path_base, ".pdf")
  pdf(pdf_file, width = width, height = height)
  plot(plot_obj)
  dev.off()
  cat("Plot saved:", pdf_file, "\n")
  
  # Save PNG
  png_file <- paste0(file_path_base, ".png")
  png(png_file, width = width, height = height, units = "in", res = 400) # Use high resolution for PNG
  plot(plot_obj)
  dev.off()
  cat("Plot saved:", png_file, "\n")
}