model_roc_plot = function(model_list, custom_col, AUC= FALSE){

# Function that takes in different models, calculate the AUC,
# and plot the ROC curve. Return AUC if specified
#   Argument: 
#   model_list: list of models (that train on train on train dataset, we can use predict on those models to predict data)
#   custom_col: vector of color for each model. The length of vector should be equal to the length of model  

  if (length(custom_col) != length(model_list)){
    stop("Model list and number of colors to plot must be equal")
  }
  
  test_roc <- function(model, data){
#     Cacluate AUC  
#  
    roc(data$income,
        predict(model, data, type = "prob")[, "More.50k"])
    
  }
  model_list_pr = model_list %>% 
    map(test_roc, data = test_origin)
  
  
  results_list_roc <- list(NA)
  num_mod <- 1
  
  for(the_roc in model_list_pr){
    
    results_list_roc[[num_mod]] =
      data_frame(tpr = the_roc$sensitivities,
                 fpr = 1 - the_roc$specificities,
                 model = names(model_list)[num_mod])
    
    num_mod = num_mod + 1
    
  }
  results_df_roc =  bind_rows(results_list_roc)
  

  g = ggplot(aes(x = fpr,  y = tpr, group = model), data = results_df_roc) +
geom_line(aes(color = model), size = 1) +
    scale_color_manual(values = custom_col) +
    geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
    theme_bw(base_size = 18)
  print (g)
  if (AUC == T){
    area= model_list_pr %>%
      map(auc)
    return (area)
  }
  
}


