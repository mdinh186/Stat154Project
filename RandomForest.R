library(rpart)
library(randomForest)
library(data.table)
library(caret)
library(doParallel)
library(plotROC)


#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

df_impute = readRDS("data/df_impute_feat.rds")
df_impute = data.table(df_impute)

#saveRDS(df_impute,"data/df_impute_feat.rds")

n = ceiling(nrow(df_impute) * 0.8)
#############################################################
# Prepocessing data:
#############################################################
col = c( "workclass", "marital_status", "occupation", "relationship", "native_country","gen_race")
df_impute[,(col):= lapply(.SD, function(x) as.factor(x)), .SDcols =col]
df_impute$sex = NULL
train_idx = sample(nrow(df_impute),n)
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income

xtest_origin  = test_origin[,-c("income")]
ytest_origin = test_origin$income


#######################################
#make baseline models and feature sections



control = trainControl(method = "cv", number = 5,savePredictions = T,classProbs=T,summaryFunction=twoClassSummary)

rf_default  = train(x = xtrain_origin, y = ytrain_origin,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)



###
selectedIndices = rf_default$pred$mtry == 9

g <- ggplot(rf_default$pred[selectedIndices, ], aes(m=More.50k, d=factor(obs, levels = c("Less.50k","More.50k")))) + 
  geom_roc(n.cuts=0) + 
  coord_equal() +
  style_roc()

g + annotate("text", x=0.75, y=0.25, label=paste("AUC =", round((calc_auc(g))$AUC, 4)))


rf_default
pred = predict(rf_default, xtest_origin, cutoff = 0.3)
confusionMatrix(ytest_origin, pred, positive = "More.50k")
varImp(rf_default)



########################################################
### result for version1:  12/5 
####impute with feature: 
#### Accuracy: train: 86 %, test = 86%, t
### evaluation:
# Prediction      Less.than.50k More.than.50k
# Less.than.50k          6011           364
# More.than.50k           792          1264


#### impute without feature: 
#### Accuracy: train:82.7?%

# Reference
# Prediction  <=50K  >50K
# <=50K   4934    25
# >50K    1111   442


### remove missing value without feature: 
#### accuracy train: 81.7%

# Reference
# Prediction  <=50K  >50K
# <=50K   4529    22
# >50K    1022   459

### remove mssing values with feature: 
#### accuracy train:  82.8%
# Reference
# Prediction      Less.than.50k More.than.50k
# Less.than.50k          4473            32
# More.than.50k          1065           462



########################################################
### result for version 2:  12/5 
# (use AUC as metrics)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  <=50K  >50K
# <=50K   4602   322
# >50K     589   999

# Reference
# Prediction Less.50k More.50k
# Less.50k     4571      377
# More.50k      567      997

# Reference
# Prediction Less.50k More.50k
# Less.50k     4571      377
# More.50k      572      992

##########################################################################################################################
# Random Forest Parameter tuning: 

###### parallel processing: 
cluster = makeCluster(detectCores() -1)
registerDoParallel(cluster)


####### Tuning mtries: 
##### random search: 
seed = 1234
set.seed(1234)

control = trainControl(method = "cv", number = 5, search = "random",allowParallel = TRUE)
mtry = sqrt(ncol(xtrain_origin))
rf_random = train(xtrain_origin, ytrain_origin, method = "rf", 
                  tuneLength = 15, trControl=control)

###rf_random results:
#      Random Forest 
      
 #     26049 samples
#      16 predictor
#      2 classes: 'Less.50k', 'More.50k' 
      
      # No pre-processing
      # Resampling: Cross-Validated (5 fold) 
      # Summary of sample sizes: 20840, 20839, 20838, 20839, 20840 
      # Resampling results across tuning parameters:
      #   
      #   mtry  Accuracy   Kappa    
      # 1    0.8393417  0.4751006
      # 4    0.8607242  0.6003849
      # 5    0.8574229  0.5924602
      # 9    0.8520486  0.5776796
      # 10    0.8510505  0.5758396
      # 11    0.8506280  0.5743715
      # 12    0.8501290  0.5736396
      # 14    0.8507819  0.5746413
      # 15    0.8502445  0.5733095
      # 
      # Accuracy was used to select the optimal model using the
      # largest value.
      # The final value used for the model was mtry = 4.
      
    
pred = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred, positive = "More.50k")

###Confusion Matrix and Statistics
      # Accuracy : 0.859           
      # 95% CI : (0.8503, 0.8674)
      # No Information Rate : 0.7919          
      # P-Value [Acc > NIR] : < 2.2e-16       
      # 
      # Kappa : 0.5964          
      # Mcnemar's Test P-Value : 4.899e-13       
      # 
      # Sensitivity : 0.7424          
      # Specificity : 0.8897          
      # Pos Pred Value : 0.6387          
      # Neg Pred Value : 0.9293          
      # Prevalence : 0.2081          
      # Detection Rate : 0.1545          
      # Detection Prevalence : 0.2419          
      # Balanced Accuracy : 0.8160          
      # 
      # 'Positive' Class : More.50k
################################### This is for the current data set?
# Prediction Less.50k More.50k
# Less.50k     4588      349
# More.50k      569     1006

################################### I think this was the the last dataset (before we input everything?)
# Prediction Less.50k More.50k
# Less.50k     4602      346
# More.50k      560     1004


##### grid search: 
control2 = trainControl(method = "cv", number = 5, search = "grid", allowParallel = TRUE)
tunegrid = expand.grid(.mtry = c(4:7))
rf_gridsearch = train(xtrain_origin, ytrain_origin, method = "rf",
                      tuneGrid=tunegrid, trControl=control2)

###rf_gridsearch results
        # Random Forest 
        # 
        # 26049 samples
        # 16 predictor
        # 2 classes: 'Less.50k', 'More.50k' 
        # 
        # No pre-processing
        # Resampling: Cross-Validated (5 fold) 
        # Summary of sample sizes: 20839, 20839, 20840, 20840, 20838 
        # Resampling results across tuning parameters:
        #   
        #   mtry  Accuracy   Kappa    
        # 4     0.8610310  0.6008713
        # 5     0.8593804  0.5965750
        # 6     0.8568851  0.5901975
        # 7     0.8553879  0.5864568
        # 
        # Accuracy was used to select the optimal model using the
        # largest value.
        # The final value used for the model was mtry = 4.

pred_grid = predict(rf_random, xtest_origin)
confusionMatrix(ytest_origin, pred_grid, positive = "More.50k")
# Prediction Less.50k More.50k
# Less.50k     4600      337
# More.50k      568     1007 
      # Confusion Matrix and Statistics
      # Accuracy : 0.861           
      # 95% CI : (0.8524, 0.8693)
      # No Information Rate : 0.7936          
      # P-Value [Acc > NIR] : < 2.2e-16       
      # 
      # Kappa : 0.6011          
      # Mcnemar's Test P-Value : 2.082e-14       
      # 
      # Sensitivity : 0.7493          
      # Specificity : 0.8901          
      # Pos Pred Value : 0.6394          
      # Neg Pred Value : 0.9317          
      # Prevalence : 0.2064          
      # Detection Rate : 0.1546          
      # Detection Prevalence : 0.2419          
      # Balanced Accuracy : 0.8197          
      # 
      # 'Positive' Class : More.50k 