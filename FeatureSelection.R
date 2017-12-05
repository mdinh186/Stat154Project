library(rpart)
library(randomForest)
library(data.table)
library(caret)

#############################################################
dir = "/Users/MyDinh/Downloads/Stat154/Stat154Project/"
setwd(dir)
set.seed(123)

df_impute = readRDS("data/df_remove.rds")
df_impute = data.table(df_impute)
saveRDS(df_impute, "data/df_remove.rds")


n = ceiling(nrow(df_impute) * 0.8)
#############################################################
# Prepocessing data:
#############################################################



train_idx = sample(nrow(df_impute),n)
train_origin = df_impute[train_idx,]
test_origin = df_impute[-train_idx, ]
xtrain_origin  = train_origin[,-c("income")]
ytrain_origin = train_origin$income

xtest_origin  = test_origin[,-c("income")]
ytest_origin = test_origin$income


#######################################
#make baseline models and feature sections

control = trainControl(method = "cv", number = 5)

rf_default  = train(x = train_origin[,-14], y = train_origin$income,
                  method = "rf", trControl = control, 
                    importance= T, ntree = 100)




rf_default
pred = predict(rf_default, xtest_origin, cutoff = 0.5)
confusionMatrix(ytest_origin, pred)
####impute with feature: 
#### Accuracy: train: 86 %, test = 86%, t
### evaluation:
# Prediction      Less.than.50k More.than.50k
# Less.than.50k          6011           364
# More.than.50k           792          1264

