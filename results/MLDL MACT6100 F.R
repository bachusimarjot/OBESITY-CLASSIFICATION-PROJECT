packages<-c("readxl", "dplyr", "caret", "nnet", "randomForest","xgboost", "keras", "tensorflow", "ggplot2") 
install.packages(setdiff(packages, rownames(installed.packages())))  
library(readxl)  
library(dplyr)  
library(caret) 
library(nnet) 
library(randomForest) 
library(xgboost) 
library(keras) 
library(tensorflow)  
library(ggplot2)  
data<-read_excel("ObesityDataSet.xlsx")  
data$Obesity<-as.factor(data$Obesity)  
names(data)  
colnames(data)[colnames(data)=="NObeyesdad"]<-"Obesity"  
data$Obesity <- as.factor(data$Obesity) 
dummies <- dummyVars(Obesity ~ ., data = data) 
X <- predict(dummies, newdata = data)  
X <- as.data.frame(X)  
y <- data$Obesity  
set.seed(123) 
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)  
X_train <- X[trainIndex, ]  
X_test <- X[-trainIndex, ]  
y_train <- y[trainIndex]  
y_test <- y[-trainIndex]  
train_glm <- cbind(X_train, Obesity = y_train)  
glm_model <- multinom(Obesity ~ ., data = train_glm) 
summary(glm_model) 
z <- summary(glm_model)$coefficients / summary(glm_model)$standard.errors  
p_values <- (1 - pnorm(abs(z), 0, 1)) * 2  
p_values  
odds_ratios <- exp(coef(glm_model))  
odds_ratios  
pred_glm <- predict(glm_model, newdata = X_test)  
confusionMatrix(pred_glm, y_test) 
library(pROC)  
library(ggplot2) 
library(tidyr) 
library(dplyr)  
glm_probs <- predict(glm_model, X_test, type = "prob") 
y_num <- as.numeric(y_test)  
roc_list <- lapply(1:nlevels(y_test), function(i) {  
multiclass.roc(as.numeric(y_test)==i, glm_probs[,i])})  
plot_data <- data.frame()  
for (i in 1:nlevels(y_test)) {
for (i in 1:nlevels(y_test)) {
roc_obj <- roc(as.numeric(y_test) == i, glm_probs[, i])
coords <- coords(roc_obj, x = "all", ret = c("specificity", "sensitivity"))
df <- data.frame(Specificity = coords$specificity,Sensitivity = coords$sensitivity,class = i)
plot_data <- rbind(plot_data, df)}
ggplot(plot_data, aes(x = 1 - Specificity, y = Sensitivity, color = factor(class))) +
geom_line(size = 1.1) +labs(title = "Multinomial GLM ROC Curves",x = "False Positive Rate",y = "True Positive Rate",color = "Class") +theme_minimal()
set.seed(123)  
rf_model <- randomForest(x = X_train,y = y_train, ntree = 500,mtry = sqrt(ncol(X_train)),importance = TRUE)
pred_rf <- predict(rf_model, X_test)
confusionMatrix(pred_rf, y_test)  
varImpPlot(rf_model)  
varImpPlot(rf_model,main = "Random Forest Variable Importance",pch = 19, col = "black")  
label_train <- as.numeric(y_train) - 1  
label_test <- as.numeric(y_test) - 1 
train_matrix <- xgb.DMatrix(data = as.matrix(X_train), label = label_train)  
test_matrix <- xgb.DMatrix(data = as.matrix(X_test), label = label_test)  
params <- list(objective = "multi:softmax",num_class = length(levels(y)),eval_metric = "mlogloss",eta = 0.1,max_depth = 6,subsample = 0.8,colsample_bytree = 0.8)  
xgb_model <- xgb.train(params = params,data = train_matrix,nrounds = 200,watchlist = list(train = train_matrix, test = test_matrix),verbose = 0)  
pred_xgb <- predict(xgb_model, newdata = test_matrix) 
pred_xgb <- factor(pred_xgb, labels = levels(y))  
confusionMatrix(pred_xgb, y_test)  
importance <- xgb.importance(model = xgb_model) 
xgb.plot.importance(importance) 
install.packages("RSNNS", dependencies = TRUE) 
install.packages("neuralnet")  
library(neuralnet)  
library(caret)  
y_train_oh <- model.matrix(~ y_train - 1)  
train_nn <- cbind(X_train, y_train_oh) 
train_nn <- as.data.frame(train_nn)  
target_names <- colnames(y_train_oh) 
formula_nn <- as.formula( paste(paste(target_names, collapse = " + "), "~", paste(colnames(X_train), collapse = " + ")))  
set.seed(123)  
nn_model <- neuralnet( formula = formula_nn, data = train_nn, hidden = c(20, 10), act.fct = "logistic",err.fct = "sse", linear.output = FALSE,lifesign = "full", stepmax = 5e6, threshold = 0.01)  
nn_pred <- compute(nn_model, X_test)$net.result  
pred_idx <- max.col(nn_pred)  
pred_labels <- factor(levels(y_train)[pred_idx], levels = levels(y_train))  
confusionMatrix(pred_labels, y_test)  
plot(nn_model)  
library(dplyr) 
library(ggplot2)  
cm <- confusionMatrix(pred_labels, y_test)$byClass  
acc_df <- data.frame(Class = rownames(cm),BalancedAccuracy = cm[, "Balanced Accuracy"]) 
ggplot(acc_df, aes(x = Class, y = BalancedAccuracy)) +geom_bar(stat = "identity", fill = "darkgreen") +coord_flip() + labs(title = "Balanced Accuracy per Class (Neural Network)",x = "Class",y = "Balanced Accuracy" ) + theme_minimal() 
library(ggplot2)  
acc_glm <- confusionMatrix(pred_glm, y_test)$overall["Accuracy"]  
acc_rf <- confusionMatrix(pred_rf, y_test)$overall["Accuracy"]  
acc_xgb <- confusionMatrix(pred_xgb, y_test)$overall["Accuracy"]  
acc_nn <- confusionMatrix(pred_labels, y_test)$overall["Accuracy"]  
accuracy_df <- data.frame( Model = c("GLM", "Random Forest", "XGBoost", "Neural Net"), Accuracy = c(acc_glm, acc_rf, acc_xgb, acc_nn) ) 
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) + geom_bar(stat = "identity", width = 0.6) + geom_text(aes(label = round(Accuracy, 3)), vjust = -0.8, size = 5) + ylim(0, 1) + labs( title = "Classification Accuracy Comparison Across Models", y = "Accuracy", x = "Model" ) + theme_minimal() + theme(legend.position = "none") 