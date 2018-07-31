rm(list=ls(all=TRUE))
setwd("C:/Insofe/capstone")

#####Train Data ####
master_data <- read.csv('train.csv', header = TRUE)
test_data <- read.csv('test.csv', header = TRUE)
names(test_data)

dim(master_data)
summary(master_data)
str(master_data)
sum(is.na(master_data))



target <- data.frame(master_data$Activity) # extraced target variable
dim(target)
data <- data.frame(master_data[1:562]) # extracted data for pca
dim(data)
summary(data)
str(target)

summary(target)
plot(target)
names(target)

test_target <- data.frame(test_data$Activity) # extraced target variable
test_data <- data.frame(test_data[1:562])
dim(test_target)
library(vegan)
pca_data <- decostand(data, method = "range") # normalized data
test_pca_data <- decostand(test_data, method = "range") # normalized data

#####Applying PCA for dimentionality reduction###############
pca_model <- princomp(~ ., data=pca_data,scale. = T)

test_pca_model <- predict(pca_model, newdata = test_pca_data)
test_pca_model <- as.data.frame(test_pca_model)
summary(pca_model)
plot(pca_model, type = "l")
names(pca_model)
pca_model$scores[1:10,1:6]
pca_model$loadings[1:10,1:6]


library(ggfortify)
autoplot(pca_model, data = master_data, colour='Activity', main = "PCA Visualization")


pca_components <- data.frame(pca_model$scores[,1:25])
test_pca_components <- data.frame(test_pca_model[,1:25])
pca_components
test_pca_components
names(pca_components)
dim(pca_components)
dim(test_pca_components)
dim(test_target)
newdata <- cbind(pca_components,target)# combied pca data and target variable
newtestdata <- cbind(test_pca_components,test_target)
names(newdata)
dim(newtestdata)
tail(newtestdata)
dim(newdata)
rows <- seq(1,nrow(newdata),1)
rows
set.seed(100)
trainrows_PCA <- sample(rows, nrow(newdata)*0.7)
train_data_PCA <- newdata[trainrows,]
validate_data_PCA <- newdata[-trainrows,]
names(train_data_PCA)

library(C50)

#####Decision tree model on PCA data###############
decisionTree <- C5.0(master_data.Activity~.,data=newdata)
require(mgcv)
save(decisionTree, file = 'models/PCA_DecisionTree.rda')

summary(decisionTree)

pred_train = predict(decisionTree,newdata=train_data_PCA, type="class")
pred_validate = predict(decisionTree, newdata=validate_data_PCA, type="class")
pred_test = predict(decisionTree, newdata=newtestdata, type="class")
tail(newtestdata)

pred_train_table <- table(pred_train, train_data$master_data.Activity)
pred_train_table

pred_validate_table <- table(pred_validate, validate_data$master_data.Activity)
pred_validate_table

pred_test_table <- table(pred_test, newtestdata$test_data.Activity)
pred_test_table

accuracy_pred_train_table <- sum(diag(pred_train_table))/sum(pred_train_table)*100
accuracy_pred_train_table

accuracy_pred_validate_table <- sum(diag(pred_validate_table))/sum(pred_validate_table)*100
accuracy_pred_validate_table #### accuray 96.37353 on validate data

accuracy_pred_test_table <- sum(diag(pred_test_table))/sum(pred_test_table)*100
accuracy_pred_test_table #### PCA data is giving 73.80387 accuracy 


install.packages("e1071")
library(e1071)

SVM_model = svm(master_data.Activity ~. , data=train_data_PCA, method = "C-classification", kernel = "linear", cost = 10,gamma = 0.1)

save(SVM_model, file = 'models/PCA_svm.rda')
load("models/PCA_svm.rda")

summary(SVM_model)
library(ggplot2)


pred_SVM_train = predict(SVM_model,newdata=train_data_PCA, type="class")
pred_SVM_validate = predict(SVM_model, newdata=validate_data_PCA, type="class")
pred_SVM_test = predict(SVM_model, newdata=newtestdata, type="class")

table(pred_SVM_test, newtestdata$test_data.Activity)


SVM_train_Conf_Matrix = table(train_data_PCA$master_data.Activity,pred_SVM_train);SVM_train_Conf_Matrix

#c. Confusion Matrix on validate Data
SVM_validate_Conf_Matrix = table(validate_data$Activity,pred_SVM_validate);SVM_validate_Conf_Matrix

SVM_test_Conf_Matrix = table(newtestdata$test_data.Activity,pred_SVM_test);SVM_test_Conf_Matrix

#d. Compute the evaluation metric
accuracy_SVM_train = round((sum(diag(SVM_train_Conf_Matrix))/sum(SVM_train_Conf_Matrix))* 100,2)

accuracy_SVM_validate = round((sum(diag(SVM_validate_Conf_Matrix))/sum(SVM_validate_Conf_Matrix))*100,2)

accuracy_SVM_test = round((sum(diag(SVM_test_Conf_Matrix))/sum(SVM_test_Conf_Matrix))*100,2)

accuracy_SVM_train
accuracy_SVM_validate
accuracy_SVM_test #### PCA data giving accuracy 75.77 on SVM


######KNN Classificaiton -- PCA data ############
library(class)

pcadata_train_withoutclass = subset(train_data_PCA,select=-c(master_data.Activity))
pcadata_validate_withoutclass = subset(validate_data_PCA,select=-c(master_data.Activity))
pcadata_test_withoutclass = subset(newtestdata,select=-c(test_data.Activity))


knn_pred_validate = knn(pcadata_train_withoutclass, pcadata_validate_withoutclass, train_data$Activity, k = 5)


knn_pred_test = knn(pcadata_train_withoutclass, pcadata_test_withoutclass, train_data$Activity, k = 5)

summary(knn_pred_validate)
table(knn_pred_validate)
a=table(knn_pred_validate,validate_data$Activity)
a
b=table(knn_pred_test,newtestdata$test_data.Activity)
b

knn_validate_accu= sum(diag(a))/nrow(pcadata_validate_withoutclass)*100
knn_test_accu= sum(diag(b))/nrow(pcadata_test_withoutclass)*100
knn_validate_accu
knn_test_accu ##KNN is giving about 82.79 accuracy on test data
######KNN Classificaiton -- PCA data ############
