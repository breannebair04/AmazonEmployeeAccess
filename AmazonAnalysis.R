#Wrangling Data
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(ggplot2)
library(dplyr)
library(embed)

traindata <- vroom("~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/train.csv") %>%
  mutate(ACTION = factor(ACTION))

ggplot(data = traindata, aes(x = RESOURCE)) +
  geom_histogram()


my_recipe <- recipe(ACTION ~ ., data = traindata) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other( all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize()%>%
  step_pca(all_predictors, threshold = .8)




prep <- prep(my_recipe)
baked <- bake(prep, new_data = traindata)
dim(baked)



# Logistic Regression

library(tidymodels)

testdata <- vroom("~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/test.csv.zip")


logRegModel <- logistic_reg() %>% 
  set_engine("glm")


amazon_wf <- workflow() %>%
  add_model(logRegModel) %>%
  add_recipe(my_recipe)

amazon_fit <- amazon_wf %>%
  fit(data = traindata)

amazon_predictions <- predict(amazon_fit,
                              new_data=testdata,
                              type= "prob") 

amazon_predictions <- predict(amazon_fit, new_data = testdata, type = "prob") %>%
  select(.pred_1)

submission <- testdata %>%
  select(id) %>%
  bind_cols(amazon_predictions) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "logreg.csv")



#Penalized Regression

library(tidymodels)

my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% 
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod)


tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) 

folds <- vfold_cv(traindata, v = 5, repeats=1)
CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
select_best(metric = "roc_auc")


final_wf <-
amazon_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=traindata)


final_preds <- predict(final_wf, new_data = testdata, type="prob")%>%
  select(.pred_1)



submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "penlogreg.csv")



#Random Forests

my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)



tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),   
  min_n(range = c(2, 10)),
  levels = 5)


folds <- vfold_cv(traindata, v = 5, repeats=1)
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")


final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=traindata)


final_preds <- predict(final_wf, new_data = testdata, type="prob")%>%
  select(.pred_1)



submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/ranforest.csv")



#KNN

library(tidymodels)

knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(knn_model)


set.seed(123)
folds <- vfold_cv(traindata, v = 5)

knn_grid <- grid_regular(
  neighbors(range = c(1, 25)),  
  levels = 10
)


knn_results <- knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = knn_grid,
    metrics = metric_set(roc_auc, accuracy)
  )


best_knn <- knn_results %>%
  select_best(metric = "roc_auc")


final_knn_wf <- knn_wf %>%
  finalize_workflow(best_knn)


final_knn_fit <- final_knn_wf %>%
  fit(data = traindata)


final_preds <- predict(final_knn_fit, new_data = testdata, type = "prob")%>%
  select(.pred_1)

submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/knn.csv")


#Naive Bayes

library(tidymodels)
library(discrim)
library(naivebayes)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") 

nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)


tuning_grid <- grid_regular(
  smoothness(),   
  Laplace(),
  levels = 5)


folds <- vfold_cv(traindata, v = 5, repeats=1)
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")


final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=traindata)


final_preds <- predict(final_wf, new_data = testdata, type = "prob") %>%
  select(.pred_1)


submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/NBayes.csv")

#NN

install.packages("remotes")
remotes::install_github("rstudio/tensorflow")

reticulate::install_python()

keras::install_keras()


nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 250 
) %>%
set_engine("keras") %>% 
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)),
                            levels=5)
nn_folds <- vfold_cv(traindata, v=5, repeats=1)

nn_results <- tune_grid(
  nn_wf,
  resamples = nn_folds,
  grid = nn_tuneGrid,
  metrics = metric_set(roc_auc)
)

collect_metrics() %>%
filter(.metric=="roc_auc") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()


#PCR


my_recipe <- recipe(ACTION ~ ., data = traindata) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other( all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.93) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = traindata)
dim(baked)



    #logreg redo

logRegModel <- logistic_reg() %>% 
  set_engine("glm")


amazon_wf <- workflow() %>%
  add_model(logRegModel) %>%
  add_recipe(my_recipe)

amazon_fit <- amazon_wf %>%
  fit(data = traindata)

amazon_predictions <- predict(amazon_fit,
                              new_data=testdata,
                              type= "prob") 

amazon_predictions <- predict(amazon_fit, new_data = testdata, type = "prob") %>%
  select(.pred_1)

submission <- testdata %>%
  select(id) %>%
  bind_cols(amazon_predictions) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/PCRlogreg.csv")

    #Random Forest redo
my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)



tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),   
  min_n(range = c(2, 10)),
  levels = 5)


folds <- vfold_cv(traindata, v = 5, repeats=1)
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")


final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=traindata)


final_preds <- predict(final_wf, new_data = testdata, type="prob")%>%
  select(.pred_1)



submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/PCRranforest.csv")


    #KNN redo
knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)


set.seed(123)
folds <- vfold_cv(traindata, v = 5)

knn_grid <- grid_regular(
  neighbors(range = c(1, 25)),  
  levels = 10
)


knn_results <- knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = knn_grid,
    metrics = metric_set(roc_auc, accuracy)
  )


best_knn <- knn_results %>%
  select_best(metric = "roc_auc")


final_knn_wf <- knn_wf %>%
  finalize_workflow(best_knn)


final_knn_fit <- final_knn_wf %>%
  fit(data = traindata)


final_preds <- predict(final_knn_fit, new_data = testdata, type = "prob")%>%
  select(.pred_1)

submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/PCRknn.csv")


#SVM

library(tidymodels)

my_recipe <- recipe(ACTION ~ ., data = traindata) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other( all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.8) 

## SVM models
svmRadial <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

amazon_wf <- workflow() %>%
  add_model(svmLinear) %>%
  add_recipe(my_recipe)%>%
  fit(data=traindata)



amazon_predictions <- predict(amazon_wf, new_data = testdata, type = "prob") %>%
  select(.pred_1)

submission <- testdata %>%
  select(id) %>%
  bind_cols(amazon_predictions) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/SVMLinear.csv")


#Balancing Data

library(tidymodels)
library(themis)

my_recipe <- recipe(ACTION ~ ., data = traindata) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other( all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize() %>%
  step_smote(all_outcomes(), neighbors=4)

prepped_recipe <- prep(my_recipe)
baked <- bake(prep, new_data = traindata)


my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

# 

tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),   
  min_n(range = c(2, 10)),
  levels = 5)


folds <- vfold_cv(traindata, v = 5, repeats=1)
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")


final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=traindata)


final_preds <- predict(final_wf, new_data = testdata, type="prob")%>%
  select(.pred_1)



submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/SMOTEranforest.csv")




#Final Try


#factor steplinecode mix, normalize, 100trees, 5 folds

traindata <- vroom("~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/train.csv") %>%
  mutate(ACTION = factor(ACTION))

my_recipe <- recipe(ACTION ~ ., data = traindata) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = traindata)
dim(baked)

testdata <- vroom("~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/test.csv.zip")


my_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)



tuning_grid <- grid_regular(
  mtry(range = c(1, 20)),   
  min_n(range = c(1, 20)),
  trees(range = c(300, 1000)),  
  levels = 5)


folds <- vfold_cv(traindata, v = 10, repeats=1)
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")


final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=traindata)


final_preds <- predict(final_wf, new_data = testdata, type="prob")%>%
  select(.pred_1)



submission <- testdata %>%
  select(id) %>%
  bind_cols(final_preds) %>%
  rename(ACTION = .pred_1)

write_csv(submission, "~/Documents/Fall 2025/Stat 348/AmazonEmployeeAccess/FINALranforest.csv")



