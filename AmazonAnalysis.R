#Wrangling Data
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(ggplot2)
library(dplyr)
library(embed)

traindata <- vroom("train.csv") %>%
  mutate(ACTION = factor(ACTION))

ggplot(data = traindata, aes(x = RESOURCE)) +
  geom_histogram()


my_recipe <- recipe(ACTION ~ ., data = traindata) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other( all_nominal_predictors(), threshold = 0.001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 




prep <- prep(my_recipe)
baked <- bake(prep, new_data = traindata)
dim(baked)



# Logistic Regression

library(tidymodels)

testdata <- vroom("test.csv.zip")


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










#ssh beb0901@stat-u02.byu.edu

