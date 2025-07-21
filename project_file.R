#include at least one quantitative variable that can be predicted
#include at least one qualitative variable that can be classified
#describe the variables in the dataset

#link to dataset: https://www.kaggle.com/datasets/nikhil1e9/loan-default

library(ggplot2) # For making data viz
library(dplyr) # Data manipulation
library(janitor) # clean names
library(tidymodels) #package for ML in R
library(rpart.plot)
library(pdp)
library(vip)
library(baguette)
library(forcats)

#first load in the dataset
fd <- read.csv("Loan_default.csv") |> 
  clean_names() |> 
  mutate(has_mortgage = as.factor(has_mortgage),
         has_co_signer = as.factor(has_co_signer),
         has_dependents = as.factor(has_dependents),
         default = as.factor(default),
         education = as.factor(education),
         employment_type = as.factor(employment_type),
         marital_status = as.factor(marital_status),
         loan_purpose = as.factor(loan_purpose))

fd <- subset(fd, select = -loan_id)

#next, split the data set into training and test set
set.seed(173)
fd_split <- rsample::initial_split(fd, prop = .7, strata = default)
fd_train <- rsample::training(fd_split)
fd_test <- rsample::testing(fd_split)


##QUALITATIVE DATA PREDICTION
#question: what variables are most important in predicting loan default?

#setup the model to be piped into a logistic regression
loan_default_setup <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

#fit the model to all vars
loan_default_fit <- loan_default_setup |> 
  fit(default ~ ., data = fd_train)

#determine the variables with the most importance to loan default and show visual
loan_default_fit$fit |> 
  vip(aesthetics = list(fill = "#6e0000", col = "black")) +
  theme_bw()

loan_default_train_pred <- loan_default_fit |> 
  augment(fd_train)

loan_default_train_pred |> 
  conf_mat(truth = default, estimate = .pred_class)

my_class_metrics <- metric_set(accuracy, sensitivity, specificity, precision)

loan_default_train_pred |> 
  my_class_metrics(truth = default, estimate = .pred_class, event_level = "second")

#now testing for overfitting

loan_default_test_pred <- loan_default_fit |> 
  augment(fd_test)

loan_default_test_pred |> 
  conf_mat(truth = default, estimate = .pred_class)

loan_default_test_pred |> 
  my_class_metrics(truth = default, estimate = .pred_class, event_level = "second")

##QUANTITATIVE DATA PREDICTION
#question: what variables are most important in determining a person's debt to income ratio (dti)?
#first, fit training to a regression tree to determine most important vars to determine debt to income ratio

dti_reg <- linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression")

dti_reg_fit <- dti_reg |> 
  fit(dti_ratio ~., data = fd_train)

pred_dti_fit <- dti_reg_fit |> 
  augment(fd_train)

my_reg_metrics <- metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq)  

pred_dti_fit |> 
  my_reg_metrics(truth = dti_ratio, estimate = .pred)

pred_dti_fit_test <- dti_reg_fit |> 
  augment(fd_test)

pred_dti_fit_test |> 
  my_reg_metrics(truth = dti_ratio, estimate = .pred)

#////
#now implementing bagging to do predictions -- compare this to the regular linear regression model later

## QUALITATIVE PREDICTION
#predicting default
set.seed(8)
loan_default_bagging_setup <- bag_tree(tree_depth = 5) |> 
  set_engine("rpart", times = 20) |> 
  set_mode("classification")

loan_default_bagging_fit <- loan_default_bagging_setup |> 
  fit(default ~ ., data = fd_train)

# loan_default_bagging_fit
loan_default_var_impotance_df <- loan_default_bagging_fit$fit$imp |> 
  arrange(desc(-value)) |> 
  mutate(term = factor(term, term))

p <- ggplot(data = loan_default_var_impotance_df,
            aes(x = term, y = value))

p + geom_col(fill = "6e0000") +
  labs(x = "variable",
       y = "importance") +
  theme_bw() +
  coord_flip()

loan_default_pred_bagging_train <- loan_default_bagging_fit |> 
  augment(fd_train)

loan_default_pred_bagging_train |> 
  conf_mat(truth = default, estimate = .pred_class)

loan_default_pred_bagging_train |> 
  my_class_metrics(truth = default, estimate = .pred_class, event_level = "second")

loan_default_pred_bagging_test <- loan_default_bagging_fit |> 
  augment(fd_test)

loan_default_pred_bagging_test |> 
  conf_mat(truth = default, estimate = .pred_class)

loan_default_pred_bagging_test |> 
  my_class_metrics(truth = default, estimate = .pred_class, event_level = "second")

set.seed(9)
loan_default_rf_setup <- rand_forest(mtry = 3, trees = 200) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

loan_default_rf_fit <- loan_default_rf_setup |> 
  fit(default ~ ., data = fd_train)

# loan_default_rf_fit

vip(loan_default_rf_fit,
    aesthetics = list(fill = "6e0000", col = "black"))

loan_default_pred_rf_train <- loan_default_rf_fit |> 
  augment(fd_train)

loan_default_pred_rf_train |> 
  conf_mat(truth = default, estimate = .pred_class)

loan_default_pred_rf_train |> 
  my_class_metrics(truth = default, estimate = .pred_class, event_level = "second")

loan_default_pred_rf_test <- loan_default_rf_fit |>  
  augment(fd_test)

loan_default_pred_rf_test |> 
  conf_mat(truth = default, estimate = .pred_class)

loan_default_pred_rf_test |> 
  my_class_metrics(truth = default, estimate = .pred_class, event_level = "second")

## QUANTITATIVE PREDICTION
#predicting dti_ratio using bagging

