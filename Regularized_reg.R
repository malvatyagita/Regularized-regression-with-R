
# Regularized Regression with R #

# import libraries
library(caTools)
library(psych)
library(dplyr)
library(glmnet)

# load data
url <- "https://raw.githubusercontent.com/pararawendy/dibimbing-materials/main/boston.csv"
data <- read.csv(url)

# split train - validate - test data
set.seed(123)
sample <- sample.split(data$medv, SplitRatio = .80)
pre_train <- subset(data, sample == TRUE)
sample_train <- sample.split(pre_train$medv, SplitRatio = .80)

# train-validation data
train <- subset(pre_train, sample_train == TRUE)
validation <- subset(pre_train, sample_train == FALSE)

# test data
test <- subset(data, sample == FALSE)

# Draw correlation plot on training data
pairs.panels(train, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
) 

# from the correlation plot, we can see the correlated features are 'rad' and 'tax'. 
# because 'tax' has a higher correlation with the target variable, we will drop 'rad' feature.

# feature selection - drop 'rad' feature
drop_cols <- ('rad')

train <- train %>% select(-drop_cols)
validation <-  validation %>% select(-drop_cols)
test <- test %>% select(-drop_cols)


# feature preprocessing
# to ensure we handle categorical features
x <- model.matrix(medv ~ ., train)[,-1]
y <-  train$medv

# Model Training
# Ridge regression
# Fit models on training data (lambdas = [0.01, 0.1, 1, 10])
ridge_reg_pointzeroone <- glmnet(x, y, alpha = 0, lambda = 0.01)
coef(ridge_reg_pointzeroone)

ridge_reg_pointone <- glmnet(x, y, alpha = 0, lambda = 0.1)
coef(ridge_reg_pointone)

ridge_reg_one <- glmnet(x, y, alpha = 0, lambda = 1)
coef(ridge_reg_pointone)

ridge_reg_ten <- glmnet(x, y, alpha = 0, lambda = 10)
coef(ridge_reg_ten)


# Choose the best lambda from the validation set
# Use RMSE as metric
x_validation <- model.matrix(medv ~., validation)[,-1]
y_validation <- validation$medv

RMSE_ridge_pointzeroone <- sqrt(mean((y_validation - predict(ridge_reg_pointzeroone, x_validation))^2))
RMSE_ridge_pointzeroone # 4.3464 --> best

RMSE_ridge_pointone <- sqrt(mean((y_validation - predict(ridge_reg_pointone, x_validation))^2))
RMSE_ridge_pointone # 4.349494 

RMSE_ridge_one <- sqrt(mean((y_validation - predict(ridge_reg_one, x_validation))^2))
RMSE_ridge_one # 4.422032

RMSE_ridge_ten <- sqrt(mean((y_validation - predict(ridge_reg_ten, x_validation))^2))
RMSE_ridge_ten # 5.342122
# The best lambda for ridge is 0.01

# Best ridge model's coefficients
# recall the best model --> ridge_reg_pointzeroone
coef(ridge_reg_pointzeroone)
# Interpretation of the best model ridge coefficients:
# medv = 2.807 + -7.972 crim + 3.796 zn + -4.106 indus + 2.893 chas + -1.603 nox + 4.517 rm + 5.679 age + -1.314 dis + -2.421 tax + -9.031 ptratio + 6.572 black + -4.779 lstat
# An increase of 1 point in ptratio, while the other features are kept fixed, is associated with an decrease of 9.031 point in medv

##### LASSO 
# LASSO regresion 
# # Fit models on training data (lambdas = [0.01, 0.1, 1, 10])
lasso_reg_pointzeroone <- glmnet(x, y, alpha = 1, lambda = 0.01)
coef(lasso_reg_pointzeroone)

lasso_reg_pointone <- glmnet(x, y, alpha = 1, lambda = 0.1)
coef(lasso_reg_pointone)

lasso_reg_one <- glmnet(x, y, alpha = 1, lambda = 1)
coef(lasso_reg_pointone)

lasso_reg_ten <- glmnet(x, y, alpha = 1, lambda = 10)
coef(lasso_reg_ten)


# Choose the best lambda from the validation set
# Use RMSE as metric
RMSE_lasso_pointzeroone <- sqrt(mean((y_validation - predict(lasso_reg_pointzeroone, x_validation))^2))
RMSE_lasso_pointzeroone # 4.340783 --> best

RMSE_lasso_pointone <- sqrt(mean((y_validation - predict(lasso_reg_pointone, x_validation))^2))
RMSE_lasso_pointone # 4.352728 

RMSE_lasso_one <- sqrt(mean((y_validation - predict(lasso_reg_one, x_validation))^2))
RMSE_lasso_one # 4.937774

RMSE_lasso_ten <- sqrt(mean((y_validation - predict(lasso_reg_ten, x_validation))^2))
RMSE_lasso_ten # 9.371755
# The best lambda for lasso is 0.01

# Best lasso model's coefficients
# recall the best model --> lasso_reg_pointzeroone
coef(lasso_reg_pointzeroone)
# Interpretation of the best model lasso coefficients:
# medv = 2.782 + -7.879 crim + 3.673 zn + -3.849 indus + 2.864 chas + -1.574 nox + 4.531 rm + 4.411 age + -1.294 dis + -2.439 tax + -9.039 ptratio + 6.556 black + -4.764 lstat
# An increase of 1 point in black, while the other features are kept fixed, is associated with an increase of 6.556 point in medv


## MODEL EVALUATION ##
# true evaluation on test data
# using the best model --> RMSE_ridge_pointzeroone
x_test <- model.matrix(medv ~., test)[,-1]
y_test <- test$medv


# Using the best model on test data
# Ridge
# RMSE
RMSE_ridge_best <- sqrt(mean((y_test - predict(ridge_reg_pointzeroone, x_test))^2))
RMSE_ridge_best
# Interpretation;
# RMSE_ridge_best is 6.820; so the standard deviation of prediction errors is 6.820
# The residuals mostly deviate between +- 6.820 from the regression line

# MAE
MAE_ridge_best <- mean(abs(y_test-predict(ridge_reg_pointzeroone, x_test)))
MAE_ridge_best
# Interpretation; On average, our prediction deviates the true medv by 3.896

# MAPE
MAPE_ridge_best <- mean(abs((predict(ridge_reg_pointzeroone, x_test) - y_test))/y_test)
MAPE_ridge_best
# Interpretation; That 3.896 in MAE_ridge_best is equivalent to 1.7% deviation relative to the true medv



##### LASSO
# RMSE
RMSE_lasso_best <- sqrt(mean((y_test - predict(lasso_reg_pointzeroone, x_test))^2))
RMSE_lasso_best
# Interpretation
# RMSE_ridge_best is 6.823; so the standard deviation of prediction errors is 6.823
# The residuals mostly deviate between +- 6.823 from the regression line

# MAE
MAE_lasso_best <- mean(abs(y_test-predict(lasso_reg_pointzeroone, x_test)))
MAE_lasso_best
# Interpretation; On average, our prediction deviates the true medv by 3.888

# MAPE
MAPE_lasso_best <- mean(abs((predict(lasso_reg_pointzeroone, x_test) - y_test))/y_test) 
MAPE_lasso_best
# Interpretation; That 3.888 in MAE_lasso_best is equivalent to 1.7% deviation relative to the true medv
