# Question 1 
set.seed(0)

source("/Users/larali/ECON460/data files/read_onlinespending.R")

library(gamlr)

# 1a, 1b, 1c 
#lasso_model <- gamlr(xweb, log(yspend), verb = TRUE)
cv_lasso <- cv.gamlr(xweb, log(yspend))
cv_lasso$lambda.min
coef_lasso <- coef(cv_lasso, select = "min")
nonzero_idx <- which(coef_lasso[-1] != 0)
nonzero_idx


n <- length(yspend)
boot_idx <- sample(1:n, size=n, replace = TRUE)

yspend_boot <- yspend[boot_idx]
xweb_boot <- xweb[boot_idx, ]

lasso_model_boot <- gamlr(xweb_boot, log(yspend_boot), verb = TRUE)
cv_lasso_boot <- cv.gamlr(xweb_boot, log(yspend_boot))
cv_lasso_boot$lambda.min
coef_lasso_boot <- coef(cv_lasso_boot, select = "min")
nonzero_idx_boot <- which(coef_lasso_boot[-1] != 0)
nonzero_idx_boot

setdiff(nonzero_idx_boot, nonzero_idx)
setdiff(nonzero_idx, nonzero_idx_boot)
intersect(nonzero_idx, nonzero_idx_boot)

#1d 
library(glmnet)

# get sequence of candidate lambdas
lambda_start = 0.232476092 
lambda_seq = lambda_start * 0.9545485^(0:99)

# split data randomly into K folds 
K <- 5
fold_id <- sample(rep(1:K,length.out = nrow(xweb)))

# estimate B for each lambda on training set, compute OOS dev on test set, average dev over K
CV_OOS <- rep(NA,length(lambda_seq))

for (i in 1:length(lambda_seq)){
  CV_OOS_i <- rep(NA,K)
  
  for(k in 1:K){
    train_idx <- which(fold_id != k) #train set index
    test_idx <- which(fold_id == k)
    
    fit <- glmnet(xweb[train_idx,],log(yspend[train_idx]), lambda = lambda_seq[i], standardize = TRUE)
    pred <- predict(fit, newx = xweb[test_idx,],s=lambda_seq[i])
    

    CV_OOS_i[k] <- mean((log(yspend[test_idx])- pred)^2) 
  }
  CV_OOS[i] <- mean(CV_OOS_i)
}

# choosing lambda with lowest deviance 
best_lambda <- lambda_seq[which.min(CV_OOS)]
best_lambda

# re-run lasso with lambda penalty 
manual_lasso <- glmnet(xweb, log(yspend), lambda = best_lambda, standardize = TRUE)
coef_manual_lasso <- coef(manual_lasso, s = best_lambda)
nonzero_idx_manual <- which(coef_manual_lasso[-1] != 0)
nonzero_idx_manual



## QUESTION 1 CHLOE
set.seed(0)
setwd("~/Downloads/Fall2025")
source("read_onlinespending.R")
library(gamlr)
set.seed(0)


ylog <- log(yspend)
cv.lasso <- cv.gamlr(xweb, ylog, verb=TRUE)
lambda_chosen <- cv.lasso$lambda.min
lambda_chosen


coef_lasso <- coef(cv.lasso, s="min") 
nonzero_idx_original <- which(coef_lasso[-1] != 0) 
nonzero_idx_original
