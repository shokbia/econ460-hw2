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

library(Matrix)
library(gamlr)

# random seed
set.seed(0)

# daata preparation (from dong's read_onlinespending.R)
browser_spend <- read.csv("C:\\Users\\rawre\\Downloads\\browser-totalspend.csv")
yspend <- browser_spend$spend

web <- read.csv("C:\\Users\\rawre\\Downloads\\browser-domains.csv")
sitenames <- scan("C:\\Users\\rawre\\Downloads\\browser-sites.txt", what="character")
web$site <- factor(web$site, levels=1:length(sitenames), labels=sitenames)
web$id <- factor(web$id, levels=1:length(unique(web$id)))

machinetotals <- as.vector(tapply(web$visits, web$id, sum))
visitpercent <- 100 * web$visits / machinetotals[web$id]

xweb <- sparseMatrix(i = as.numeric(web$id), 
                     j = as.numeric(web$site), 
                     x = visitpercent,
                     dims = c(nlevels(web$id), nlevels(web$site)),
                     dimnames = list(id = levels(web$id), site = levels(web$site)))

# log spending variable
log_yspend <- log(yspend)

# 2a 
#Sample split 8000 for estimation, 2000 for holdout
n <- 8000
m <- 2000
sample_indices <- sample.int(length(log_yspend), n, replace = FALSE)

# Split data
xweb_est <- xweb[sample_indices, ]
log_yspend_est <- log_yspend[sample_indices]

xweb_holdout <- xweb[-sample_indices, ]
log_yspend_holdout <- log_yspend[-sample_indices]

# 5-fold cross-validated lasso on estimation sample
cat("2a running 5-fold cross-validated lasso on estimation sample...\n")
cv_lasso_est <- cv.gamlr(xweb_est, log_yspend_est, nfold = 5, verb = TRUE)

# Plot out-of-sample cross validation error
plot(cv_lasso_est, main = "Out-of-Sample CV Error vs Lambda")

# 2b in-sample prediction error
lambda_min <- cv_lasso_est$lambda.min
lasso_min <- gamlr(xweb_est, log_yspend_est, lambdas = lambda_min)

# predictions for estimation sample
pred_est <- predict(lasso_min, xweb_est, type = "response")

# MSE in-sample prediction error
in_sample_error <- mean((log_yspend_est - pred_est)^2)
cat("2b. In-sample prediction error:", in_sample_error, "\n")

# 2c oos prediction error
# use model trained on estimation sample to predict holdout sample
pred_holdout <- predict(lasso_min, xweb_holdout, type = "response")

# calc mse oos prediction error
out_of_sample_error <- mean((log_yspend_holdout - pred_holdout)^2)
cat("2c. Out-of-sample prediction error using holdout:", out_of_sample_error, "\n")

# Comparison
cat("Comparisons..... \n")
cat("In-sample MSE:", in_sample_error, "\n")
cat("Out-of-sample MSE:", out_of_sample_error, "\n")
cat("Ratio (OOS/IS):", out_of_sample_error / in_sample_error, "\n")
cat("The out-of-sample error is", 
    round((out_of_sample_error/in_sample_error - 1)*100, 1), 
    "% higher than the in-sample error.\n")
