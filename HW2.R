# Question 1 
set.seed(0)

source("/Users/larali/ECON460/data files/read_onlinespending.R")

library(gamlr)

# 1a 
lasso_model <- gamlr(xweb, log(yspend), verb = TRUE)
cv_lasso <- cv.gamlr(xweb, log(yspend))
cv_lasso$lambda.min
coef_lasso <- coef(cv_lasso, select = "min")
nonzero_idx <- which(coef_lasso[-1] != 0)
nonzero_idx

#1b
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
