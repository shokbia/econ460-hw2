
library(Matrix)
set.seed(0)
library(gamlr)

#2a) 
xweb <- sparseMatrix(
  i = as.numeric(web$id),
  j = as.numeric(web$site),
  x = visitpercent,
  dims = c(nlevels(web$id), nlevels(web$site)),
  dimnames = list(id = levels(web$id), site = levels(web$site))
)

#sanity chwck
stopifnot(length(yspend) == nrow(xweb))
cat("Built xweb with", nrow(xweb), "households ×", ncol(xweb), "sites\n")
summary(yspend)

# Train / holdout split
n_est <- 8000L
all_idx <- seq_len(nrow(xweb))
est_idx <- sample(all_idx, n_est, replace = FALSE)
hold_idx <- setdiff(all_idx, est_idx)

x_est <- xweb[est_idx, ]
y_est <- yspend[est_idx]

x_hold <- xweb[hold_idx, ]
y_hold <- yspend[hold_idx]

# 5-fold cross-validated lasso on estimation sample 
cv.lasso_model <- cv.gamlr(
  x = x_est,
  y = log(y_est),
  nfold = 5,
  verb = TRUE   
)

# plot: OOS CV error vs lambda 
png("cv_lasso_oos_error.png", width = 1100, height = 800, res = 140)
plot(cv.lasso_model)  # produces the U-shaped curve with two dotted lines
dev.off()

# report the key numbers for writeup
c(
  lambda_min = cv.lasso_model$lambda.min,
  lambda_1se = cv.lasso_model$lambda.1se,
  min_cv_err = min(cv.lasso_model$cvm)    # average OOS deviance at lambda_min
)

#QUESTION 2

install.packages("gamlr")
library(gamlr)

library(Matrix)


set.seed(0)

# Load yspend
spend_raw <- read.csv("/Users/baur/Downloads/browser-totalspend.csv")
yspend <- spend_raw$spend

# 3. Load the long table (browser-domains.csv)
web_raw <- read.csv("/Users/baur/Downloads/browser-domains.csv")

# Identify column names automatically
names(web_raw) <- tolower(names(web_raw))
id_col    <- grep("^(id|hh|household|uid)$", names(web_raw), value = TRUE)
site_col  <- grep("^(site|domain|url|host)$", names(web_raw), value = TRUE)
visit_col <- grep("(visit.*percent|percent|share|visits?)$", names(web_raw), value = TRUE)

# Keep the right columns
web <- web_raw[, c(id_col, site_col, visit_col)]
names(web) <- c("id", "site", "visitpercent")

# -------------------------

#2a) 
#Convert to factors and sparse matrix
web$id   <- factor(web$id)
web$site <- factor(web$site)

xweb <- sparseMatrix(
  i = as.numeric(web$id),
  j = as.numeric(web$site),
  x = web$visitpercent,
  dims = c(nlevels(web$id), nlevels(web$site)),
  dimnames = list(id = levels(web$id), site = levels(web$site))
)

#sanity chwck
stopifnot(length(yspend) == nrow(xweb))
cat("Built xweb with", nrow(xweb), "households ×", ncol(xweb), "sites\n")
summary(yspend)

# Train / holdout split
n_est <- 8000L
all_idx <- seq_len(nrow(xweb))
est_idx <- sample(all_idx, n_est, replace = FALSE)
hold_idx <- setdiff(all_idx, est_idx)

x_est <- xweb[est_idx, ]
y_est <- yspend[est_idx]

x_hold <- xweb[hold_idx, ]
y_hold <- yspend[hold_idx]

# 5-fold cross-validated lasso on estimation sample 
cv.lasso_model <- cv.gamlr(
  x = x_est,
  y = log(y_est),
  nfold = 5,
  verb = TRUE   # prints per-fold progress like in the slides
)

# Plot: OOS CV error vs lambda 
png("cv_lasso_oos_error.png", width = 1100, height = 800, res = 140)
plot(cv.lasso_model)  # produces the U-shaped curve with two dotted lines
dev.off()

# report the key numbers for writeup
c(
  lambda_min = cv.lasso_model$lambda.min,
  lambda_1se = cv.lasso_model$lambda.1se,
  min_cv_err = min(cv.lasso_model$cvm)    # average OOS deviance at lambda_min
)
#2b
lambda_min <- cv.lasso_model$lambda.min

# In-sample fitted values
yhat_in <- predict(cv.lasso_model, newdata = x_est, select = "min")

# in-sample prediction error
in_sample_mse <- mean((log(y_est) - yhat_in)^2)

cat("In-sample prediction MSE:", round(in_sample_mse, 5), "\n")
cat("Lambda_min used:", lambda_min, "\n")
