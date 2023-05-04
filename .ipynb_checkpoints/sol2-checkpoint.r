library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(stringr)
library(corrplot)
library(car)
library(caret)
library(torch)
library(nnet)
library(broom)



# Question 2
# Perform logistic regression

url <- "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df <- read_csv(url) %>% 
mutate_if(\(x) is.character(x), as.factor) %>% 
mutate(y = as.factor(Survived)) %>% 
select(-c(Name, Survived)) %>% 
(\(x) {names(x) <- tolower(names(x)); x})
df


full_model <- glm(y ~ ., df, family=binomial())
summary(full_model)
full_pred <- predict(full_model, df, type = "response") > 0.5

step_model <- step(full_model)
summary(step_model)
step_pred <- predict(step_model, df, type = "response") > 0.5

# Lasso
controls <- trainControl(method = "repeatedcv", number = 5, repeats=10)
lasso_fit <- train(
  x = model.matrix(full_model)[, -1],
  y = df$y %>% as.factor(),
  method = "glmnet",
  trControl = controls, 
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = 2^seq(-20, 0, length.out = 40)
    ),
  family = "binomial"
)
plot(lasso_fit$results$lambda %>% log2, lasso_fit$results$Accuracy, type="b", col="blue")
abline(v=lasso_fit$bestTune$lambda %>% log2, col="red")
lasso_fit$bestTune$lambda

lasso_pred <- predict(lasso_fit, model.matrix(full_model)[, -1], type="raw") == "1"


# NNet
x <- model.matrix(full_model)[, -1] %>% torch_tensor(dtype=torch_float())
y <- (df$y == "1") %>% torch_tensor(dtype=torch_float())

module <- nn_module(
  initialize = function() {
    self$fc1 <- nn_linear(m, 1)
    self$fc2 <- nn_sigmoid()
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      self$fc2()
  }
)

Loss <- function(x, y, model){
  nn_bce_loss()(model(x), y)
}

penalty <- function(model){
  sum(abs(model$parameters$fc1.weight))
}

fit <- function(f, lambda, n=2000, lr=0.01){
  optimizer <- optim_adam(f$parameters, lr=lr)

  for (i in 1:n){
    loss <- Loss(x, y, f) + lambda * penalty(f)
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()

    if (i < 10 || i %% 100 == 0){
      cat(sprintf("Step: %d, Loss: %.4f\n", i, loss$item()))
    }
  }
  return(f)
}

if (T){
  set.seed(123)
  f <- module()
  g <- fit(f, 1e-4, 2000, lr=0.01)
}

nn_pred <- (g(x) %>% as_array) > 0.5


overview <- function(predicted, expected){
    accuracy <- mean(predicted == expected)
    error <- 1 - accuracy
    fp <- sum(predicted & !expected)
    tp <- sum(predicted & expected)
    fn <- sum(!predicted & expected)
    tn <- sum(!predicted & !expected)
    false_positive_rate <- fp / (fp + tn)
    false_negative_rate <- fn / (fn + tp)
    return(
        data.frame(
            accuracy = accuracy, 
            error=error, 
            false_positive_rate = false_positive_rate, 
            false_negative_rate = false_negative_rate
        )
    )
}

list(full_pred, step_pred, lasso_pred, nn_pred) %>% 
lapply(\(x) overview(x, df$y == "1")) %>% 
bind_rows()

length(df$y == "1")
length(full_pred)
