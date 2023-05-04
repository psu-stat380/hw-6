packages <- c(
  "dplyr", 
  "readr", 
  "tidyr", 
  "purrr", 
  "broom",
  "magrittr",
  "corrplot",
  "caret",
  "rpart",
  "rpart.plot",
  "e1071",
  "torch", 
  "luz"
)

# renv::install(packages)
sapply(packages, require, character.only=T)


# Q1
filename <- "data/housing.csv"
df <- read_csv(filename, col_types = cols())

df <- df %>% 
  mutate_if(\(x) is.character(x), as.factor) %>% 
  drop_na()

# Q2

df %>% 
  select(where(is.numeric)) %>% 
  cor() %>% 
  round(digits=2) %>% 
  corrplot(diag=F)


# Q3

set.seed(42)
test_ind <- sample(
  1:nrow(df), 
  floor( nrow(df)/10 ),
  replace=FALSE
)

df_train <- df[-test_ind, ]
df_test <- df[test_ind, ]

# Q4
lm_fit <- lm(median_house_value ~ ., df_train)
summary(lm_fit)


# Q5
rmse <- function(y, yhat) {
  sqrt(mean((y - yhat)^2))
}

lm_predictions <- predict(reg_fit, df_test)
rmse(
  df_test$median_house_value, 
  lm_predictions
)


# Q6
rpart_fit <- rpart(median_house_value ~ ., df_train)


rpart_predictions <- predict(rpart_fit, df_test)
rmse(
  df_test$median_house_value, 
  rpart_predictions
)

# Q7
svm_fit <- svm(median_house_value ~ ., df_train, kernel="radial")

svm_predictions <- predict(svm_fit, df_test)
rmse(
  df_test$median_house_value, 
  svm_predictions
)


# Q8
NNet <- nn_module(
  initialize = function(p, q1, q2, q3){
    self$linear1 <- nn_linear(p, q1)
    self$linear2 <- nn_linear(q1, q2)
    self$linear3 <- nn_linear(q2, q3)
    self$output <- nn_linear(q3, 1)
    self$activation <- nn_relu()
  },
  forward = function(x){
    x %>% self$linear1() %>% self$activation() %>% 
      self$linear2() %>% self$activation() %>% 
      self$linear3() %>% self$activation() %>%
      self$output()
  }
)

M <- model.matrix(median_house_value ~ 0 + ., df)

fit_nn <- NNet %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_rmse()
    )
  ) %>% 
  set_hparams(p = ncol(M), q1 = 312, q2 = 116, q3 = 18) %>%
  set_opt_hparams(lr = 1e-2) %>%
  fit(
    data = list(
      model.matrix(median_house_value ~ 0 + ., df_train),
      df_train$median_house_value %>% as.matrix
    ),
    valid_data = list(
      model.matrix(median_house_value ~ 0 + ., df_test),
      df_test$median_house_value %>% as.matrix
    ),
    epochs = 100,
    verbose = TRUE,
    dataloader_options = list(batch_size = 512)
  )


nnet_predictions <- predict(
  fit_nn, 
  model.matrix(median_house_value ~ 0 + ., df_test)
) %>% as_array

rmse(
  df_test$median_house_value, 
  nnet_predictions
)

# Q10

# Summarize your results in a table comparing the RMSE for the different models. 
predictions <- sapply(
  list(
    lm_predictions, 
    rpart_predictions, 
    svm_predictions, 
    nnet_predictions
  ),
  function(x) {
    rmse(df_test$median_house_value, x)
  }
) %>% as.data.frame
rownames(predictions) <- c("lm", "rpart", "svm", "nnet")
colnames(predictions) <- c("rmse")

predictions %>%
  select(rmse) %>% 
  round(digits=2)




# Question 2

Spam email classification

# Q1

filename <- "data/spambase.csv"
df <- read_csv(filename, col_types = cols())

df <- df %>% 
  mutate_if(\(x) is.character(x), as.factor) %>% 
  drop_na()

# Q2

set.seed(42)
test_ind <- sample(
  1:nrow(df), 
  floor( nrow(df)/10 ),
  replace=FALSE
)

df_train <- df[-test_ind, ]
df_test <- df[test_ind, ]


overview <- function(true_class, pred_class){
  accuracy <- mean(true_class == pred_class)
  error <- 1 - accuracy
  true_positives <- sum(true_class == 1 & pred_class == 1)
  true_negatives <- sum(true_class == 0 & pred_class == 0)
  false_positives <- sum(true_class == 0 & pred_class == 1)
  false_negatives <- sum(true_class == 1 & pred_class == 0)
  false_positive_rate <- false_positives / (true_positives + false_negatives)
  false_negative_rate <- false_negatives / (true_positives + false_negatives)
  return(
    data.frame(
      accuracy = accuracy,
      error = error,
      false_positive_rate = false_positive_rate,
      false_negative_rate = false_negative_rate
    )
  )
}

# Q3

glm_fit <- glm(spam ~ ., df_train, family=binomial())
summary(glm_fit)

glm_predictions <- predict(glm_fit, df_test, type="response") > 0.5
overview(df_test$spam, glm_predictions)


# Q4

rpart_fit <- rpart(spam ~ ., df_train, method="class")
rpart_predictions <- predict(rpart_fit, df_test, type="class") == "1"

overview(df_test$spam, rpart_predictions)

# Q5

svm_fit <- svm(spam ~ ., df_train, kernel="radial", type="C-classification")
svm_predictions <- predict(svm_fit, df_test) == "1"

overview(df_test$spam, svm_predictions)

# Q6

NNet <- nn_module(
  initialize = function(p, q1, q2, q3){
    self$linear1 <- nn_linear(p, q1)
    self$linear2 <- nn_linear(q1, q2)
    self$linear3 <- nn_linear(q2, q3)
    self$output <- nn_linear(q3, 1)
    self$activation <- nn_relu()
    self$sigmoid <- nn_sigmoid()
  },
  forward = function(x){
    x %>% self$linear1() %>% self$activation() %>% 
      self$linear2() %>% self$activation() %>% 
      self$linear3() %>% self$activation() %>%
      self$output() %>% self$sigmoid()
  }
)

M <- model.matrix(spam ~ 0 + ., df)

fit_nn <- NNet %>% 
  setup(
    loss = nn_bce_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>% 
  set_hparams(p = ncol(M), q1 = 64, q2 = 32, q3 = 16) %>%
  set_opt_hparams(lr = 1e-3) %>%
  fit(
    data = list(
      model.matrix(spam ~ 0 + ., df_train),
      df_train$spam %>% as.matrix
    ),
    valid_data = list(
      model.matrix(spam ~ 0 + ., df_test),
      df_test$spam %>% as.matrix
    ),
    epochs = 100,
    verbose = TRUE,
    dataloader_options = list(batch_size = 256, shuffle=TRUE)
  )

plot(fit_nn)

nnet_predictions <- predict(
  fit_nn, 
  model.matrix(spam ~ 0 + ., df_test)
) %>% as_array > 0.5

overview(df_test$spam, nnet_predictions)

# Q7

# Summarize your results in a table comparing the accuracy for the different models.

list(
  glm_predictions, 
  rpart_predictions, 
  svm_predictions, 
  nnet_predictions
) %>%
  lapply(\(x) overview(df_test$spam, x)) %>%
  bind_rows() %>%
  set_rownames(c("glm", "rpart", "svm", "nnet")) %>%
  knitr::kable()



# Question 3

# Q1

generate_two_spirals <- function(){
  set.seed(42)
  n <- 500
  noise <- 0.2
  t <- (1:n) / n * 2 * pi
  x1 <- c(
      t * (sin(t) + rnorm(n, 0, noise)),
      t * (sin(t + 2 * pi/3) + rnorm(n, 0, noise)),
      t * (sin(t + 4 * pi/3) + rnorm(n, 0, noise))
    )
  x2 <- c(
      t * (cos(t) + rnorm(n, 0, noise)),
      t * (cos(t + 2 * pi/3) + rnorm(n, 0, noise)),
      t * (cos(t + 4 * pi/3) + rnorm(n, 0, noise))
    )
  y <- as.factor(
    c(
      rep(0, n), 
      rep(1, n), 
      rep(2, n)
    )
  )
  return(tibble(x1=x1, x2=x2, y=y))
}

df <- generate_two_spirals()
plot(df$x1, df$x2, col=as.factor(df$y), pch=20)




# Q2

grid <- expand.grid(
    x1 = seq(-10, 10, length.out=100),
    x2 = seq(-10, 10, length.out=100)
  )

df_test <- as_tibble(grid)


# Q3

plot_decision_boundary <- function(predictions){
  plot(
    df_test$x1, df_test$x2, 
    col = predictions,
    pch = 0
  )
  points(
    df$x1, df$x2,
    col = df$y,
    pch = 20
  )
}


rpart_fit <- rpart(y ~ ., df, method="class")
rpart_classes <- predict(rpart_fit, df_test, type="class")
plot_decision_boundary(rpart_classes)

# Q5

svm_fit <- svm(y ~ ., df, kernel="radial", type="C-classification")
svm_classes <- predict(svm_fit, df_test)
plot_decision_boundary(svm_classes)


# Q3

out_dim <- 3
test_matrix <- df_test %>% select(x1, x2) %>% as.matrix


NN1 <- nn_module(
  initialize = function(p, q1, o){
    self$linear1 <- nn_linear(p, q1)
    self$output <- nn_linear(q1, o)
    self$activation <- nn_relu()
  },
  forward = function(x){
    x %>% 
      self$linear1() %>% 
      self$activation() %>% 
      self$output()
  }
)

NN2 <- nn_module(
  initialize = function(p, q1, q2, o){
    self$linear1 <- nn_linear(p, q1)
    self$linear2 <- nn_linear(q1, q2)
    self$output <- nn_linear(q2, o)
    self$activation <- nn_relu()
  },
  forward = function(x){
    x %>% 
      self$linear1() %>% 
      self$activation() %>% 
      self$linear2() %>% 
      self$activation() %>% 
      self$output()
  }
)

NN3 <- nn_module(
  initialize = function(p, q1, q2, q3, o){
    self$linear1 <- nn_linear(p, q1)
    self$linear2 <- nn_linear(q1, q2)
    self$linear3 <- nn_linear(q2, q3)
    self$output <- nn_linear(q3, o)
    self$activation <- nn_relu()
  },
  forward = function(x){
    x %>% 
      self$linear1() %>% 
      self$activation() %>% 
      self$linear2() %>% 
      self$activation() %>% 
      self$linear3() %>% 
      self$activation() %>% 
      self$output()
  }
)

NN4 <- nn_module(
  initialize = function(p, q1, q2, q3, q4, o){
    self$linear1 <- nn_linear(p, q1)
    self$linear2 <- nn_linear(q1, q2)
    self$linear3 <- nn_linear(q2, q3)
    self$linear4 <- nn_linear(q3, q4)
    self$output <- nn_linear(q4, o)
    self$activation <- nn_relu()
  },
  forward = function(x){
    x %>% 
      self$linear1() %>% 
      self$activation() %>% 
      self$linear2() %>% 
      self$activation() %>% 
      self$linear3() %>% 
      self$activation() %>% 
      self$linear4() %>% 
      self$activation() %>% 
      self$output()
  }
)


#########################
# Q4

fit_nn1 <- NN1 %>% 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_hparams(p=2, q1=10, o=out_dim) %>% 
  set_opt_hparams(lr=0.01) %>%
  fit(
    data = list(
      df %>% select(x1, x2) %>% as.matrix,
      df$y %>% as.integer
    ),
    dataloader_options = list(batch_size = 256, shuffle=TRUE),
    epochs = 100,
    verbose = TRUE
  )


plot_decision_boundary(
  predict(fit_nn1, test_matrix) %>% 
      torch_argmax(2) %>% 
      as_array
)


#########################
# Q5

fit_nn2 <- NN2 %>% 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_hparams(p=2, q1=10, q2=10, o=out_dim) %>% 
  set_opt_hparams(lr=0.01) %>%
  fit(
    data = list(
      df %>% select(x1, x2) %>% as.matrix,
      df$y %>% as.integer
    ),
    dataloader_options = list(batch_size = 256, shuffle=TRUE),
    epochs = 100,
    verbose = TRUE
  )


plot_decision_boundary(
  predict(fit_nn2, test_matrix) %>% 
      torch_argmax(2) %>% 
      as_array
)


#########################
# Q3

fit_nn3 <- NN3 %>% 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_hparams(p=2, q1=10, q2=10, q3=10, o=out_dim) %>% 
  set_opt_hparams(lr=0.005) %>%
  fit(
    data = list(
      df %>% select(x1, x2) %>% as.matrix,
      df$y %>% as.integer
    ),
    dataloader_options = list(batch_size = 256, shuffle=TRUE),
    epochs = 100,
    verbose = TRUE
  )


plot_decision_boundary(
  predict(fit_nn3, test_matrix) %>% 
      torch_argmax(2) %>% 
      as_array
)



#########################
# Q3

fit_nn4 <- NN4 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
  ) %>% 
  set_hparams(p=2, q1=10, q2=10, q3=10, q4=10, o=out_dim) %>% 
  set_opt_hparams(lr=0.01) %>%
  fit(
    data = list(
      df %>% select(x1, x2) %>% as.matrix,
      df$y %>% as.integer
    ),
    dataloader_options = list(batch_size = 256, shuffle=TRUE),
    epochs = 100,
    verbose = TRUE
  )


plot_decision_boundary(
  predict(fit_nn4, test_matrix) %>% 
      torch_argmax(2) %>% 
      as_array
)
