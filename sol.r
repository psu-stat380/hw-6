packages <- c(
  "tibble",
  "dplyr", 
  "readr", 
  "tidyr", 
  "purrr", 
  "broom",
  "magrittr",
  "corrplot",
  "car"
)

# renv::install(packages)
sapply(packages, require, character.only=T)

df <- read_csv("data/spending.csv")

par(mfrow=c(6, 7))
for(x in colnames(df)){
    plot(
      df[["income"]] ~ df[[x]],
      xlab = x, ylab = "income"
    )
}

model1 <- lm(y ~ ., df)

model1 %>% summary()
model1 %>% vif()


df_x %>%
  cor() %>%
  corrplot()


pca <- df %>%
  select(-y) %>%
  princomp(cor = T)

pca$loadings
plot(pca, type="l")

loadings <- ifelse(
  abs(pca$loadings[, 1:4]) < 0.1,
  0,
  round(pca$loadings[, 1:4], 2)
) %>%
  as.data.frame()
rownames(loadings) <- colnames(df)
heatmap(loadings %>% as.matrix(), Colv = NA)

df_z <- predict(pca, x) %>%
  as_tibble() %>%
  select(1:4) %>%
  mutate(income = df[["income"]])

model2 <- lm(y ~ ., df_z)
model2 %>% summary
model2 %>% vif