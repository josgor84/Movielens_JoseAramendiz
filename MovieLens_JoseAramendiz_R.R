######################################################################################
##  Here starts the initial code provided on the instruction where one have to      ##
##  Create edx set, validation set (final hold-out test set), and a submission file ##
######################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#######################################################################################
## Here Ends the code provided, and starts the development of the movielens projects ##
#######################################################################################


##############################################################
## load the required/important package ##
##############################################################

library(dslabs)
library(tidyverse)
library(caret)
library(lubridate)
library(purrr)
library(knitr)
library(ggpubr)
library(rafalib)
library(ggplot2)
library(raster)

#########################################################
## Descriptive analysis/data exploration Section       ##
#########################################################

## Analize the dimension of edx(training set) and validation (test set)

dim(validation)
dim(edx)

## Determine how many different movies and users are in the data sets

edx %>% summarize(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId))

validation %>% summarize(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId))

## Evaluate how many variables are in the data sets and determine if there are NA values

head(validation)
head(edx)

mean(is.na(edx))
mean(is.na(validation))

## Create a histogram with the data set to analize ratings behavior

# Histogram Ratings per movie: Some movies get more ratings than others

hist_movies <- edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  scale_x_log10() +
  labs( title = "Histogram of ratings per movie",
        x = "Number of ratings per movie", y = "Count", fill = element_blank()
  ) +
  theme_classic()

# Histogram Ratings per user: Some users rated movies more frequently than others

hist_users <- edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  scale_x_log10() + 
  labs(
    title = "Histogram of ratings per user",
    x = "Number of ratings per user", y = "Count", fill = element_blank()
  ) +
  theme_classic()

ggarrange(hist_movies, hist_users,
          ncol = 2, nrow = 1
)


## Compare integer rating to half-integer rating

  
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(fill = "steelblue", color = "black") +
  labs(
    title = "Histogram of ratings",
    x = "Ratings", y = "Count", fill = element_blank()
  ) +
  theme_classic()



## View the classification of genre movies inside the variable genre and the count of ratings for each genre

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


#########################################################################
## The RMSE function that will be used in this project is defined as   ##
#########################################################################


RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


###################################################################
## Here starts the Analysis Section fitting the different models ##
###################################################################

###########################################################################################
## METHOD 1: Evaluate the baseline model by calculating the average rating of all movies ##
###########################################################################################

mu_hat <- mean(edx$rating)
baseline_rmse <- RMSE(edx$rating, mu_hat)

rmse_results <- tibble(Method = "Just the rating average", RMSE = baseline_rmse)
kable(rmse_results)

#########################################################################
## METHOD 2: Evaluate the movieId bias also known as the movie effect  ##
#########################################################################


## Plot the rating variability ##.

mu_hat <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 20, data = .,  color = I("black")) +
  theme_classic()


## Fit the movie effect model ##

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- mu_hat + validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)


movie_effect_m <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Movie Bias Effect Model",
                                 RMSE = movie_effect_m ))

kable(rmse_results[2, ])
rmse_results


################################################
## METHOD 3: Evaluate the movie + user effect ##
################################################

## Plot the users rating variability ##

user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))
user_avgs %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))+
  theme_classic()

## Fit the movie + user effect model ##

user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

movie_user_effect_m <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Movie + User Effects Model",
                                 RMSE = movie_user_effect_m))

kable(rmse_results[3, ])
rmse_results

########################################################
## METHOD 4: Evaluate the movie + user + genre effect ##
########################################################


## Plot the genre rating variability ##

genres_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))
genres_avgs %>% qplot(b_g, geom ="histogram", bins = 30, data = ., color = I("black"))+
  theme_classic()

## Fit the movie + user + genre effect model ##

genres_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genres_avgs, by = c("genres")) %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  pull(pred)


movie_user_gender_model <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                                    tibble(Method="Movie + User + Gender Effects",
                                           RMSE = movie_user_gender_model))

kable(rmse_results[4, ])
rmse_results

## As one can see the effect is minimum, maybe because this model treated some genres together and not independly##


####################################################################
## METHOD 5: Evaluate the movie + user + Independent genre effect ##
####################################################################

## First, create a long version of both the train and validation datasets with separeted genres ##

edx_genres <- edx %>% separate_rows(genres, sep = "\\|", convert = TRUE)

validation_genres <- validation %>% separate_rows(genres, sep = "\\|", convert = TRUE)


## Second, Fit the model with independent genres ##

genres_ind_m <- edx_genres %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_gInd = mean(rating - mu_hat - b_i - b_u))

predicted_ratings <- validation_genres %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genres_ind_m, by = c("genres")) %>%
  mutate(pred = mu_hat + b_i + b_u + b_gInd) %>%
  pull(pred)

movie_user_ind_genre_m <- RMSE(predicted_ratings, validation_genres$rating)
rmse_results <- bind_rows(rmse_results,
                  tibble(Method="Movie + User + Genres Ind. Effects Model",
                      RMSE = movie_user_ind_genre_m))

kable(rmse_results[5, ])
rmse_results

##################################################################################################################
## METHOD 6: Apply regularization to the Movie + User + Ind. Gender Effects and fit model to the validation set ##                           
##################################################################################################################

## the regularization seeks to minimize the RMSE using cross validation to pick a lambda that optimized the RMSE ##

lambdas <- seq(0, 12.5, 0.25)

# Grid search to tune lambda #

rmses <- sapply(lambdas, function(l) {
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  b_g <- edx_genres %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + l))

## Fit model in the validation set ##  
    
   predicted_ratings <- validation_genres %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g ) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation_genres$rating))
})

plot_rmses <- qplot(lambdas, rmses,
                   main = "Regularization",
                   xlab = "lambda", ylab = "RMSE") 

plot_rmses # lambda vs RMSE

lambda <- lambdas[which.min(rmses)]
lambda # lambda which optimizes the model (minimizes RMSE) which is 5.25 in this case


rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Regularized Movie + User + Genre Ind Effect Model",
                                 RMSE = min(rmses)))
 
kable(rmse_results[6, ])

# Results section summary ##

kable(rmse_results)




