
# This is the data analysis codes for the summer project
# Created by Diheng Zhang (dihengzhang@email.arizona.edu)
# Created at Jun 25, 2019

install.packages("lintr")
library("lintr")
sessionInfo()
devtools::install_github("jimhester/lintr")
library(dplyr)
library(data.table)

trials_testing <- read.csv('../testTrial_balance', sep = '\t')
head(trials_testing)

negative_trials <- trials_testing %>% filter(X.Trial < 60)
positive_trials <- trials_testing %>% filter((X.Trial >= 60)&(X.Trial < 120))
neutral_trials <- trials_testing %>% filter(X.Trial >= 120)

# Mean cosDif
negative_Dif <- mean(negative_trials$X.CosDiff)
positive_Dif <- mean(positive_trials$X.CosDiff)
neutral_Dif <- mean(neutral_trials$X.CosDiff)

# SUM SSE
negative_Dif <- sum(negative_trials$X.SSE)
positive_Dif <- sum(positive_trials$X.SSE)
neutral_Dif <- sum(neutral_trials$X.SSE)

# plot
result <- c(negative_Dif,positive_Dif,neutral_Dif)
barplot(result, main = "Balance Input Projection (1=Ne, 2=Po, 3=Neutral)",ylab = 'SUM SSE of testing')

barplot(result, main = "Inbalance Input Projection (1=Ne, 2=Po, 3=Neutral)",ylab='SUM SSE of testing')
