# ChadModel------------------------------------------------------------------------------
# script to predict first ball attack results from input conditions
# 
# basic flow is
#   1. Prepare dataset
#     1.Read in a DV data set
#     2. Locate first ball attack rows in data
#     3. Extract input factors from the serve, receive, set, and attack rows
#     4. Combine factors into single row of new dataframe
#
#   2. Model
#     1. Build and fit a classifier model
#     2. Evaluate accuracy of predictions
#     3. See which input factors have most impact on predictions

library(tidyverse)
library(zoo)
library(caret)
# clean up workspace
rm(list = ls())

ModelType = 'Regression'
# ModelType = 'Classification'

# load and wrangle-------------------------------------------------------------
# Load and massage dataset

# match_example <- read.csv("datavolleyexample.csv")
match_example <- read.csv("master.csv", stringsAsFactors =TRUE)

# locate the first ball attack sequences
attack_sequence <- factor(c('Serve', 'Reception', 'Set', 'Attack'), levels(match_example$skill))
attack_indeces <- which(rollapply(match_example$skill, length(attack_sequence), identical, attack_sequence))

# create new dataframes containing metadata and key factors from each skill

# serve
serve_factors <- select(match_example[attack_indeces,],
                        team,
                        skill_type, 
                        evaluation_code, 
                        start_coordinate_x,
                        start_coordinate_y,
                        end_coordinate_x,
                        end_coordinate_y)

serve_factors$skill_type <- droplevels(serve_factors$skill_type)
colnames(serve_factors) <- c('opponent', 'serve_type', 'serve_quality', 'serve_start_x',
                             'serve_start_y', 'serve_end_x', 'serve_end_y')
serve_factors$index <- attack_indeces
rownames(serve_factors) <- attack_indeces

#receive
receive_factors <- select(match_example[attack_indeces+1,], 
                          evaluation_code, 
                          skill_subtype)

receive_factors$skill_subtype <- droplevels(receive_factors$skill_subtype)
colnames(receive_factors) <- c('receive_quality', 'receive_skill_subtype')
receive_factors$index <- attack_indeces
rownames(receive_factors) <- attack_indeces


# set
set_factors <- select(match_example[attack_indeces+2,], 
                      skill_type, 
                      evaluation_code,
                      start_coordinate_x,
                      start_coordinate_y)
                      
set_factors$skill_type <- droplevels(set_factors$skill_type)
colnames(set_factors) <- c('set_type', 'set_quality', 'set_start_x', 'set_start_y')
set_factors$index <- attack_indeces
rownames(set_factors) <- attack_indeces


# from the attack row, grab both metadata and factors
attack_metadata <- select(match_example[attack_indeces+3,], 
                          team,
                          player_name)
attack_metadata$index <- attack_indeces
rownames(attack_metadata) <- attack_indeces

# attack factors
attack_factors <- select(match_example[attack_indeces+3,], 
                      skill_type, 
                      skill_subtype,
                      evaluation_code,
                      num_players)

attack_factors$skill_subtype <- droplevels(attack_factors$skill_subtype)
attack_factors$skill_type <- droplevels(attack_factors$skill_type)
attack_factors$num_players <- droplevels(attack_factors$num_players)

colnames(attack_factors) <- c('attack_type', 'attack_subtype', 'attack_quality',
                              'attack_block_type')
attack_factors$index <- attack_indeces
rownames(attack_factors) <- attack_indeces

# merge skills-----------------------------------------------------------------
# merge all these skills together
all_factors <- inner_join(serve_factors, receive_factors, by=c('index'))
all_factors <- inner_join(all_factors, set_factors, by=c('index'))
all_factors <- inner_join(all_factors, attack_factors, by=c('index'))
rownames(all_factors) <- all_factors$index


# map quality codes------------------------------------------------------------
# convert quality codes to numeric ranking
all_factors$serve_quality <- recode(all_factors$serve_quality, 
                                    '=' = 0, 
                                    '-' = 1,  
                                    '!' = 2, 
                                    '+' = 3, 
                                    '/' = 4,
                                    '#' = 5)

all_factors$receive_quality <- recode(all_factors$receive_quality, 
                                    '=' = 0,
                                    '/' = 1,
                                    '-' = 2,  
                                    '!' = 3, 
                                    '+' = 4, 
                                    '#' = 5)

all_factors$set_quality <- recode(all_factors$set_quality, 
                                      '=' = 0,
                                      '/' = 1,
                                      '-' = 2,  
                                      '!' = 3, 
                                      '+' = 4, 
                                      '#' = 5)

all_factors$attack_quality <- recode(all_factors$attack_quality,
                                     '=' = -1,
                                     '/' = -1,
                                     '-' = 0,
                                     '!' = 0,
                                     '+' = 0,
                                     '#' = 1)


all_factors$attack_qual_fact <- recode_factor(all_factors$attack_quality,
                                  '-1' = 'Error',
                                  '0' = 'In Play',
                                  '1' = 'Kill')

# drop tracking variables
all_factors <- all_factors[, !colnames(all_factors) %in% c('opponent')]


if (ModelType == 'Classification') {
  # group coordinates by 10s
  coord_cols <- c('serve_start_x', 'serve_start_y', 'serve_end_x', 'serve_end_y',
                 'set_start_x', 'set_start_y')
  all_factors[coord_cols] <- lapply(all_factors[coord_cols], function(x) round(x/10) * 10)
}

rm('match_example', 'serve_factors', 'receive_factors', 'set_factors', 'attack_factors')

# MODEL ---------------------------------------------------------------------
# MODELING
# to support generic models, input data needs:
# 1. Convert all factors into one-hot variables
# 2. Scale mode variables to near mean=0 std=1
#
# These steps are not needed for tree algorithms


# Now run best mode (XGBoost) on partial or full dataset
sampleSize = round(0.1 * nrow(all_factors))
# sampleSize = nrow(all_factors)
sampleRows <- sample(rownames(all_factors), sampleSize)


all_factors[sampleRows,] %>% 
  ggplot(aes(serve_start_y)) +
  geom_histogram()

all_cols = colnames(all_factors)
fact_cols = grepl('type|fact', all_cols)

plot_fact_column = function (data, column) {
  ggplot(data, aes_string(x = column)) +
    geom_histogram(stat = 'count') +
    xlab(column)
}

plot_data_column = function (data, column) {
  ggplot(data, aes_string(x = column)) +
    geom_histogram() +
    xlab(column)
}

fact_plots <- lapply(all_cols[fact_cols], plot_fact_column, data = all_factors[sampleRows,])
data_plots <- lapply(all_cols[!fact_cols], plot_data_column, data = all_factors[sampleRows,])

require(gridExtra)
do.call("grid.arrange", c(fact_plots, ncol=4))
do.call("grid.arrange", c(data_plots, ncol=4))



trainControl <- trainControl(method="LGOCV", p=0.75, number=5,
                             verboseIter = TRUE)

if (ModelType == 'Regression') {
  # parameters to avoid over fit
  xgbGrid <-  expand.grid(nrounds = 200,
                          max_depth = 2,
                          eta = 0.1,
                          gamma = 1,
                          colsample_bytree = 1,
                          min_child_weight = 1,
                          subsample = 0.5)
} else {
  # Best parameters for prediction using a slight over fit
  xgbGrid <-  expand.grid(nrounds = 2000,
                          max_depth = 8,
                          eta = 0.3,
                          gamma = 1,
                          colsample_bytree = 1,
                          min_child_weight = 1,
                          subsample = 0.5)
}

# Allow fit to choose best parameters
# xgbGrid <-  expand.grid(nrounds = c(100, 200, 500, 1000),
#                         max_depth = c(2, 4, 6),
#                         eta = c(0.01, 0.1, 0.3),
#                         gamma = 1,
#                         colsample_bytree = 1,
#                         min_child_weight = 1,
#                         subsample = c(0.5, 0.75, 1))

if (ModelType == 'Regression') {
  X <- all_factors[, !colnames(all_factors) %in% c('index',
                                                   'blocking_team',
                                                   'blocking_team_wr_score',
                                                   'attack_qual_fact')]
  
  model_xgb_all <- train(attack_quality ~ .,
                         data=X[sampleRows,],
                         method='xgbTree',
                         trControl=trainControl,
                         tuneGrid=xgbGrid,
                         nthread = 2,
                         na.action = na.pass)
} else {
  X <- all_factors[, !colnames(all_factors) %in% c('index',
                                                   'blocking_team',
                                                   'blocking_team_wr_score',
                                                   'attack_quality')]
  
  model_xgb_all <- train(attack_qual_fact ~ ., 
                         data=X[sampleRows,], 
                         method='xgbTree',
                         trControl=trainControl,
                         tuneGrid=xgbGrid,
                         nthread = 2,
                         na.action = na.pass)
}


results_xgb_all <- inner_join(attack_metadata[sampleRows,], all_factors[sampleRows,], by=c('index'))
rownames(results_xgb_all) <- results_xgb_all$index

results_xgb_all$pred <- predict(model_xgb_all, X[sampleRows,], na.action=na.pass)

if (ModelType == 'Regression') {
  results_xgb_all$Model <- round(results_xgb_all$pred)
  results_xgb_all$pred_error <- results_xgb_all$attack_quality - results_xgb_all$pred
} else {
  results_xgb_all$Model <- recode(results_xgb_all$pred,
                                  'Error' = -1,
                                  'In Play' = 0,
                                  'Kill' = 1)
  results_xgb_all$pred_error <- results_xgb_all$attack_quality - results_xgb_all$Model
}
results_xgb_all <- results_xgb_all[with(results_xgb_all, order(index)),]

# basic analysis ----------------------------------------------------------------
# plotting model results
results_xgb_all %>%
  pivot_longer(., c('attack_quality', 'Model')) %>%
  ggplot(aes(value, group=name, fill=name)) +
  geom_histogram(binwidth=1, center=0, position='dodge') + 
  ggtitle('XG Boost')

# For regression, add a comparison against raw prediction values
if (ModelType == 'Regression') {
  results_xgb_all %>%
    pivot_longer(., c('attack_quality', 'pred')) %>%
    ggplot(aes(value, group=name, fill=name)) +
    geom_histogram(binwidth=0.1, center=0, position='dodge') +
    ggtitle('XG Boost')
}

# print out input factor importance
plot(varImp(model_xgb_all), top=10, main='XGB Variable Importance')


# us <- filter(results_xgb_all, player_name %in% 
#          c('Aaron Russell',
#            'David Smith',
#            'Matt Anderson',
#            'Maxwell Holt',
#            'Taylor Sander')) %>%
#   rename(., Actual = attack_quality) %>% 
#   pivot_longer(., c('Actual', 'Model'), 
#                names_to='Source', 
#                values_to='Attack_Quality')
# us$player_name <- droplevels(us$player_name)


plot_by_factor <- function(us, plot_factor) {

    temp <- us %>%
      group_by_at(c('player_name', plot_factor, 'Source', 'Attack_Quality')) %>%
      tally() %>%
      pivot_wider(names_from = Attack_Quality, names_prefix = 'X', values_from = n) %>%
      replace_na(list('X-1' = 0,
                      'X0' = 0,
                      'X1' = 0))
  
  temp$attempts <- rowSums(temp[,c('X-1', 'X0', 'X1')])
  temp$Eff <- with(temp, (X1 - `X-1`)/attempts)
  
  filter(temp, attempts >= 10) %>%
    ggplot(aes_string(plot_factor, 'Eff', fill='Source')) +
    geom_col(position='dodge') +
    facet_grid(vars(player_name))
}

# plot_by_factor(us = us, plot_factor = 'attack_set_type')
# plot_by_factor(us = us, plot_factor = 'attack_block_type')
# plot_by_factor(us = us, plot_factor = 'set_description')


# Local sensitivities -----------------------------------------------------------
# LIME
library(lime)

# need to create custom model_type function
model_type.train <- function(x, ...) {
  # Function tells lime() what model type we are dealing with
  # 'classification', 'regression', 'survival', 'clustering', 'multilabel', etc
  
  if (ModelType == 'Regression') {
    return("regression")
  } else {
    return("classification")
  }
}

model_type(model_xgb_all)

# need to create custom predict_model function
predict_model.train <- function(x, newdata, ...) {
  # Function performs prediction and returns data frame with Response
  if (ModelType == 'Regression') {
    pred <- predict(model_xgb_all, newdata, na.action=na.pass)
    return(as.data.frame(pred))
  } else {
    pred <- predict(model_xgb_all, newdata, na.action=na.pass, type = 'prob')
    return(pred)
  }
}


xgb_expl <- lime(X, model_xgb_all)

plot_explaination <- function(att_name, att_set_type, block_team, 
                              model, inputs, results, explainer) {
  # find test cases in data set that match inputs  
  match_ix <- results_xgb_all %>%
    subset((player_name == att_name & 
              attack_set_type == att_set_type & 
              blocking_team == block_team)) %>%
    rownames()
  
  # save the attackers team
  team <- results[match_ix[1], 'team']
  
  # sort the test cases by outcomes
  match_ix <- match_ix[order(results[match_ix, 'attack_qual_fact'])]
  test_rows = inputs[match_ix,]
  
  # get explanations for test cases
  if (ModelType == 'Regression') {
    test_expl <- explain(test_rows, explainer, n_features = 6)
    actual_outcome <- results[match_ix, 'attack_qual_fact']
    model_outcome <- predict_model(model, newdata = test_rows)$pred
  } else {
    test_expl <- explain(test_rows, explainer, 
                         n_features = 8, 
                         labels = 'Kill',
                         dist_fun = "manhattan",
                         kernel_width = 2,
                         # feature_select = "lasso_path"
    )
    actual_outcome <- results[match_ix, 'attack_qual_fact']
    model_outcome <- predict_model(model, newdata = test_rows)[,3]
  }
  # first plot is all attacks on one chart  
  expl_plt <- plot_explanations(test_expl)
  print(expl_plt + 
    scale_x_discrete(limits = match_ix,
                     labels = paste(actual_outcome, ' (', 
                                    format(model_outcome, digits = 1), ')')) +
    ggtitle(paste(team, att_name, att_set_type, 'vs', block_team)) + 
    xlab('Result (prob of Kill)'))
  
  #next plot details of 1 each Error, In Play, Kill
  
  # change the labels to actual outcomes
  test_expl[test_expl$case %in% match_ix[actual_outcome == 'Error'],'label'] <- 'Error'
  test_expl[test_expl$case %in% match_ix[actual_outcome == 'In Play'],'label'] <- 'In Play'

  ex_att_ix <- c(match('Error', actual_outcome),
                 match('In Play', actual_outcome),
                 match('Kill', actual_outcome))
  
  print(plot_features(filter(test_expl, case %in% match_ix[ex_att_ix])))
}

plot_explaination('Matt Anderson', 'Red', 'Brazil',
                  model_xgb_all, X, results_xgb_all, xgb_expl)

plot_explaination('Wallace De Souza', 'Red', 'USA',
                  model_xgb_all, X, results_xgb_all, xgb_expl)

plot_explaination('Bartosz Kurek', 'Red', 'USA',
                  model_xgb_all, X, results_xgb_all, xgb_expl)


                     
# global sensitivities -----------------------------------------------------------
# Partial Dependence Plots
library(pdp)

# Plot XY factor sensitivity
p1 <- partial(model_xgb_all, pred.var = 'serve_start_x', chull = TRUE, prob = TRUE, which.class = 3)
p2 <- partial(model_xgb_all, pred.var = 'serve_end_x', chull = TRUE, prob = TRUE, which.class = 3)
p3 <- partial(model_xgb_all, pred.var = 'set_start_x', chull = TRUE, prob = TRUE, which.class = 3)

names(p1) <- c('Coord', 'Change in Prob')
p1$input <- 'serve_start'
p1$Direction <- 'X'
names(p2) <- c('Coord', 'Change in Prob')
p2$input <- 'serve_end'
p2$Direction <- 'X'
names(p3) <- c('Coord', 'Change in Prob')
p3$input <- 'set_start'
p3$Direction <- 'X'

xy_sens <- rbind(p1, p2, p3)

p1 <- partial(model_xgb_all, pred.var = 'serve_start_y', chull = TRUE, prob = TRUE, which.class = 3)
p2 <- partial(model_xgb_all, pred.var = 'serve_end_y', chull = TRUE, prob = TRUE, which.class = 3)
p3 <- partial(model_xgb_all, pred.var = 'set_start_y', chull = TRUE, prob = TRUE, which.class = 3)

names(p1) <- c('Coord', 'Change in Prob')
p1$input <- 'serve_start'
p1$Direction <- 'Y'
names(p2) <- c('Coord', 'Change in Prob')
p2$input <- 'serve_end'
p2$Direction <- 'Y'
names(p3) <- c('Coord', 'Change in Prob')
p3$input <- 'set_start'
p3$Direction <- 'Y'

xy_sens <- rbind(xy_sens, p1, p2, p3)
ggplot(xy_sens, aes(Coord, `Change in Prob`, color=input)) + 
  geom_line() + 
  facet_grid(cols = vars(Direction))


# plot quality factor sensitivities
p1 <- partial(model_xgb_all, pred.var = 'serve_quality', chull = TRUE, prob = TRUE, which.class = 3)
p2 <- partial(model_xgb_all, pred.var = 'receive_quality', chull = TRUE, prob = TRUE, which.class = 3)
p3 <- partial(model_xgb_all, pred.var = 'set_quality', chull = TRUE, prob = TRUE, which.class = 3)

names(p1) <- c('Quality', 'Change in Prob')
p1$input <- 'serve_quality'
names(p2) <- c('Quality', 'Change in Prob')
p2$input <- 'receive_quality'
names(p3) <- c('Quality', 'Change in Prob')
p3$input <- 'set_quality'

quality_sens <- rbind(p1, p2, p3)
ggplot(quality_sens, aes(Quality, `Change in Prob`, fill=input)) + 
  geom_bar(stat='identity', position='dodge')
  
# plot quality factor sensitivities
p1 <- partial(model_xgb_all, pred.var = 'blocking_team_rank', chull = TRUE, prob = TRUE, which.class = 3)
p2 <- partial(model_xgb_all, pred.var = 'set_description', chull = TRUE, prob = TRUE, which.class = 3)
p3 <- partial(model_xgb_all, pred.var = 'attack_set_type', chull = TRUE, prob = TRUE, which.class = 3)
p4 <- partial(model_xgb_all, pred.var = 'attack_block_type', chull = TRUE, prob = TRUE, which.class = 3)

names(p1)[2] <- 'Change in Prob'
names(p2)[2] <- 'Change in Prob'
names(p3)[2] <- 'Change in Prob'
names(p4)[2] <- 'Change in Prob'

g1 <- ggplot(p1, aes(blocking_team_rank, `Change in Prob`, fill = `Change in Prob`)) + 
  geom_bar(stat='identity', position='dodge') +
  scale_fill_gradient2(midpoint = 0.5) +
  theme(legend.position = "none")
g2 <- ggplot(p2, aes(set_description, `Change in Prob`, fill = `Change in Prob`)) + 
  geom_bar(stat='identity', position='dodge') +
  scale_fill_gradient2(midpoint = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        legend.position = "none")
g3 <- ggplot(p3, aes(attack_set_type, `Change in Prob`, fill = `Change in Prob`)) + 
  geom_bar(stat='identity', position='dodge') +
  scale_fill_gradient2(midpoint = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        legend.position = "none")
g4 <- ggplot(p4, aes(attack_block_type, `Change in Prob`, fill = `Change in Prob`)) + 
  geom_bar(stat='identity', position='dodge') +
  scale_fill_gradient2(midpoint = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        legend.position = "none")

require(gridExtra)
grid.arrange(g1, g2, g3, g4)
