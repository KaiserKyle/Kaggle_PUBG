library(Metrics)
library(dplyr)
library(ranger)
library(BBmisc)
library(e1071)
library(tidyverse)
library(xgboost)

minMaxScale <- function(vect) {
  maxValue <- max(vect)
  minValue <- min(vect)
  
  scaled <- (vect - minValue) / (maxValue - minValue)
  
  return(scaled)
}

notFactor <- function(thing) {
  return(!is.factor(thing))
}

diff <- function(thing) {
    return(max(thing) - min(thing))
}

scaleSoloData <- function(data, train) {
  print("Scaling Solo Data")
  data <- data %>% group_by(groupId) %>% mutate(groupMembers = n())
  data <- data %>% group_by(matchId) %>% mutate(avgTeamSize = mean(groupMembers))
  data <- data %>% group_by(matchId) %>% mutate(killPlace_minMax = minMaxScale(killPlace), killStreak_scaled = scale(killStreaks), boosts_scaled = scale(boosts),
                                                kills_scaled = scale(kills), rideDistance_scaled = scale(rideDistance), weaponsAcquired_scaled = scale(weaponsAcquired))
                                                
  data$walkDistance_scaled <- data$walkDistance / data$matchDuration
  
  if (train) {
    data <- subset(data, select = c(Id, groupId, matchId, matchType, maxPlace, winPlacePerc, groupMembers, avgTeamSize, numGroups, matchDuration, mapNumber, killPlace_minMax, killStreak_scaled, boosts_scaled, kills_scaled, rideDistance_scaled, weaponsAcquired_scaled, walkDistance_scaled, headshotRate, healthItems, damageOverTime, killStreakRatio))
  } else {
    data <- subset(data, select = c(Id, groupId, matchId, matchType, maxPlace, groupMembers, avgTeamSize, numGroups, matchDuration, mapNumber, killPlace_minMax, killStreak_scaled, boosts_scaled, kills_scaled, rideDistance_scaled, weaponsAcquired_scaled, walkDistance_scaled, headshotRate, healthItems, damageOverTime, killStreakRatio))
  }
  
  return (data)
}

scaleDatasetByGroup <- function(data) {
  print("Per group operations")
  data <- data %>% group_by(groupId) %>% mutate_if(notFactor, funs(sum = sum, max = max))
  data <- data %>% group_by(groupId) %>% mutate(killPlace_min = min(killPlace), kills_min = min(kills), longestKill_min = min(longestKill), walkDistance_min = min(walkDistance))

  data <- subset(data, select = -c(killStreaks, headshotKills, assists, heals, weaponsAcquired, DBNOs, boosts, swimDistance, participation, headshotRate, healthItems, damageOverTime, longestKill))
  data <- subset(data, select = -c(headshotRate_max, killStreaks_max, assists_max, headshotKills_max, headshotKills_sum, DBNOs_sum, swimDistance_max))
  data[is.na(data)] <- 0
  
  return (data)
}

scaleDatasetByMatch <- function(data) {
  print("Per match operations")
  data <- data %>% group_by(groupId) %>% mutate(groupMembers = n())
  data <- data %>% group_by(matchId) %>% mutate_if(notFactor, funs(scaled = scale))
  data <- data %>% group_by(matchId) %>% mutate(matchCount = n(), avgTeamSize = mean(groupMembers)) 

  data <- subset(data, select = -c(killStreaks, headshotKills, assists, heals, weaponsAcquired, DBNOs, boosts, swimDistance, participation, headshotRate, healthItems, damageOverTime, longestKill))
  data[is.na(data)] <- 0
  
  return (data)
}

runModel <- function(trainData, testData, perGroup, modelEta) {
  if (perGroup) {
    trainData <- subset(trainData, select = -c(winPlacePerc_sum,  winPlacePerc_max)) 
    testData <- subset(testData, select = -c(winPlacePerc_sum, winPlacePerc_max)) 
    trainData <- subset(trainData, select = -c(winPlacePerc_scaled))
    testData <- subset(testData, select = -c(winPlacePerc_scaled))
  }
  trainLabels <- trainData$winPlacePerc
  testLabels <- testData$winPlacePerc
  trainData <- subset(trainData, select = -c(Id, groupId, matchId, matchType, winPlacePerc))
  testData <- subset(testData, select = -c(Id, groupId, matchId, matchType, winPlacePerc))
  dTrain = xgb.DMatrix(data = as.matrix(trainData), label = trainLabels)
  dTest = xgb.DMatrix(data = as.matrix(testData), label = testLabels)
  
  watchList = list(train=dTrain, test=dTest)
  
  print("Training XGBoost")
  model <- xgb.train(data = dTrain, max.depth = 9, eta = modelEta, min_child_weight = 2, nthread = 2, nround = 100000, watchlist = watchList, objective = "reg:linear", early_stopping_rounds = 50, print_every_n = 50, eval_metric = "mae")
  
  return(model)
}

predictOutcome <- function(model, testData, perGroup) {
  testDataTemp <- subset(testData, select = -c(Id, groupId, matchId, matchType))
  dTest = xgb.DMatrix(data = as.matrix(testDataTemp))
  rm(testDataTemp)
  gc()
  
  testData <- subset(testData, select = c(Id, groupId, matchId, maxPlace))
  
  print("Predicting test data results")
  randForPredict <- predict(model, dTest)
  comparison <- data.frame(cbind(testData, predicted = randForPredict))
  
  print("Calculating average team rank")
  comparison <- comparison %>% group_by(groupId) %>% mutate(avgPred = mean(predicted))
  
  print("Running per match adjustments")
  matchResults <- unique(data.frame(matchId = comparison$matchId, groupId = comparison$groupId, avgPred = comparison$avgPred))
  matchResults <- matchResults %>% group_by(matchId) %>% mutate(projectedRank = order(order(avgPred, decreasing = TRUE))) %>% mutate(maxProjRank = max(projectedRank))
  
  print("Final adjustments to score")
  comparison <- merge(comparison, matchResults, by=c("matchId","groupId"))
  comparison$scaledRankBasedOnProj <- (comparison$maxProjRank - comparison$projectedRank) / (comparison$maxProjRank - 1)
  comparison$projWinPlaceBasedOnProj <- round(comparison$maxPlace - comparison$scaledRankBasedOnProj * (comparison$maxPlace - 1))
  comparison$roundPredFinal <- (comparison$maxPlace - comparison$projWinPlaceBasedOnProj) / (comparison$maxPlace - 1)
  
  comparison[is.na(comparison)] <- 0

  return(comparison)
}

print("Reading training data")
trainData <- read.csv("../input/train_V2.csv")
trainData <- trainData[-which(trainData$matchId == "224a123c53e008"),]
trainData$headshotRate = trainData$kills / trainData$headshotKills
trainData$healthItems = trainData$boosts + trainData$heals
trainData$participation = trainData$kills + trainData$assists + trainData$DBNOs
trainData$damageOverTime = trainData$damageDealt / trainData$matchDuration
trainData$killStreakRatio = trainData$killStreaks / trainData$kills
trainData$mapNumber <- trainData$matchDuration < 1600
trainData <- subset(trainData, select = -c(teamKills, roadKills, winPoints, killPoints, rankPoints, vehicleDestroys, revives))

print("Subsetting train data")
soloTrainData <- trainData[which(grepl("solo", trainData$matchType, fixed = TRUE)),]
soloTrainData <- subset(soloTrainData, select= -c(DBNOs, assists, headshotKills, heals, longestKill, swimDistance, damageDealt))
duosTrainData <- trainData[which(grepl("duo", trainData$matchType, fixed = TRUE)),]
squadTrainData <- trainData[which(grepl("squad", trainData$matchType, fixed = TRUE) | trainData$matchType == "flarefpp" | trainData$matchType == "flaretpp"),]
crashTrainData <- trainData[which(trainData$matchType == "crashfpp" | trainData$matchType == "crashtpp"),]
rm(trainData)

print("Sampling training data for memory usage")
soloTrainData$matchId <- factor(soloTrainData$matchId)
matchList <- levels(soloTrainData$matchId)
#
# First cut down sample, then split into train + verification
#
trainMatches <- sample(matchList, length(matchList))
soloTrainData <- soloTrainData[which(soloTrainData$matchId %in% trainMatches),]
soloTrainData$matchId <- factor(soloTrainData$matchId)
matchList <- levels(soloTrainData$matchId)
trainMatches <- sample(matchList, 3 * length(matchList) / 5)
soloVerificationData <- soloTrainData[which(!(soloTrainData$matchId %in% trainMatches)),]
soloTrainData <- soloTrainData[which(soloTrainData$matchId %in% trainMatches),]

duosTrainData$matchId <- factor(duosTrainData$matchId)
matchList <- levels(duosTrainData$matchId)
trainMatches <- sample(matchList, length(matchList))
duosTrainData <- duosTrainData[which(duosTrainData$matchId %in% trainMatches),]
duosTrainData$matchId <- factor(duosTrainData$matchId)
matchList <- levels(duosTrainData$matchId)
trainMatches <- sample(matchList, 3 * length(matchList) / 5)
duosVerificationData <- duosTrainData[which(!(duosTrainData$matchId %in% trainMatches)),]
duosTrainData <- duosTrainData[which(duosTrainData$matchId %in% trainMatches),]

squadTrainData$matchId <- factor(squadTrainData$matchId)
matchList <- levels(squadTrainData$matchId)
trainMatches <- sample(matchList, length(matchList))
squadTrainData <- squadTrainData[which(squadTrainData$matchId %in% trainMatches),]
squadTrainData$matchId <- factor(squadTrainData$matchId)
matchList <- levels(squadTrainData$matchId)
trainMatches <- sample(matchList, 1 * length(matchList) / 2)
squadVerificationData <- squadTrainData[which(!(squadTrainData$matchId %in% trainMatches)),]
squadTrainData <- squadTrainData[which(squadTrainData$matchId %in% trainMatches),]

crashTrainData$matchId <- factor(crashTrainData$matchId)
matchList <- levels(crashTrainData$matchId)
trainMatches <- sample(matchList, 4 * length(matchList) / 5)
crashVerificationData <- crashTrainData[which(!(crashTrainData$matchId %in% trainMatches)),]
crashTrainData <- crashTrainData[which(crashTrainData$matchId %in% trainMatches),]

rm(matchList)
rm(trainMatches)

print("Scaling Training Data")
soloTrainData <- scaleSoloData(soloTrainData, TRUE)
duosTrainByGroup <- scaleDatasetByGroup(duosTrainData)
duosTrainByMatch <- scaleDatasetByMatch(duosTrainData)
squadTrainByGroup <- scaleDatasetByGroup(squadTrainData)
squadTrainByMatch <- scaleDatasetByMatch(squadTrainData)
crashTrainByGroup <- scaleDatasetByGroup(crashTrainData)
crashTrainByMatch <- scaleDatasetByMatch(crashTrainData)

duosTrainData <- merge(duosTrainByGroup, duosTrainByMatch)
squadTrainData <- merge(squadTrainByGroup, squadTrainByMatch)
crashTrainData <- merge(crashTrainByGroup, crashTrainByMatch)
rm(duosTrainByGroup)
rm(duosTrainByMatch)
rm(squadTrainByGroup)
rm(squadTrainByMatch)
rm(crashTrainByGroup)
rm(crashTrainByMatch)


print("Scaling Verification Data")
soloVerificationData <- scaleSoloData(soloVerificationData, TRUE)
duosTrainByGroup <- scaleDatasetByGroup(duosVerificationData)
duosTrainByMatch <- scaleDatasetByMatch(duosVerificationData)
squadTrainByGroup <- scaleDatasetByGroup(squadVerificationData)
squadTrainByMatch <- scaleDatasetByMatch(squadVerificationData)
crashTrainByGroup <- scaleDatasetByGroup(crashVerificationData)
crashTrainByMatch <- scaleDatasetByMatch(crashVerificationData)

duosVerificationData <- merge(duosTrainByGroup, duosTrainByMatch)
squadVerificationData <- merge(squadTrainByGroup, squadTrainByMatch)
crashVerificationData <- merge(crashTrainByGroup, crashTrainByMatch)
rm(duosTrainByGroup)
rm(duosTrainByMatch)
rm(squadTrainByGroup)
rm(squadTrainByMatch)
rm(crashTrainByGroup)
rm(crashTrainByMatch)

print("Reading test data")
testData <- read.csv("../input/test_V2.csv")
print(nrow(testData))
testData$headshotRate = testData$kills / testData$headshotKills
testData$healthItems = testData$boosts + testData$heals
testData$participation = testData$kills + testData$assists + testData$DBNOs
testData$damageOverTime = testData$damageDealt / testData$matchDuration
testData$killStreakRatio = testData$killStreaks / testData$kills
testData$mapNumber <- testData$matchDuration < 1600
testData <- subset(testData, select = -c(teamKills, roadKills, winPoints, killPoints, rankPoints, vehicleDestroys, revives))

print("Scaling Testing Data")
soloTestData <- testData[which(grepl("solo", testData$matchType, fixed = TRUE)),]
soloTestData <- subset(soloTestData, select= -c(DBNOs, assists, headshotKills, heals, longestKill, swimDistance, damageDealt))
duosTestData <- testData[which(grepl("duo", testData$matchType, fixed = TRUE)),]
squadTestData <- testData[which(grepl("squad", testData$matchType, fixed = TRUE) | testData$matchType == "flarefpp" | testData$matchType == "flaretpp"),]
crashTestData <- testData[which(testData$matchType == "crashfpp" | testData$matchType == "crashtpp"),]
rm(testData)
soloTestData <- scaleSoloData(soloTestData, FALSE)
duosTestByGroup <- scaleDatasetByGroup(duosTestData)
duosTestByMatch <- scaleDatasetByMatch(duosTestData)
squadTestByGroup <- scaleDatasetByGroup(squadTestData)
squadTestByMatch <- scaleDatasetByMatch(squadTestData)
crashTestByGroup <- scaleDatasetByGroup(crashTestData)
crashTestByMatch <- scaleDatasetByMatch(crashTestData)

duosTestData <- merge(duosTestByGroup, duosTestByMatch)
squadTestData <- merge(squadTestByGroup, squadTestByMatch)
crashTestData <- merge(crashTestByGroup, crashTestByMatch)
rm(duosTestByGroup)
rm(duosTestByMatch)
rm(squadTestByGroup)
rm(squadTestByMatch)
rm(crashTestByGroup)
rm(crashTestByMatch)

print("Running Solo model")
print(nrow(soloTrainData))
model <- runModel(soloTrainData, soloVerificationData, FALSE, 0.01)
rm(soloTrainData)
rm(soloVerificationData)

print("Outputting Solo Importance Matrix")
imp <- xgb.importance(model = model)
write.csv(imp, file = "solo_importance.csv", row.names=FALSE)

print("Prediction Solo Outcome")
comp <- predictOutcome(model, soloTestData)
soloPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(soloTestData)
rm(model)
rm(comp)

print("Running Duos model")
print(nrow(duosTrainData))
model <- runModel(duosTrainData, duosVerificationData, TRUE, 0.07)
rm(duosTrainData)
rm(duosVerificationData)

print("Outputting Duos Importance Matrix")
imp <- xgb.importance(model = model)
write.csv(imp, file = "duos_importance.csv", row.names=FALSE)

print("Prediction Duos Outcome")
comp <- predictOutcome(model, duosTestData)
duosPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(duosTestData)
rm(model)
rm(comp)

print("Running Crash model")
print(nrow(crashTrainData))
model <- runModel(crashTrainData, crashVerificationData, TRUE, 0.01)
rm(crashTrainData)

print("Outputting Crash Importance Matrix")
imp <- xgb.importance(model = model)
write.csv(imp, file = "crash_importance.csv", row.names=FALSE)

print("Prediction Crash Outcome")
comp <- predictOutcome(model, crashTestData)
crashPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(comp)
rm(crashTestData)
rm(model)

gc()

print("Running Squad model")
print(nrow(squadTrainData))
model <- runModel(squadTrainData, squadVerificationData, TRUE, 0.07)
rm(squadTrainData)
rm(squadVerificationData)

print("Outputting Squad Importance Matrix")
imp <- xgb.importance(model = model)
write.csv(imp, file = "squad_importance.csv", row.names=FALSE)
rm(imp)

print("Prediction Squad Outcome")
comp <- predictOutcome(model, squadTestData)
squadPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(comp)

print("Outputting result")
outputData <- rbind(soloPredictions, squadPredictions)
outputData <- rbind(outputData, duosPredictions)
outputData <- rbind(outputData, crashPredictions)
write.csv(outputData, file = "submission_rounded.csv", row.names=FALSE)