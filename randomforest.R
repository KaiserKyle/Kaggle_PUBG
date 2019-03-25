library(Metrics)
library(dplyr)
library(ranger)
library(BBmisc)
library(e1071)

minMaxScale <- function(vect) {
  maxValue <- max(vect)
  minValue <- min(vect)
  
  scaled <- (vect - minValue) / (maxValue - minValue)
  
  return(scaled)
}

notFactor <- function(thing) {
  return(!is.factor(thing))
}

scaleDatasetByGroup <- function(data) {
  
  print("Messing with points")
  data$points <- ifelse(data$rankPoints == -1, (data$killPoints + data$winPoints) / 2, data$rankPoints)
  data$nonRoadKills <- data$kills - data$roadKills
  
  print("Per group operations")
  data <- data %>% group_by(groupId) %>% mutate(groupMembers = n()) %>% mutate_if(notFactor, funs(sum = sum, max = max, min = min))
  
  #data$teamWalkDiff <- data$maxGroupWalkDist - data$minGroupWalkDist
  
  print("Replacing NAs")
  data[is.na(data)] <- 0
  
  return (data)
}

scaleDatasetByMatch <- function(data) {
  print("Per match operations")
  data <- data %>% group_by(groupId) %>% mutate(groupMembers = n())
  data <- data %>% group_by(matchId) %>% mutate_if(notFactor, funs(scaled = scale, minMax = minMaxScale))
  data <- data %>% group_by(matchId) %>% mutate(matchCount = n(), avgTeamSize = mean(groupMembers)) 
  
  print("Replacing NAs")
  data[is.na(data)] <- 0
  
  return (data)
}

runModel <- function(trainData, perGroup) {
    print("Training Random Forest Algorithm")
    if (perGroup) {
        trainData <- subset(trainData, select = -c(winPlacePerc_sum, winPlacePerc_min, winPlacePerc_max)) 
    }
    trainData <- subset(trainData, select = -c(winPlacePerc_scaled, winPlacePerc_minMax))
    randFor <- ranger(winPlacePerc ~ . - Id - groupId - matchId - matchCount - avgTeamSize - maxPlace - matchType, data = trainData, importance = 'impurity', num.trees = 500)
    print(importance(randFor))
    print(randFor$prediction.error)
    
    return(randFor)
}

predictOutcome <- function(model, testData) {
    print("Predicting test data results")
    randForPredict <- predict(model, testData)
    comparison <- data.frame(cbind(testData, randForPredicted = randForPredict$predictions))
    #comparison <- data.frame(cbind(testData, randForPredicted = randForPredict))
  
    print("Calculating average team rank")
    comparison <- comparison %>% group_by(groupId) %>% mutate(avgPred = mean(randForPredicted))
  
    print("Running per match adjustments")
    matchResults <- unique(data.frame(matchId = comparison$matchId, groupId = comparison$groupId, avgPred = comparison$avgPred))
    matchResults <- matchResults %>% group_by(matchId) %>% mutate(projectedRank = order(order(avgPred, decreasing = TRUE))) %>% mutate(maxProjRank = max(projectedRank))
  
    print("Final adjustments to score")
    comparison <- merge(comparison, matchResults, by=c("matchId","groupId"))
    comparison$scaledRankPred <- (comparison$maxPlace - comparison$projectedRank) / (comparison$maxPlace - 1)
    comparison$scaledRankBasedOnProj <- (comparison$maxProjRank - comparison$projectedRank) / (comparison$maxProjRank - 1)
    comparison$projWinPlaceBasedOnProj <- round(comparison$maxPlace - comparison$scaledRankBasedOnProj * (comparison$maxPlace - 1))
    comparison$roundPredFinal <- (comparison$maxPlace - comparison$projWinPlaceBasedOnProj) / (comparison$maxPlace - 1)

    comparison[is.na(comparison)] <- 0
  
    return(comparison)
}

print("Reading training data")
trainData <- read.csv("../input/train_V2.csv")
trainData <- trainData[-which(trainData$matchId == "224a123c53e008"),]

print("Subsetting train data")
#trainData <- subset(trainData, select= -c(heals, scaledHeals, totalGroupHeals, roadKills, teamKills, vehicleDestroys, headshotKills, killStreaks, numGroups))
trainData <- trainData[complete.cases(trainData),]
soloTrainData <- trainData[which(trainData$matchType == "solo" | trainData$matchType == "solo-fpp"),]
soloTrainData <- subset(soloTrainData, select= -c(DBNOs, revives, assists, roadKills, teamKills, vehicleDestroys, killPoints, rankPoints, winPoints, headshotKills))
duosTrainData <- trainData[which(trainData$matchType == "duo" | trainData$matchType == "duo-fpp"),]
squadTrainData <- trainData[which(trainData$matchType != "solo" & trainData$matchType != "solo-fpp" & trainData$matchType != "duo" & trainData$matchType != "duo-fpp"),]
rm(trainData)

print("Sampling training data for memory usage")
soloTrainData$matchId <- factor(soloTrainData$matchId)
matchList <- levels(soloTrainData$matchId)
trainMatches <- sample(matchList, length(matchList) / 3)
soloTrainData <- soloTrainData[which(soloTrainData$matchId %in% trainMatches),]

duosTrainData$matchId <- factor(duosTrainData$matchId)
matchList <- levels(duosTrainData$matchId)
trainMatches <- sample(matchList, length(matchList) / 10)
duosTrainData <- duosTrainData[which(duosTrainData$matchId %in% trainMatches),]

squadTrainData$matchId <- factor(squadTrainData$matchId)
matchList <- levels(squadTrainData$matchId)
trainMatches <- sample(matchList, length(matchList) / 17)
squadTrainData <- squadTrainData[which(squadTrainData$matchId %in% trainMatches),]

print("Scaling Training Data")
soloTrainData <- scaleDatasetByMatch(soloTrainData)
duosTrainByGroup <- scaleDatasetByGroup(duosTrainData)
duosTrainByMatch <- scaleDatasetByMatch(duosTrainData)
squadTrainByGroup <- scaleDatasetByGroup(squadTrainData)
squadTrainByMatch <- scaleDatasetByMatch(squadTrainData)

duosTrainData <- merge(duosTrainByGroup, duosTrainByMatch)
squadTrainData <- merge(squadTrainByGroup, squadTrainByMatch)
rm(duosTrainByGroup)
rm(duosTrainByMatch)
rm(squadTrainByGroup)
rm(squadTrainByMatch)

print("Reading test data")
testData <- read.csv("../input/test_V2.csv")
print(nrow(testData))

print("Scaling Testing Data")
soloTestData <- testData[which(testData$matchType == "solo" | testData$matchType == "solo-fpp"),]
duosTestData <- testData[which(testData$matchType == "duo" | testData$matchType == "duo-fpp"),]
nonSoloTestData <- testData[which(testData$matchType != "solo" & testData$matchType != "solo-fpp" & testData$matchType != "duo" & testData$matchType != "duo-fpp"),]
rm(testData)
soloTestData <- scaleDatasetByMatch(soloTestData)
duosTestByGroup <- scaleDatasetByGroup(duosTestData)
duosTestByMatch <- scaleDatasetByMatch(duosTestData)
squadTestByGroup <- scaleDatasetByGroup(nonSoloTestData)
squadTestByMatch <- scaleDatasetByMatch(nonSoloTestData)

duosTestData <- merge(duosTestByGroup, duosTestByMatch)
squadTestData <- merge(squadTestByGroup, squadTestByMatch)
rm(duosTestByGroup)
rm(duosTestByMatch)
rm(squadTestByGroup)
rm(squadTestByMatch)

print("Running Solo model")
print(nrow(soloTrainData))
model <- runModel(soloTrainData, FALSE)
rm(soloTrainData)

print("Prediction Solo Outcome")
comp <- predictOutcome(model, soloTestData)
soloPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(soloTestData)
rm(model)
rm(comp)

print("Running Duos model")
print(nrow(duosTrainData))
model <- runModel(duosTrainData, TRUE)
rm(duosTrainData)

print("Prediction Duos Outcome")
comp <- predictOutcome(model, duosTestData)
duosPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(duosTestData)
rm(model)
rm(comp)

print("Running Non-Solo model")
print(nrow(squadTrainData))
model <- runModel(squadTrainData, TRUE)
rm(squadTrainData)

print("Prediction Squad Outcome")
comp <- predictOutcome(model, squadTestData)
nonSoloPredictions <- data.frame(Id = comp$Id, winPlacePerc = comp$roundPredFinal)
rm(comp)

print("Outputting result")
outputData <- rbind(soloPredictions, nonSoloPredictions)
outputData <- rbind(outputData, duosPredictions)
write.csv(outputData, file = "submission.csv", row.names=FALSE)
outputData$winPlacePerc <- round(outputData$winPlacePerc, 4)
write.csv(outputData, file = "submission_rounded.csv", row.names=FALSE)
