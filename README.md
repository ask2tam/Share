# Share
HUE: https://pcdgwkafkap02.datalake.local:8889/hue/editor/?type=hive

CDSW:  http://cdsw.datalake.local/login

> factorizeColumnsFromDict
function(dt,
                                     dictFile = config.path %+% "factorsDict_" %+% Sys.info()["user"] %+% ".csv"
                                     ) {
  dt.dict <- fread(dictFile)
  for(col in unique(dt.dict[, varName])){
    dt[, c(col) := factor(get(col), levels = dt.dict[varName == col, value])]
  }
  
  return(dt)
}
<bytecode: 0x7fb2d3cdc268>

getPredictions <- function(dt,
                           modelResultsInput.path){
  #Load the model
  varsToModel <- fread(modelResultsInput.path %+% "trainingVars.csv")
  algorithm <- first(varsToModel[, algorithm])
  varsToModel <- varsToModel[, trainingVars]
  
  missingVars <- setdiff(varsToModel, colnames(dt))
  if(length(missingVars) > 0){
    stop("The following variables were used to train the model but are not present in the table that we want to score: " %+% paste(missingVars, collapse = ", "))
 }  
  
  charVars <- intersect(varsToModel,
                        colnames(dt)[dt[, sapply(.SD, function(col){any(class(col) == "character")})]])
  if(length(charVars) > 0){
    stop("The following variables were used to train the model but are character type in the table that we want to score: " %+% paste(charVars, collapse = ", "))
  }
  
  #Make the predictions
  if(algorithm == "xgboost"){
    modelObject <- readRDS(modelResultsInput.path %+% "modelObject.rds")
    modelData.test <- xgb.DMatrix(data.matrix(dt[,c(varsToModel), with=FALSE]), missing=NA)
    modelPredVec  <- predict(modelObject, modelData.test, missing=NA)
  }else if(algorithm == "lightgbm"){
    modelObject <- lgb.load(modelResultsInput.path %+% "modelObject.lgbm")
    modelData.test <- data.matrix(dt[,c(varsToModel), with=FALSE])
    modelPredVec  <- predict(modelObject, modelData.test)
  }
  return(modelPredVec)
}

