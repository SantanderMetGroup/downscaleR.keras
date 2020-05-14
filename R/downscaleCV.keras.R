##     downscaleCV.keras.R Downscaling method calibration in cross validation mode.
##
##     Copyright (C) 2018 Santander Meteorology Group (http://www.meteo.unican.es)
##
##     This program is free software: you can redistribute it and/or modify
##     it under the terms of the GNU General Public License as published by
##     the Free Software Foundation, either version 3 of the License, or
##     (at your option) any later version.
## 
##     This program is distributed in the hope that it will be useful,
##     but WITHOUT ANY WARRANTY; without even the implied warranty of
##     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##     GNU General Public License for more details.
## 
##     You should have received a copy of the GNU General Public License
##     along with this program.  If not, see <http://www.gnu.org/licenses/>.

#' @title Downscale climate data and reconstruct the temporal serie by splitting the data following a user-defined scheme
#' @description Downscale climate data and reconstruct the temporal serie by splitting the data following a user-defined scheme.
#' The statistical downscaling methods currently implemented are: analogs, generalized linear models (GLM) and Neural Networks (NN). 
#' @param x The input grid (admits both single and multigrid, see \code{\link[transformeR]{makeMultiGrid}}). It should be an object as returned by \pkg{loadeR}.
#' @param y The observations dataset. It should be an object as returned by \pkg{loadeR}.
#' @param MC A numeric value, default to NULL. The number of Monte-Carlo samples in case the model
#' is a bayesian neural network (note that any network containing dropout is equivalent 
#' mathematically to a bayesian neural network). We refer the reader to the 
#' custom \code{\link[downscaleR.keras]{concreteDropout}} and \code{\link[downscaleR.keras]{spatialConcreteDropout}} of how to learn 
#' the dropout probability.
#' @param model A keras sequential or functional model. 
#' @param sampling.strategy Specifies a sampling strategy to define the training and test subsets. Possible values are 
#' \code{"kfold.chronological"} (the default), \code{"kfold.random"}, \code{"leave-one-year-out"} and NULL.
#' The \code{sampling.strategy} choices are next described:
#' \itemize{
#'   \item \code{"kfold.random"} creates the number of folds indicated in the \code{folds} argument by randomly sampling the entries along the time dimension.
#'   \item \code{"kfold.chronological"} is similar to \code{"kfold.random"}, but the sampling is performed in ascending order along the time dimension.
#'   \item \code{"leave-one-year-out"}. This scheme performs a leave-one-year-out cross validation. It is equivalent to introduce in the argument \code{folds} a list of all years one by one.
#'   \item \code{NULL}. The folds are specified by the user in the function parameter \code{folds}.
#' }
#' The first two choices will be controlled by the argument \code{folds} (see below)
#' @param folds This arguments controls the number of folds, or how these folds are created (ignored if \code{sampling.strategy = "leave-one-year-out"}). If it is given as a fraction in the range (0-1), 
#' it splits the data in two subsets, one for training and one for testing, being the given value the fraction of the data used for training (i.e., 0.75 will split the data so that 75\% of the instances are used for training, and the remaining 25\% for testing). 
#' In case it is an integer value (the default, which is 4), it sets the number of folds in which the data will be split (e.g., \code{folds = 10} for the classical 10-fold cross validation). 
#' Alternatively, this argument can be passed as a list, each element of the list being a vector of years to be included in each fold (See examples).
#' @param scaleGrid.args A list of the parameters related to scale grids. This parameter calls the function \code{\link[transformeR]{scaleGrid}}. See the function definition for details on the parameters accepted.
#' @param prepareData.keras.args A list with the arguments of the \code{\link[downscaleR]{prepareData}} function. Please refer to \code{\link[downscaleR]{prepareData}} help for
#' more details about this parameter.
#' @param compile.args A list of the arguments passed to the \code{\link[keras]{compile}} keras function, or equivalently to the
#' \code{compile.args} parameter that appears in the \code{\link[downscaleR.keras]{downscaleTrain.keras}} function.
#' @param fit.args A list of the arguments passed to the \code{\link[keras]{fit}} keras function, or equivalently to the
#' \code{fit.args} parameter that appears in the \code{\link[downscaleR.keras]{downscaleTrain.keras}} function.
#' @param transferLearning A logic value. Whether there should the transfer learning among folds? If TRUE
#' then the parameters learned in fold 1 are used as the initial state on fold 2 and so on...
#' @param loss Default to NULL. Otherwise a string indicating the loss function used to train the model. This is only
#' relevant where we have used the 2 custom loss functions of this library: "gaussianLoss" or "bernouilliGammaLoss"
#' @param binarySerie A logic value, default to FALSE. Indicate whether to conver the predicted probabilities of rain
#' to a binary value by adjusting the frequency of rainy days to that observed in the train period. Note that this is
#' only valid when our aim is to downscale precipitation and we set the "loss" parameter to the custom function
#' "bernouilliGammaLoss". We need to define what we consider as rainy day, see the parameters \code{condition} and \code{threshold} to
#' set these values.
#' @param condition Inequality operator to be applied considering the given threshold.
#' \code{"GT"} = greater than the value of \code{threshold}, \code{"GE"} = greater or equal,
#' \code{"LT"} = lower than, \code{"LE"} = lower or equal than. Values that accomplish the condition turn to 1 whereas the others turn to 0.
#' @param threshold An integer. Threshold used as reference for the condition. Default is NULL. 
#' @details The function relies on \code{\link[downscaleR.keras]{prepareData.keras}}, \code{\link[downscaleR.keras]{prepareNewData.keras}}, \code{\link[downscaleR.keras]{downscaleTrain.keras}}, and \code{\link[downscaleR.keras]{downscalePredict.keras}}. 
#' For more information please visit these functions. It is envisaged to allow for a flexible fine-tuning of the cross-validation scheme. It uses internally the \pkg{transformeR} 
#' helper \code{\link[transformeR]{dataSplit}} for flexible data folding. 
#' Note that the indices for data splitting are obtained using \code{\link[transformeR]{getYearsAsINDEX}} when needed (e.g. in leave-one-year-out cross validation), 
#' thus adequately handling potential inconsistencies in year selection when dealing with year-crossing seasons (e.g. DJF).
#' 
#' If the variable to downscale is the precipitation and it is a binary variable,
#'  then two temporal series will be returned:
#' \enumerate{
#' \item The temporal serie with binary values filtered by a threshold adjusted by the train dataset, see \code{\link[transformeR]{binaryGrid}} for more details.
#' \item The temporal serie with the results obtained by the downscaling, without binary conversion process.
#' }
#' 
#' Please note that Keras do not handle missing data and these are removed previous to infer the model. 
#' 
#' According to the concept of cross-validation, a particular year should not appear in more than one fold
#' (i.e., folds should constitute disjoint sets). For example, the choice \code{fold =list(c(1988,1989), c(1989, 1990))}
#'  will raise an error, as 1989 appears in more than one fold.
#' 
#' @return The reconstructed downscaled temporal serie.
#' @importFrom transformeR dataSplit scaleGrid binaryGrid makeMultiGrid filterNA getYearsAsINDEX intersectGrid convert2bin
#' @author J. Bano-Medina
#' @export
#' @examples 
#' # Loading data
#' require(transformeR)
#' data("VALUE_Iberia_tas")
#' y <- VALUE_Iberia_tas
#' data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
#' x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
#' # mse
#' inputs <- layer_input(shape = c(getShape(x,"lat"),getShape(x,"lon"),getShape(x,"var")))
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs <- layer_dense(hidden,units = getShape(y,"loc"))
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' pred <- downscaleCV.keras(x, y, model,
#'            sampling.strategy = "kfold.chronological", folds = 4, 
#'            scaleGrid.args = list(type = "standardize"),
#'            prepareData.keras.args = list(first.connection = "conv",
#'                                          last.connection = "dense",
#'                                          channels = "last"),
#'            compile.args = list(loss = "mse", 
#'                           optimizer = optimizer_adam()),
#'            fit.args = list(batch_size = 100, epochs = 100, validation_split = 0.1))
#' # gaussianLoss 
#' inputs <- layer_input(shape = c(getShape(x,"lat"),getShape(x,"lon"),getShape(x,"var")))
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs1 <- layer_dense(hidden,units = getShape(y,"loc"))
#' outputs2 <- layer_dense(hidden,units = getShape(y,"loc"))
#' outputs <- layer_concatenate(list(outputs1,outputs2))
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' pred <- downscaleCV.keras(x, y, model,
#'          sampling.strategy = "kfold.chronological", folds = 2, 
#'          scaleGrid.args = list(type = "standardize"),
#'          prepareData.keras.args = list(first.connection = "conv",
#'                                        last.connection = "dense",
#'                                        channels = "last"),
#'          compile.args = list(loss = gaussianLoss(last.connection = "dense"), 
#'                                               optimizer = optimizer_adam()),
#'          fit.args = list(batch_size = 100, epochs = 100, validation_split = 0.1),
#'                     loss = "gaussianLoss")
#' 
#' # bernouilliGammaLoss
#' data("VALUE_Iberia_pr")
#' y <- VALUE_Iberia_pr
#' inputs <- layer_input(shape = c(getShape(x,"lat"),getShape(x,"lon"),getShape(x,"var")))
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs1 <- layer_dense(hidden,units = getShape(y,"loc"), activation = "sigmoid")
#' outputs2 <- layer_dense(hidden,units = getShape(y,"loc"))
#' outputs3 <- layer_dense(hidden,units = getShape(y,"loc"))
#' outputs <- layer_concatenate(list(outputs1,outputs2,outputs3))
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' y <- gridArithmetics(y,0.99,operator = "-") %>% binaryGrid("GT",0,partial = TRUE) 
#' pred <- downscaleCV.keras(x, y, model,
#'            sampling.strategy = "kfold.chronological", folds = 4, 
#'            scaleGrid.args = list(type = "standardize"),
#'            prepareData.keras.args = list(first.connection = "conv",
#'                                          last.connection = "dense",
#'                                          channels = "last"),
#'            compile.args = list(loss = bernouilliGammaLoss(last.connection = "dense"), 
#'                                optimizer = optimizer_adam()),
#'            fit.args = list(batch_size = 100, epochs = 100, validation_split = 0.1),
#'            loss = "bernouilliGammaLoss",
#'            binarySerie = TRUE)
downscaleCV.keras <- function(x, y, model,MC = NULL,
                        sampling.strategy = "kfold.chronological", folds = 4, 
                        scaleGrid.args = NULL,
                        prepareData.keras.args = NULL,
                        compile.args = NULL,
                        fit.args = NULL,
                        transferLearning = FALSE,
                        loss = NULL, 
                        binarySerie = FALSE,
                        condition = NULL,
                        threshold = NULL) {
  
  x <- getTemporalIntersection(x,y,which.return = "obs")
  y <- getTemporalIntersection(x,y,which.return = "prd")
  
  if (!is.null(sampling.strategy)) {
    if (sampling.strategy == "leave-one-year-out") {
      type <- "chronological"
      folds <- as.list(getYearsAsINDEX(y) %>% unique())
    }
    
    if (sampling.strategy == "kfold.chronological") {
      type <- "chronological"
      if (!is.numeric(folds)) {
        folds.user <- unlist(folds) %>% unique() %>% sort()
        folds.data <- getYearsAsINDEX(y) %>% unique()
        if (any(folds.user != folds.data)) stop("In the parameters folds you have indicated years that do not belong to the dataset. Please revise the setup of this parameter.")
      }
    }
    if (sampling.strategy == "kfold.random") {
      type <- "random"
      if (!is.numeric(folds)) stop("In kfold.random, the parameter folds represent the NUMBER of folds and thus, it should be a NUMERIC value.")
    }
  }
  if (is.list(folds)) {
    if (any(duplicated(unlist(folds)))) stop("Years can not appear in more than one fold")
  }
  
  data <- dataSplit(x,y, f = folds, type = type)
  p <- lapply(1:length(data), FUN = function(xx) {
    message(paste("fold:",xx,"-->","calculating..."))
    modelCV <- if(isTRUE(transferLearning)) {model} else {clone_model(model)}
    xT <- data[[xx]]$train$x ; yT <- data[[xx]]$train$y
    xt <- data[[xx]]$test$x  ; yt <- data[[xx]]$test$y
    yT <- filterNA(yT)
    xT <- intersectGrid(xT,yT,which.return = 1)
    if (!is.null(scaleGrid.args)) {
      scaleGrid.args$base <- xT
      scaleGrid.args$grid <- xt
      scaleGrid.args$skip.season.check <- TRUE
      xt <- do.call("scaleGrid",args = scaleGrid.args)
      scaleGrid.args$grid <- xT
      xT <- do.call("scaleGrid",args = scaleGrid.args)
    }
    prepareData.keras.args[["x"]] <- xT
    prepareData.keras.args[["y"]] <- yT
    xy <- do.call("prepareData.keras",args = prepareData.keras.args) 
    xt <- prepareNewData.keras(newdata = xt, data.structure = xy)
    modelCV <- downscaleTrain.keras(xy, modelCV,compile.args = compile.args,fit.args = fit.args)
    out <- if (is.null(MC)) {
      downscalePredict.keras(xt, modelCV, C4R.template = yT, loss = loss)
    } else {  
      lapply(1:MC,FUN = function(z) downscalePredict.keras(xt, modelCV, C4R.template = yT, loss = loss)) %>%
        bindGrid(dimension = "member")
    }
    if (isTRUE(binarySerie)) {
      xT <- prepareNewData.keras(newdata = xT, data.structure = xy)
      aux2 <- downscalePredict.keras(xT, modelCV, C4R.template = yT, loss = loss) %>%
        subsetGrid(var = "p")
      aux1 <- subsetGrid(out,var = "p")
      bin <- binaryGrid(aux1, ref.obs = binaryGrid(yT,condition = condition,threshold = threshold), ref.pred = aux2) 
      bin$Variable$varName <- "bin"
      out <- makeMultiGrid(subsetGrid(out,var = "p"),
                           subsetGrid(out,var = "log_alpha"),
                           subsetGrid(out,var = "log_beta"),
                           bin) %>% redim(member = FALSE)
    }
    k_clear_session()
    rm(modelCV)
    return(out)
  }) %>% bindGrid(dimension = "time")
  return(p)
}
