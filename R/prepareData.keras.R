#   prepareData.keras.R Configuration of predictors for downscaling
#
#   Copyright (C) 2017 Santander Meteorology Group (http://www.meteo.unican.es)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

#' @title Configuration of data for downscaling with a keras model
#' @description Configuration of data for flexible downscaling keras experiment definition
#' @param x A grid (usually a multigrid) of predictor fields
#' @param y A grid (usually a stations grid, but not necessarily) of observations (predictands)
#' @param global.vars An optional character vector with the short names of the variables of the input \code{x} 
#'  multigrid to be retained as global predictors (use the \code{\link{getVarNames}} helper if not sure about variable names).
#'  This argument just produces a call to \code{\link[transformeR]{subsetGrid}}, but it is included here for better
#'  flexibility in downscaling experiments (predictor screening...). For instance, it allows to use some 
#'  specific variables contained in \code{x} as local predictors and the remaining ones, specified in \code{subset.vars},
#'  as either raw global predictors or to construct the combined PC.
#' @param y A grid (usually a stations grid, but not necessarily) of observations (predictands)
#' @param spatial.predictors Default to \code{NULL}, and not used. Otherwise, a named list of arguments in the form \code{argument = value},
#'  with the arguments to be passed to \code{\link[transformeR]{prinComp}} to perform Principal Component Analysis
#'  of the predictors grid (\code{x}). See Details on principal component analysis of predictors.
#' @param combined.only Optional, and only used if spatial.predictors parameters are passed. Should the combined PC be used as the only
#' global predictor? Default to TRUE. Otherwise, the combined PC constructed with \code{which.combine} argument in 
#' \code{\link{prinComp}} is append to the PCs of the remaining variables within the grid.
#' @param local.predictors Default to \code{NULL}, and not used. Otherwise, a named list of arguments in the form \code{argument = value},
#'  with the following arguments:
#'  \itemize{
#'    \item \code{vars}: names of the variables in \code{x} to be used as local predictors
#'    \item \code{fun}: Optional. Aggregation function for the selected local neighbours.
#'    The aggregation function is specified as a list, indicating the name of the aggregation function in
#'     first place (as character), and other optional arguments to be passed to the aggregation function.
#'     For instance, to compute the average skipping missing values: \code{fun = list(FUN= "mean", na.rm = TRUE)}.
#'     Default to NULL, meaning that no aggregation is performed.
#'    \item \code{n}: Integer. Number of nearest neighbours to use. If a single value is introduced, and there is more
#'    than one variable in \code{vars}, the same value is used for all variables. Otherwise, this should be a vector of the same
#'    length as \code{vars} to indicate a different number of nearest neighbours for different variables.
#'  }
#'  Note that grid 'y' has to be single-site, otherwise this will cause errors in the model training, since downscaleTrain.keras
#'  is designed to store only one model at a time due to Keras particularities. If your desire is to downscale to multiple-sites
#'  for independent models, please loop over this function for the different sites.
#' @param first.connection A string. Possible values are c("dense","conv") depending on whether 
#' the first connection (i.e., input layer to first hidden layer) is dense or convolutional.
#' @param last.connection A string. Same as \code{first.connection} but for the last connection
#' (i.e., last hidden layer to output layer).
#' @param channels A string. Possible values are c("first","last") and indicates the dimension of the channels (i.e., climate variables)
#' in the array. If "first" then dimensions = c("channel","latitude","longitude") for regular grids or c("channel","loc") for irregular grids.
#' If "last" then dimensions = c("latitude","longitude","channel") for regular grids or c("loc","channel") for irregular grids.
#' @param time.frames The number of time frames to build the recurrent neural network. If e.g., time.frame = 2, then the value 
#' y(t) is a function of x(t) and x(t-1). The time frames stack in the input array prior to the input neurons or channels (in conv. layers). 
#' See \code{\link[keras]{layer_simple_rnn}},\code{\link[keras]{layer_lstm}} or \code{\link[keras]{layer_conv_lstm_2d}}. 
#' @return A named list with components \code{y} (the predictand), \code{x.global} (global predictors) and other attributes. For the case when 
#' spatial and local predictors are both computed, these are stacked together in the \code{x.global} object. 
#' @details Remove days containing NA in at least one predictand site.
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @importFrom transformeR getTemporalIntersection getRefDates getCoordinates getVarNames
#' @importFrom magrittr %<>% %>% 
#' @importFrom downscaleR prepareData
#' @import keras
#' @importFrom transformeR array3Dto2Dmat mat2Dto3Darray isRegular bindGrid redim getDim subsetGrid getVarNames
#' @importFrom abind abind
#' @seealso \href{https://github.com/SantanderMetGroup/downscaleR/wiki/preparing-predictor-data}{downscaleR Wiki} for preparing predictors for downscaling and seasonal forecasting.
#' @family downscaling.helpers
#' @export
#'  
#' @author J. Baño-Medina
#' @examples \donttest{
#' require(climate4R.datasets) 
#' # Loading data
#' require(transformeR)
#' require(downscaleR)
#' data("VALUE_Iberia_tas")
#' y <- VALUE_Iberia_tas
#' data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
#' x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
#' # We standardize the predictors using transformeR function scaleGrid
#' x <- scaleGrid(x,type = "standardize")
#' # Preparing the predictors
#' data <- prepareData.keras(x = x, y = y, 
#'                           first.connection = "conv",
#'                           last.connection = "dense",
#'                           channels = "last")
#' # We can visualize the outputield not imported f
#' str(data)
#' 
#' # We can call prepareData.keras to compute PCs over the predictor field
#' data <- prepareData.keras(x = x, y = y, 
#'                           spatial.predictors = list(v.exp = 0.95),   # the EOFs that explain the 95% of the total variance
#'                           first.connection = "dense",
#'                           last.connection = "dense",
#'                           channels = "last")
#' }
prepareData.keras <- function(x,y,
                              global.vars = NULL,
                              combined.only = TRUE,
                              spatial.predictors = NULL,
                              local.predictors = NULL,
                              first.connection = c("dense","conv"),
                              last.connection = c("dense","conv"),
                              channels = c("first","last"),
                              time.frames = NULL) {
  x <- x %>% redim(drop = TRUE)
  if(any(getDim(x) == "member")) stop("No members allowed for training keras model")
  x <- x %>% redim(var = TRUE, member = FALSE)
  
  # predictor 'x' ---------------------------------------------------------------------------------
  if (first.connection == "dense") {
    if (any(!is.null(global.vars) || !is.null(spatial.predictors) || !is.null(local.predictors))) {
      x <- do.call("prepareData", args = list("x" = x, "y" = y, "global.vars" = global.vars, "spatial.predictors" = spatial.predictors, "local.predictors" = local.predictors))  
      x$y <- NULL
      if (!is.null(local.predictors)) {
        x.global <- cbind(x$x.global,x$x.local[[1]]$member_1)
      } else {
        x.global <- x$x.global
      }
      attr(x.global,"data.structure") <- x
    } else {
      if (isRegular(x)) {
        x.global <- lapply(getVarNames(x), FUN = function(z){
          array3Dto2Dmat(subsetGrid(x,var = z)$Data)
        }) %>% abind::abind(along = 0)
      } else{
        x.global <- x$Data
      } 
      x.global <- x.global %>% aperm(c(2,3,1)) 
      dim(x.global) <- c(dim(x.global)[1],prod(dim(x.global)[2:3]))
    }
    if (anyNA(x.global)) stop("There are NaNs in object: x, please consider using function filterNA prior to prepareData.keras")
  } else if (first.connection == "conv") {
    if (!isRegular(x)) stop("Object 'x' must be a regular grid")
    if (anyNA(x$Data)) stop("NaNs were found in object: x, please consider using function filterNA prior to prepareData.keras")
    
    if (channels == "last") x.global <- x$Data %>% aperm(c(2,3,4,1))
    if (channels == "first") x.global <- x$Data %>% aperm(c(2,1,3,4))
  }
  # Adding time frame for recurrent layers
  if (!is.null(time.frames)) {
    xx.global <- array(dim = c(dim(x.global)[1]-time.frames+1,time.frames,dim(x.global)[-1]))
    for (t in 1:dim(xx.global)[1]) {
      if (first.connection == "dense") xx.global[t,,] <- x.global[t:(t+time.frames-1),]
      if (first.connection == "conv") xx.global[t,,,,] <- x.global[t:(t+time.frames-1),,,] 
    }
    x.global <- xx.global
    rm(xx.global)
  }
  
  # predictand 'y' ---------------------------------------------------------------------------------
  if (last.connection == "dense") {
    if (isRegular(y)) {
      y$Data <- array3Dto2Dmat(y$Data)
    } 
    if (anyNA(y$Data)) warning("removing gridpoints containing NaNs of object: y")
    ind.y <- (!apply(y$Data,MARGIN = 2,anyNA)) %>% which()
    y$Data <- y$Data[,ind.y, drop = FALSE]
  } else if (last.connection == "conv") {
    if (!isRegular(y)) stop("Object 'y' must be a regular grid")
    if (anyNA(y$Data)) stop("NaNs were found in object: y")
  }
  
  # Adding time frame for recurrent layers
  if (!is.null(time.frames)) {
    if (last.connection == "dense") y$Data <- y$Data[time.frames:dim(y$Data)[1],, drop = FALSE]
    if (last.connection == "conv") y$Data <- y$Data[time.frames:dim(y$Data)[1],,, drop = FALSE]
    y$Dates$start <- y$Dates$start[time.frames:dim(y$Data)[1]]
    y$Dates$end <- y$Dates$end[time.frames:dim(y$Data)[1]]
  }
  
  predictor.list <- list("y" = y, "x.global" = x.global)
  if (last.connection  == "dense") attr(predictor.list,"indices_noNA_y") <- ind.y
  attr(predictor.list,"first.connection") <- first.connection
  attr(predictor.list,"last.connection") <- last.connection
  attr(predictor.list,"channels") <- channels
  attr(predictor.list,"time.frames") <- time.frames
  
  return(predictor.list)
}
