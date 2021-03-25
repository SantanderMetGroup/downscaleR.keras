#   prepareNewData.keras.R Configuration of data for downscaling method predictions
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

#' @title Prepare newdata for predictions going to be donw with a keras model
#' @description Prepare the prediction data according to the definition of the keras deep model's experiment
#' @param newdata A grid containing the prediction data.
#' @param data.structure A structure, as returned by \code{\link{prepareData.keras}}
#' @return A named list with the components required by downscalePredict.keras in order to perform the predictions
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareData.keras for predictor preparation of training data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @import keras
#' @importFrom transformeR array3Dto2Dmat mat2Dto3Darray isRegular bindGrid redim getDim subsetGrid getVarNames 
#' @export
#' @seealso \href{https://github.com/SantanderMetGroup/downscaleR/wiki/preparing-predictor-data}{downscaleR Wiki} for preparing predictors for downscaling and seasonal forecasting.
#' @author J Baño-Medina
#' @family downscaling.keras.helpers
#' @importFrom transformeR getVarNames subsetGrid redim getShape getCoordinates grid2PCs getRefDates array3Dto2Dmat grid2PCs
#' @importFrom magrittr %>% extract2 
#' @importFrom downscaleR prepareNewData
#' @examples \donttest{
#' # Loading data
#' require(climate4R.datasets)
#' require(transformeR)
#' data("VALUE_Iberia_tas")
#' y <- VALUE_Iberia_tas
#' data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
#' x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
#' # We divide in train and test data and standardize the predictors 
#' # using transformeR functions subsetGrid and scaleGrid, respectively.
#' xT <- subsetGrid(x,years = 1983:1995)
#' xt <- subsetGrid(x,years = 1996:2002) %>% scaleGrid(base = xT, type = "standardize")
#' xT <- scaleGrid(xT,type = "standardize")
#' yT <- subsetGrid(y,years = 1983:1995)
#' yt <- subsetGrid(y,years = 1996:2002)
#' # Preparing the predictors for training...
#' xy.T <- prepareData.keras(x = xT, y = yT, 
#'                           first.connection = "conv",
#'                           last.connection = "dense",
#'                           channels = "last")
#' # Preparing the predictors for prediction...
#' xy.t <- prepareNewData.keras(newdata = xt,data.structure = xy.T)
#' str(xy.t)
#' }

prepareNewData.keras <- function(newdata,data.structure) {
  first.connection <- attr(data.structure,"first.connection")
  last.connection  <- attr(data.structure,"last.connection") 
  channels <- attr(data.structure,"channels")
  time.frames <- attr(data.structure,"time.frames")
  
  if (first.connection == "dense") ind.x <- attr(data.structure,"indices_noNA_x")
  
  newdata <- newdata %>% redim(var = TRUE)
  n.mem <- getShape(newdata, "member")
  newdata.global.list <- lapply(1:n.mem, function(j) {
    newdata <- subsetGrid(newdata,members = j) %>% redim(member = FALSE, var = TRUE)
    if (first.connection == "dense") {
      if (any(names(attributes(data.structure$x.global)) == "data.structure")) {
        newdata <- do.call("prepareNewData", args = list("newdata" = newdata, "data.structure" =  attr(data.structure$x.global,"data.structure")))
        attr(data.structure$x.global,"data.structure") <- NULL  
        if (!is.null(newdata$x.local)) {
          x.global <- cbind(newdata$x.global$member_1,newdata$x.local[[1]]$member_1)
        } else {
          x.global <- newdata$x.global$member_1
        }
      } else {
        if (isRegular(newdata)) {
          x.global <- lapply(getVarNames(newdata), FUN = function(z){
            array3Dto2Dmat(subsetGrid(newdata,var = z)$Data)
          }) %>% abind::abind(along = 0)
        } else{
          x.global <- newdata$Data
        } 
        x.global <- x.global %>% aperm(c(2,3,1)) 
        dim(x.global) <- c(dim(x.global)[1],prod(dim(x.global)[2:3]))
      } 
    } else if (first.connection == "conv") {
      if (!isRegular(newdata)) stop("Object 'newdata' must be a regular grid")
      if (anyNA(newdata$Data)) stop("NaNs were found in object: newdata")
      
      if (channels == "last") x.global <- newdata$Data %>% aperm(c(2,3,4,1))
      if (channels == "first") x.global <- newdata$Data %>% aperm(c(2,1,3,4))
    }
    
    # Adding time frame for recurrent layers
    if (!is.null(time.frames)) {
      xx.global <- array(dim = c(dim(x.global)[1]-time.frames+1,time.frames,dim(x.global)[-1]))
      for (t in 1:dim(xx.global)[1]) {
        if (first.connection == "dense") xx.global[t,,] <- x.global[t:(t+time.frames-1),]
        if (first.connection == "conv") xx.global[t,,,,] <- x.global[t:(t+time.frames-1),,,] 
      }
      x.global <- xx.global
    }
    return(x.global)
  })
  names(newdata.global.list) <- paste("member", 1:n.mem, sep = "_")
  predictor.list  <- list("x.global" = newdata.global.list)
  if (last.connection  == "dense") attr(predictor.list,"indices_noNA_y") <- attr(data.structure,"indices_noNA_y")
  attr(predictor.list,"first.connection") <- first.connection
  attr(predictor.list,"last.connection") <- last.connection
  attr(predictor.list,"channels") <- channels
  dates <- subsetGrid(newdata,var = getVarNames(newdata)[1])$Dates
  attr(predictor.list,"dates") <- if (!is.null(time.frames)) {
    dates$start <- dates$start[time.frames:length(dates$start)]
    dates$end <- dates$end[time.frames:length(dates$end)]
    dates
  } else {
    dates
  }
  return(predictor.list)
}