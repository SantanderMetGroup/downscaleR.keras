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
#' @param first.connection A string. Possible values are c("dense","conv") depending on whether 
#' the first connection (i.e., input layer to first hidden layer) is dense or convolutional.
#' @param last.connection A string. Same as \code{first.connection} but for the last connection
#' (i.e., last hidden layer to output layer).
#' @param channels A string. Possible values are c("first","last") and indicates the dimension of the channels (i.e., climate variables)
#' in the array. If "first" then dimensions = c("channel","latitude","longitude") for regular grids or c("channel","loc") for irregular grids.
#' If "last" then dimensions = c("latitude","longitude","channel") for regular grids or c("loc","channel") for irregular grids.
#' @return A named list with components \code{y} (the predictand), \code{x.global} (global predictors) and other attributes. See Examples.
#' @details Remove days containing NA in at least one predictand site.
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @importFrom transformeR getTemporalIntersection getRefDates getCoordinates getVarNames
#' @importFrom magrittr %<>% %>% 
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
#' }
prepareData.keras <- function(x,y,
                              first.connection = c("dense","conv"),
                              channels = c("first","last"),
                              last.connection = c("dense","conv")) {
  x <- x %>% redim(drop = TRUE)
  if(any(getDim(x) == "member")) stop("No members allowed for training keras model")
  x <- x %>% redim(var = TRUE, member = FALSE)
  
  # predictor 'x'
  if (first.connection == "dense") {
    if (isRegular(x)) {
      x.global <- lapply(getVarNames(x), FUN = function(z){
        array3Dto2Dmat(subsetGrid(x,var = z)$Data)
      }) %>% abind::abind(along = 0)
    } else{
      x.global <- x$Data
    } 
    
    if (channels == "last")  x.global <- x.global %>% aperm(c(2,3,1)) 
    if (channels == "first") x.global <- x.global %>% aperm(c(2,1,3))
    dim(x.global) <- c(dim(x.global)[1],prod(dim(x.global)[2:3]))
    if (anyNA(x.global)) warning("removing gridpoints containing NaNs in object: x")
    ind.x <- (!apply(x.global,MARGIN = 2,anyNA)) %>% which()
    x.global <- x.global[,ind.x]
    
  } else if (first.connection == "conv") {
    if (!isRegular(x)) stop("Object 'x' must be a regular grid")
    if (anyNA(x$Data)) stop("NaNs were found in object: x")
    
    if (channels == "last") x.global <- x$Data %>% aperm(c(2,3,4,1))
    if (channels == "first") x.global <- x$Data %>% aperm(c(2,1,3,4))
  }
  
  
  # predictand 'y'
  if (last.connection == "dense") {
    if (isRegular(y)) {
      y$Data <- array3Dto2Dmat(y$Data)
    } 
    if (anyNA(y$Data)) warning("removing gridpoints containing NaNs of object: y")
    ind.y <- (!apply(y$Data,MARGIN = 2,anyNA)) %>% which()
    y$Data <- y$Data[,ind.y]
  } else if (last.connection == "conv") {
    if (!isRegular(y)) stop("Object 'y' must be a regular grid")
    if (anyNA(y$Data)) stop("NaNs were found in object: y")
  }
  
  predictor.list <- list("y" = y, "x.global" = x.global)
  if (first.connection == "dense") attr(predictor.list,"indices_noNA_x") <- ind.x
  if (last.connection  == "dense") attr(predictor.list,"indices_noNA_y") <- ind.y
  attr(predictor.list,"first.connection") <- first.connection
  attr(predictor.list,"last.connection") <- last.connection
  attr(predictor.list,"channels") <- channels
  
  return(predictor.list)
}
