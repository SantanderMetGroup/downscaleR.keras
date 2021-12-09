##     downscalePredict.keras.R Downscale climate data for a a previous defined deep learning keras model.
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
#' @title Downscale climate data for a a previous defined deep learning keras model.
#' @description Downscale data to local scales by deep learning models previously obtained by \code{\link[downscaleR.keras]{downscaleTrain.keras}}.
#' @param newdata The grid data. It should be an object as returned by  \code{\link[downscaleR.keras]{prepareNewData.keras}}.
#' @param model An object containing the statistical model as returned from  \code{\link[downscaleR.keras]{downscaleTrain.keras}} 
#' or a list of arguments passed to \code{\link[keras]{load_model_hdf5}}.
#' @param C4R.template A climate4R grid that serves as template for the returned prediction object.
#' @param clear.session A logical value. Indicates whether we want to destroy the current tensorflow graph and clear the model from memory.
#' In particular, refers to whether we want to use the function \code{\link[keras]{k_clear_session}} after training.
#' If FALSE, model is returned. If TRUE, then k_clear_session() is applied and no model is returned.
#' Default to FALSE.
#' @param loss Default to NULL. Otherwise a string indicating the loss function used to train the model. This is only
#' relevant where we have used the 2 custom loss functions of this library: "gaussianLoss" or "bernouilliGammaLoss"
#' 
#' @details This function relies on keras, which is a high-level neural networks API capable of running on top of tensorflow, CNTK or theano.
#' There are official \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep learning models. We suggest the user, especially the beginners,
#' to consult these tutorials before using downscalePredict.keras.
#' @return A regular/irregular grid object.
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @author J. Bano-Medina
#' @family downscaling.functions
#' @import keras
#' @importFrom transformeR array3Dto2Dmat mat2Dto3Darray isRegular bindGrid redim makeMultiGrid
#' @export
#' @examples \donttest{ 
#' # Loading data
#' require(climate4R.datasets)
#' require(transformeR)
#' require(magrittr)
#' require(keras)
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
#' # Preparing the predictors
#' xy.T <- prepareData.keras(x = xT, y = yT, 
#'                           first.connection = "conv",
#'                           last.connection = "dense",
#'                           channels = "last")
#' # Defining the keras model.... 
#' # We define 3 hidden layers that consists on
#' # 2 convolutional steps followed by a dense connection.
#' input_shape  <- dim(xy.T$x.global)[-1]
#' output_shape  <- dim(xy.T$y$Data)[2]
#' inputs <- layer_input(shape = input_shape)
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs <- layer_dense(hidden,units = output_shape)
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' # We can print model in console to observe its configuration
#' model
#' # Training the deep learning model
#' model <- downscaleTrain.keras(xy.T,
#'                               model = model,
#'                               compile.args = list("loss" = "mse", 
#'                               "optimizer" = optimizer_adam(lr = 0.01)),
#'                               fit.args = list("epochs" = 30, "batch_size" = 100))
#' # Predicting on the test set...
#' xy.t <- prepareNewData.keras(newdata = xt,data.structure = xy.T)
#' pred <- downscalePredict.keras(newdata = xy.t,
#'                                model = model,
#'                                clear.session = TRUE,
#'                                C4R.template = yT)
#' # We can now apply the visualizeR functions to the prediction
#' # as it preserves the climate4R template.
#' require(visualizeR)
#' temporalPlot(yt,pred)
#' }
downscalePredict.keras <- function(newdata,
                                   model,
                                   C4R.template,
                                   clear.session = FALSE,
                                   loss = NULL) {
  if (is.list(model)) model <- do.call("load_model_hdf5",model)
  
  x.global <- newdata$x.global
  n.mem <- length(x.global)
  pred <- lapply(1:n.mem, FUN = function(z) {
    x.global[[z]] %>% model$predict()  
  })
  names(pred) <- paste("member", 1:n.mem, sep = "_")
  if (isTRUE(clear.session)) k_clear_session()
  
  template <- C4R.template
  if (attr(newdata,"last.connection") == "dense") {
    ind <- attr(newdata,"indices_noNA_y")
    n.vars <- ncol(pred[[1]])/length(ind)
    if (isRegular(template)) {ncol.aux <- array3Dto2Dmat(template$Data) %>% ncol()} else {ncol.aux <- getShape(template,dimension = "loc")}
    pred <- lapply(1:n.mem,FUN = function(z) {
      aux <- matrix(nrow = nrow(pred[[z]]), ncol = ncol.aux)
      lapply(1:n.vars, FUN = function(zz) {
        aux[,ind] <- pred[[z]][,((ncol(pred[[1]])/n.vars)*(zz-1)+1):(ncol(pred[[1]])/n.vars*zz)]
        if (isRegular(template)) aux <- mat2Dto3Darray(aux, x = template$xyCoords$x, y = template$xyCoords$y)
        aux
      })
    })
  } 
  dimNames <- attr(template$Data,"dimensions")
  pred <- lapply(1:n.mem, FUN = function(z) {
    if (attr(newdata,"last.connection") == "dense") {
      lapply(1:n.vars, FUN = function(zz) {
        template$Data <- pred[[z]][[zz]]
        attr(template$Data,"dimensions") <- dimNames
        if (isRegular(template))  template <- redim(template, var = FALSE)
        if (!isRegular(template)) template <- redim(template, var = FALSE, loc = TRUE)
        return(template)
      }) %>% makeMultiGrid()
    } else {
      if (attr(newdata,"channels") == "first") n.vars <- dim(pred$member_1)[2]
      if (attr(newdata,"channels") == "last")  n.vars <- dim(pred$member_1)[4]
      lapply(1:n.vars, FUN = function(zz) {
        if (attr(newdata,"channels") == "first") template$Data <- pred[[z]] %>% aperm(c(2,1,3,4))
        if (attr(newdata,"channels") == "last")  template$Data <- pred[[z]] %>% aperm(c(4,1,2,3))
        template$Data <- template$Data[zz,,,,drop = FALSE]
        attr(template$Data,"dimensions") <- c("var","time","lat","lon")
        return(template)
      }) %>% makeMultiGrid()
    }
  })
  
  if (isRegular(template)) pred <- do.call("bindGrid",pred) %>% redim(drop = TRUE)
  if (!isRegular(template)) pred <- do.call("bindGrid",pred) %>% redim(drop = TRUE) %>% redim(member = FALSE, loc = TRUE)
  pred$Dates <- attr(newdata,"dates")
  n.vars <- getShape(redim(pred,var = TRUE),"var")
  if (n.vars > 1) {
    if (loss == "gaussianLoss") {
      pred$Variable$varName <- c("mean","log_var")
    } else if (loss == "bernouilliGammaLoss") {
      pred$Variable$varName <- c("p","log_alpha","log_beta")
    } else {
      pred$Variable$varName <- paste0(pred$Variable$varName,1:n.vars)
    }
    pred$Dates <- rep(list(pred$Dates),n.vars)
    
  }
  return(pred)
}
