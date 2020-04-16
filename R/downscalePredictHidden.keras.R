##     downscalePredictHidden.keras.R Downscale climate data for a a previous defined deep learning keras model.
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
#' @title Obtain the output of the hidden layers in a deep learning keras model for downscaling.
#' @description Obtain the output of the hidden layers in a deep learning keras model for downscaling, previously infered with \code{\link[downscaleR.keras]{downscaleTrain.keras}}
#' Also, returns the kernel of the corresponding hidden layer that generated the output.
#' @param newdata The grid data. It should be an object as returned by  \code{\link[downscaleR.keras]{prepareNewData.keras}}.
#' @param model An object containing the statistical model as returned from  \code{\link[downscaleR.keras]{downscaleTrain.keras}} 
#' or a list of arguments passed to \code{\link[keras]{load_model_hdf5}}.
#' @param layer A string or numeric value. The output of which hidden layer should be returned? 
#' @param clear.session A logical value. Indicates whether we want to destroy the current tensorflow graph and clear the model from memory.
#' In particular, refers to whether we want to use the function \code{\link[keras]{k_clear_session}} after training.
#' If FALSE, model is returned. If TRUE, then k_clear_session() is applied and no model is returned.
#' Default to FALSE.
#' 
#' @details This function relies on keras, which is a high-level neural networks API capable of running on top of tensorflow, CNTK or theano.
#' There are official \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep learning models. We suggest the user, especially the beginners,
#' to consult these tutorials before using downscalePredictHidden.keras.
#' @return A list containing the filter maps and the associated kernels. 
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @author J. Bano-Medina
#' @export

downscalePredictHidden.keras <- function(newdata,
                                         model,
                                         layer,
                                         clear.session) {
  
  if (is.list(model)) model <- do.call("load_model_hdf5",model)
  x.global <- newdata$x.global
  n.mem <- length(x.global)
  if (is.numeric(layer)) {index <- layer; layer_name <- name <-  NULL
  } else {
    layer_name <- name <- layer; index <- NULL  
  }
  pred <- list()  
  intermediate_layer_model <- keras_model(inputs = model$input,
                                          outputs = get_layer(model, name = layer_name, index = index)$output)
  pred$filterMap <- lapply(1:n.mem, FUN = function(z) {
    aux <- x.global[[z]] %>% intermediate_layer_model$predict()
    if (length(dim(aux)) > 2) {
      if (attr(newdata,"channels") == "last") {
        attr(aux,"dimensions") <- c("time","lat","lon","var")
      } else if (attr(newdata,"channels") == "first") {
        attr(aux,"dimensions") <- c("time","var","lat","lon")
      }
    } else{
      attr(aux,"dimensions") <- c("time","neurons")
    }
    return(aux)
  })
  names(pred$filterMap) <- paste("member", 1:n.mem, sep = "_")
  
  pred$kernel$Weigths <- get_layer(object = model,name = name, index = index)$get_weights()[[1]] 
  pred$kernel$Bias <- get_layer(object = model,name = name, index = index)$get_weights()[[2]] 
  if (isTRUE(clear.session)) k_clear_session()
  
  attr(pred,"dates") <- attr(newdata,"dates")
  return(pred)
}