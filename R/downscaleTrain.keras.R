##     downscaleTrain.keras.R Train a deep model with keras in the climate4R framework
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

#' @title Train a deep model with keras in the climate4R framework
#' @description Train a deep model with keras in the climate4R framework.
#' @param obj The object as returned by \code{\link[downscaleR.keras]{prepareData.keras}}.
#' @param model A keras sequential or functional model. 
#' @param compile.args List of arguments passed to \code{\link[keras]{compile}} function of keras. 
#' Some arguments are the \href{https://keras.rstudio.com/reference/loss_mean_squared_error.html}{loss function} or
#' the  \href{https://www.rdocumentation.org/packages/kerasR/versions/0.6.1/topics/Optimizers}{optimizer}. An example could be:
#' compile.args = list("loss" = "mse", "optimizer" = optimizer_adam(lr = 0.0001)). 
#' The default parameters are those used by default in the official \href{https://keras.rstudio.com/reference/compile.html}{compile} keras function.
#' Note that the \code{loss} == "mse" and \code{optimizer} = optimizer_adam() as DEFAULT.
#' @param fit.args List of arguments passed to \code{\link[keras]{fit}} function of keras.
#' Arguments are those encountered in the \href{https://keras.rstudio.com/reference/fit.html}{fit R documentation}. An example could be:
#' fit.args = list("batch_size" = 100,"epochs" = 50, "validation_split" = 0.1). 
#' The default parameters are those used by default in the official \href{https://keras.rstudio.com/reference/fit.html}{fit} keras function.
#' @param clear.session A logical value. Indicates whether we want to destroy the current tensorflow graph and clear the model from memory.
#' In particular, refers to whether we want to use the function \code{\link[keras]{k_clear_session}} after training.
#' If FALSE, model is returned. If TRUE, then k_clear_session() is applied and no model is returned.
#' Default to FALSE.
#' @details This function relies on keras, which is a high-level neural networks API capable of running on top of tensorflow, CNTK or theano.
#' There are official \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep learning models. We suggest the user, especially the beginners,
#' to consult these tutorials before using downscaleTrain.keras.
#'  
#' @return The infered keras model.
#' @seealso 
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' prepareData.keras for predictor preparation of training data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @import keras
#' @author J. Bano-Medina
#' @family downscaling.functions
#' @importFrom transformeR gridArithmetics
#' @export
#' @examples \donttest{
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
#' 
#' # Defining the keras model.... 
#' # We define 3 hidden layers that consists on 
#' # 2 convolutional steps followed by a dense connection.
#' input_shape  <- dim(data$x.global)[-1]
#' output_shape  <- dim(data$y$Data)[2]
#' inputs <- layer_input(shape = input_shape)
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs <- layer_dense(hidden,units = output_shape)
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' 
#' # We can print model in console to observe its configuration
#' model
#' 
#' # Training the deep learning model
#' model <- downscaleTrain.keras(data,
#'                               model = model,
#'                               compile.args = list("loss" = "mse", "optimizer" = optimizer_adam(lr = 0.01)),
#'                               fit.args = list("epochs" = 30, "batch_size" = 100))
#' 
#' # Training a deep learning model 
#' # (saving the model using callbacks according to an early-stopping criteria)
#' downscaleTrain.keras(data,
#'                      model = model,
#'                      compile.args = list("loss" = "mse", "optimizer" = optimizer_adam(lr = 0.01)),
#'                      fit.args = list("epochs" = 50, "batch_size" = 100, "validation_split" = 0.1,
#'                                      "callbacks" = list(callback_early_stopping(patience = 10),
#'                                                         callback_model_checkpoint(filepath=paste0(getwd(),"/model.h5"),
#'                                                                                   monitor='val_loss', save_best_only=TRUE))),
#'                      clear.session = TRUE)
#' }
downscaleTrain.keras <- function(obj,
                                 model,
                                 compile.args = list("object" = model),
                                 fit.args = list("object" = model),
                                 clear.session = FALSE) {
  # compile
  compile.args[["object"]] <- model
  if (is.null(compile.args[["optimizer"]])) compile.args[["optimizer"]] <- optimizer_adam()
  if (is.null(compile.args[["loss"]])) compile.args[["loss"]] <- "mse"
  do.call(compile,args = compile.args)
  
  # fit
  fit.args[["object"]] <- model
  fit.args[["x"]] <- obj$x.global
  fit.args[["y"]] <- obj$y$Data ;   # dim(y) <- c(dim(y),1)
  do.call(fit,args = fit.args)
  if (isTRUE(clear.session)) {k_clear_session()} else {model}
}