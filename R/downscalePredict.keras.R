#' @title Downscale climate data for a a previous defined deep learning keras model.
#' @description Downscale data to local scales by deep learning models previously obtained by \code{\link[downscaleR.keras]{downscaleTrain.keras}}.
#' @param newdata The grid data. It should be an object as returned by  \code{\link[downscaleR.keras]{prepareNewData.keras}}.
#' @param model An object containing the statistical model as returned from  \code{\link[downscaleR.keras]{downscaleTrain.keras}}.
#' @param C4R.template A climate4R grid that serves as template for the returned prediction object.
#' @param clear.session A logical value. Indicates whether we want to destroy the current tensorflow graph and clear the model from memory.
#' In particular, refers to whether we want to use the function \code{\link[keras]{k_clear_session}} after training.
#' If FALSE, model is returned. If TRUE, then k_clear_session() is applied and no model is returned.
#' Default to FALSE.
#' @param ... A list of arguments passed to \code{\link[transformeR]{aggregateGrid}} that will operate over the prediction.
#' 
#' @details This function relies on keras, which is a high-level neural networks API capable of running on top of tensorflow, CNTK or theano.
#' There are official \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep learning models. We suggest the user, especially the beginners,
#' to consult these tutorials before using downscalePredict.keras.
#' @return A regular/irregular grid object.
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR/wiki/training-downscaling-models}{downscaleR Wiki} for downscaling seasonal forecasting and climate projections.
#' @author J. Bano-Medina
#' @family downscaling.functions
#' @import keras
#' @importFrom transformeR array3Dto2Dmat mat2Dto3Darray isRegular bindGrid redim
#' @export
#' @examples 
#' # Loading data
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
#'   layer_flatten(l3) %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs <- layer_dense(hidden,units = output_shape)
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' # We can print model in console to observe its configuration
#' model
#' # Training the deep learning model
#' model <- downscaleTrain.keras(xy.T,
#'                               model = model,
#'                               compile.args = list("loss" = "mse", "optimizer" = optimizer_adam(lr = 0.01)),
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
downscalePredict.keras <- function(newdata,
                                   model,
                                   C4R.template,
                                   clear.session = FALSE,
                                   ...) {
  aggr.args <- list(...)
  if (is.list(model)) model <- do.call("load_model_hdf5",model)
  
  x.global <- newdata$x.global
  n.mem <- length(x.global)
  pred <- lapply(1:n.mem, FUN = function(z) {
    x.global[[z]] %>% model$predict()  
  })
  names(pred) <- paste("member", 1:n.mem, sep = "_")
  if (isTRUE(clear.session)) k_clear_session()
  
  if (attr(newdata,"last.connection") == "dense") {
    template <- C4R.template
    if (isRegular(template)) {ncol.aux <- array3Dto2Dmat(template$Data) %>% ncol()} else {ncol.aux <- getShape(template,dimension = "loc")}
    ind <- attr(newdata,"indices_noNA_y")
    pred <- lapply(1:n.mem,FUN = function(z) {
      aux <- matrix(nrow = nrow(pred[[z]]), ncol = ncol.aux)
      aux[,ind] <- pred[[z]]
    })
    if (isRegular(template)) pred <- lapply(1:n.mem,FUN = function(z) mat2Dto3Darray(pred[[z]], x = template$xyCoords$x, y = template$xyCoords$y))
  }
  dimNames <- attr(template$Data,"dimensions")
  pred <- lapply(1:n.mem, FUN = function(z) {
    template$Data <- pred[[z]]
    attr(template$Data,"dimensions") <- dimNames
    if (isRegular(template))  template <- redim(template, var = FALSE)
    if (!isRegular(template)) template <- redim(template, var = FALSE, loc = TRUE)
    return(template)
  })
  
  pred <- do.call("bindGrid",pred) %>% redim(drop = TRUE)
  pred$Dates <- attr(newdata,"dates")
  
  # # manipulate_output
  # if (is.null(aggr.args)) {
  #   fun.args <- c("grid" = pred, aggr.args)
  #   pred <- do.call("aggregateGrid", fun.args)
  # }
  return(pred)
}