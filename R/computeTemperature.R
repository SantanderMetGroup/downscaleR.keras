##     computeRainfall.R Custom loss function for BernouilliGamma distributions.
##
##     Copyright (C) 2017 Santander Meteorology Group (http://www.meteo.unican.es)
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

#' @title Sample from the daily conditional gaussian distributions.
#' @description Compute the stochastic temperature given the parameters of conditional gaussian distributions
#' as returned by \code{\link[downscaleR.keras]{downscalePredict.keras}} on a trained model 
#' (see \code{\link[downscaleR.keras]{downscaleTrain.keras}})
#' that optimized the loss function \code{\link[downscaleR.keras]{gaussianLoss}}.
#' @param mean A grid. The mean of the daily gaussian conditional distributions, as returned by downscalePredict.keras when we use the gaussian loss function.
#' @param log_var A grid. The log of the variance ("log_beta") of the daily conditional distributions, as returned by downscalePredict.keras when we use the gaussian loss function.
#' @seealso 
#' bernouilliGammaLoss a custom loss function for Bernouilli-Gamma distributions
#' gaussianLoss a custom loss function for gaussian distributions
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' @return A climate4R grid with the deterministic or stochastic temporal serie 
#' depending on whether the \code{simulate} parameter is FALSE or TRUE.
#' @author J. Bano-Medina
#' @import tensorflow
#' @importFrom stats rnorm runif
#' @export
#' @examples \donttest{
#' require(climate4R.datasets)
#' require(transformeR)
#' data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
#' x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
#' data("VALUE_Iberia_tas")
#' y <- VALUE_Iberia_tas
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
#' sampling.strategy = "kfold.chronological", folds = 4, 
#' scaleGrid.args = list(type = "standardize"),
#' prepareData.keras.args = list(first.connection = "conv",
#'                               last.connection = "dense",
#'                               channels = "last"),
#' compile.args = list(loss = gaussianLoss(last.connection = "dense"), 
#'                     optimizer = optimizer_adam()),
#' fit.args = list(batch_size = 100, epochs = 10, validation_split = 0.1),
#'                           loss = "gaussianLoss")
#' 
#' # We sample from the daily conditional gaussian distributions
#' pred <- computeTemperature(mean = subsetGrid(pred,var = "mean"),
#'                            log_var = subsetGrid(pred,var = "log_var"))
#' }

computeTemperature <- function(mean=NULL,
                            log_var=NULL) {
  mean %<>% redim(mean, drop = TRUE) %>% redim(member = TRUE)
  log_var  %<>% redim(log_var, drop = TRUE) %>% redim(member = TRUE)
  n.mem <- getShape(log_var,"member")
  out <- lapply(1:n.mem, FUN = function(z) {
    mean %<>% subsetGrid(members = z)
    log_var  %<>% subsetGrid(members = z)
    t <- mean
    dimNames <- attr(mean$Data,"dimensions")
    ntime <- getShape(mean,"time")
    if (isRegular(mean)) {
      mean <- array3Dto2Dmat(mean$Data)
      sd <- array3Dto2Dmat(exp(log_var$Data) %>% sqrt())
    } else {
      mean <- mean$Data
      sd <- exp(log_var$Data) %>% sqrt()
    }
    aux <- matrix(nrow = ntime,ncol = ncol(mean))
    for (zz in 1:ncol(mean)) {
      aux[,zz] <- rnorm(n = ntime, 
                        mean = mean[,zz], 
                        sd = sd[,zz])
    }
    if (isRegular(t)) {
      t$Data <- mat2Dto3Darray(aux,x = t$xyCoords$x, y = t$xyCoords$y)
    } else {
      t$Data <- aux  
    }
    return(t)
  }) %>% bindGrid() %>% redim(drop = TRUE)
  out$Variable$varName <- "tas"
  return(out)
}