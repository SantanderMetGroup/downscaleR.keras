##     gaussianLoss.R Custom loss function for BernouilliGamma distributions.
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

#' @title Custom loss function for Gaussian distributions.
#' @description This loss function optimizes the negative log-likelihood of a Gaussian distribution. It is a custom
#' loss function defined according to keras functions.
#' @param last.connection A string with values "conv" o "dense" depending on the type of the net's last connection. DEFAULT is NULL.
#' @details Note that infering a conditional gaussian distribution means to estimate their associated parameters:
#' mean, and variance. To avoid computational instabilities
#' we actually estimate log var. Therefore, make sure that your output layers are designed according
#' to this property of the distribution.
#' @seealso 
#' bernouilliGammaLoss for computing the negative log-likelihood of a Bernouilli-Gamma distribution
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' \href{https://github.com/SantanderMetGroup/downscaleR/wiki/training-downscaling-models}{downscaleR Wiki} for downscaling seasonal forecasting and climate projections.
#' @author J. Bano-Medina
#' @import tensorflow
#' @export
gaussianLoss <- function(last.connection = NULL) {
  if (last.connection == "dense") {
    custom_metric("custom_loss", function(true, pred){
      K <- backend()
      D <- K$int_shape(pred)[[2]]/2
      mean <- pred[, 1:D]
      log_var <- pred[, (D+1):(D*2)]
      precision <- tf$exp(-log_var)
      return(tf$reduce_mean( 0.5 * precision * (true - mean) ^ 2 +  0.5 * log_var))
    })
  } else if (last.connection == "conv") {
    custom_metric("custom_loss", function(true, pred){
      K <- backend()
      mean <- pred[,,,1, drop = TRUE]
      log_var <- pred[,,,2, drop = TRUE]
      precision <- tf$exp(-log_var)
      return(tf$reduce_mean(0.5 * precision * (true - mean) ^ 2 + 0.5 * log_var))
    })
  }
}
