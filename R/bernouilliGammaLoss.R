##     bernouilliGammaLoss.R Custom loss function for BernouilliGamma distributions.
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

#' @title Custom loss function for Bernouilli-Gamma distributions.
#' @description This loss function optimizes the negative log-likelihood of a Bernouill-Gamma distribution. It is a custom
#' loss function defined according to keras functions.
#' @param last.connection A string with values "conv" o "dense" depending on the type of the net's last connection. DEFAULT is NULL.
#' @details Note that infering a conditional Bernouilli-Gamma distribution means to estimate their associated parameters:
#' probability, shape and scale factor. To avoid computational instabilities
#' we actually estimate log alpha and log beta. Therefore, make sure that your output layers are designed according
#' to this property of the distribution.
#' @seealso 
#' bernouilliGammaStatistics for computing the expectance or simulate from the discrete continuous distribution
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' \href{https://github.com/SantanderMetGroup/downscaleR/wiki/training-downscaling-models}{downscaleR Wiki} for downscaling seasonal forecasting and climate projections.
#' @author J. Bano-Medina
#' @import tensorflow
#' @export
bernouilliGammaLoss <- function(last.connection = NULL) {
  if (last.connection == "dense") {
    custom_metric("custom_loss", function(true, pred){
      K <- backend()
      D <- K$int_shape(pred)[[2]]/3
      ocurrence = pred[,1:D]
      shape_parameter = tf$exp(pred[,(D+1):(D*2)])
      scale_parameter = tf$exp(pred[,(D*2+1):(D*3)])
      bool_rain = tf$cast(tf$greater(true,0),tf$float32)
      epsilon = 0.000001
      return (- tf$reduce_mean((1-bool_rain)*tf$math$log(1-ocurrence+epsilon) + bool_rain*(tf$math$log(ocurrence+epsilon) + (shape_parameter - 1)*tf$math$log(true+epsilon) - shape_parameter*tf$math$log(scale_parameter+epsilon) - tf$math$lgamma(shape_parameter+epsilon) - true/(scale_parameter+epsilon))))
    })
  } else if (last.connection == "conv") {
    custom_metric("custom_loss", function(true, pred){
      K = backend()
      ocurrence = pred[,,,1, drop = TRUE]
      shape_parameter = tf$exp(pred[,,,2, drop = TRUE])
      scale_parameter = tf$exp(pred[,,,3, drop = TRUE])
      bool_rain = tf$cast(tf$greater(true,0),tf$float32)
      epsilon = 0.000001
      return (- tf$reduce_mean((1-bool_rain)*tf$math$log(1-ocurrence+epsilon) + bool_rain*(tf$math$log(ocurrence+epsilon) + (shape_parameter - 1)*tf$math$log(true+epsilon) - shape_parameter*tf$math$log(scale_parameter+epsilon) - tf$math$lgamma(shape_parameter+epsilon) - true/(scale_parameter+epsilon))))
    })
  }
}
