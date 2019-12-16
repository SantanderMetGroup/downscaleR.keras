##     discreteGammaRegular.R Custom loss function for BernouilliGamma distributions.
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
#' loss function defined according to keras functions. Use this function when working with irregular grids, otherwise 
#' optimize \code{\link[downscaleR.keras]{discreteGammaIrregular}}
#' @param ... An empty character
#' @seealso 
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR/wiki/training-downscaling-models}{downscaleR Wiki} for downscaling seasonal forecasting and climate projections.
#' @author J. Bano-Medina
#' @family downscaling.functions
#' @export
discreteGammaRegular <- custom_metric("custom_loss", function(true, pred){
  K = backend()
  ocurrence = pred[,,,1, drop = FALSE]
  shape_parameter = K$exp(pred[,,,2, drop = FALSE])
  scale_parameter = K$exp(pred[,,,3, drop = FALSE])
  bool_rain = K$cast(K$greater(true,0),K$tf$float32)
  epsilon = 0.000001
  return (- K$mean((1-bool_rain)*K$tf$log(1-ocurrence+epsilon) + bool_rain*(K$tf$log(ocurrence+epsilon) + (shape_parameter - 1)*K$tf$log(true+epsilon) - shape_parameter*K$tf$log(scale_parameter+epsilon) - K$tf$lgamma(shape_parameter+epsilon) - true/(scale_parameter+epsilon))))
})