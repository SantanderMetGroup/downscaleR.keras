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

#' @title Compute the amount of precipitation given the parameters of a Gamma distribution.
#' @description Compute the amount of precipitation given the parameters of a Gamma distribution
#' as returned by \code{\link[downscaleR.keras]{downscalePredict.keras}} on a trained model 
#' (see \code{\link[downscaleR.keras]{downscaleTrain.keras}})
#' that optimized the loss function \code{\link[downscaleR.keras]{bernouilliGammaLoss}}.
#' @param log_alpha A grid. The log of the shape parameter ("log_alpha") as returned by downscalePredict.keras when we use the bernouilliGamma loss function.
#' @param log_beta A grid. The log of the scale parameter ("log_beta") as returned by downscalePredict.keras when we use the bernouilliGamma loss function.
#' @param simulate A logical value. If TRUE then the output is an stochastic sample for the given distribution parameters (p,alpha and beta).
#' @param bias An integer. Added to the quantity of rain after its computation, either deterministically or
#' stochastically. It basically displaces the distribution. Default is NULL.
#' @seealso 
#' bernouilliGammaLoss a custom loss function for Bernouilli-Gamma distributions
#' gaussianLoss a custom loss function for gaussian distributions
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' @details This function uses \code{tensorflow} internally. Please if not, install tensorflow (>= v.)
#' @return A climate4R grid with the deterministic or stochastic temporal serie 
#' depending on whether the \code{simulate} parameter is FALSE or TRUE.
#' @author J. Bano-Medina
#' @import tensorflow
#' @importFrom stats rgamma runif
#' @export
#' @examples \donttest{
#' require(climate4R.datasets)
#' require(transformeR)
#' data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
#' x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
#' data("VALUE_Iberia_pr")
#' y <- VALUE_Iberia_pr
#' inputs <- layer_input(shape = c(getShape(x,"lat"),getShape(x,"lon"),getShape(x,"var")))
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs1 <- layer_dense(hidden,units = getShape(y,"loc"), activation = "sigmoid")
#' outputs2 <- layer_dense(hidden,units = getShape(y,"loc"))
#' outputs3 <- layer_dense(hidden,units = getShape(y,"loc"))
#' outputs <- layer_concatenate(list(outputs1,outputs2,outputs3))
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' y <- gridArithmetics(y,0.99,operator = "-") %>% binaryGrid("GT",0,partial = TRUE) 
#' pred <- downscaleCV.keras(x, y, model,
#'              sampling.strategy = "kfold.chronological", folds = 4, 
#'              scaleGrid.args = list(type = "standardize"),
#'              prepareData.keras.args = list(first.connection = "conv",
#'                                            last.connection = "dense",
#'                                            channels = "last"),
#'              compile.args = list(loss = bernouilliGammaLoss(last.connection = "dense"), 
#'                                  optimizer = optimizer_adam()),
#'              fit.args = list(batch_size = 100, epochs = 10, validation_split = 0.1),
#'              loss = "bernouilliGammaLoss",
#'              binarySerie = TRUE)
#' # Deterministic
#' pred_amo <- computeRainfall(log_alpha = subsetGrid(pred,var = "log_alpha"),
#'                             log_beta = subsetGrid(pred,var = "log_beta"),
#'                             bias = 0.99)
#' # Stochastic
#' pred_amo <- computeRainfall(log_alpha = subsetGrid(pred,var = "log_alpha"),
#'                             log_beta = subsetGrid(pred,var = "log_beta"),
#'                             simulate = TRUE,
#'                             bias = 0.99)
#' }                             
computeRainfall <- function(log_alpha=NULL,
                            log_beta=NULL,
                            simulate = FALSE,
                            bias = NULL) {
  log_alpha %<>% redim(log_alpha, drop = TRUE) %>% redim(member = TRUE)
  log_beta  %<>% redim(log_beta, drop = TRUE) %>% redim(member = TRUE)
  n.mem <- getShape(log_alpha,"member")
  out <- lapply(1:n.mem, FUN = function(z) {
    log_alpha %<>% subsetGrid(members = z)
    log_beta  %<>% subsetGrid(members = z)
    amo <- log_alpha
    dimNames <- attr(log_alpha$Data,"dimensions")
    if (isTRUE(simulate)) {
      ntime <- getShape(log_alpha,"time")
      if (isRegular(log_alpha)) {
      alpha_mat <- array3Dto2Dmat(exp(log_alpha$Data))
      beta_mat <- array3Dto2Dmat(exp(log_beta$Data))
      } else {
        alpha_mat <- exp(log_alpha$Data)
        beta_mat <- exp(log_beta$Data)
      }
      aux <- matrix(nrow = ntime,ncol = ncol(alpha_mat))
      for (zz in 1:ncol(alpha_mat)) {
        aux[,zz] <- rgamma(n = ntime, 
                           shape = alpha_mat[,zz], 
                           scale = beta_mat[,zz])
      }
      if (isRegular(log_alpha)) amo$Data <- mat2Dto3Darray(aux,x = amo$xyCoords$x, y = amo$xyCoords$y)
    } else {
      amo$Data <- exp(log_alpha$Data)*exp(log_beta$Data)   
    }
    if (!is.null(bias)) amo <- amo %>% gridArithmetics(bias,operator = "+")
    return(amo)
  }) %>% bindGrid() %>% redim(drop = TRUE)
  out$Variable$varName <- "pr"
  return(out)
}