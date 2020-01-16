##     bernouilliGamma.statistics.R Custom loss function for BernouilliGamma distributions.
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

#' @title Computing statistics from a Bernouilli-Gamma distribution.
#' @description This function permits to sample from a Bernouill-Gamma distribution or to compute the expectance of the function
#' in the climate4R framework. 
#' @param p A string with values "conv" o "dense" depending on the type of the net's last connection. DEFAULT is NULL.
#' @param alpha A grid. The values are related to the shape parameter in the form: shape_parameter = exp(alpha).
#' @param beta A grid. The values are related to the scale parameter in the form: scale_parameter = exp(beta).
#' @param simulate A logical value. If TRUE then the output is an stochastic sample for the given distribution parameters (p,alpha and beta).
#' @param bias An integer. Added to the quantity of rain after its computation, either deterministically or
#' stochastically. It basically displaces the distribution. Default is NULL.
#' @details The output of a function is a multigrid object (see \code{\link[transformeR]{makeMultiGrid}}) with 2 variables: 
#'
#' @seealso 
#' bernouilliGamma.loss_function a custom loss function for Bernouilli-Gamma distributions
#' @details This function uses \code{tensorflow} internally. Please if not, install tensorflow (>= v.)
#' downscaleTrain.keras for training a downscaling deep model with keras
#' downscalePredict.keras for predicting with a keras model
#' prepareNewData.keras for predictor preparation with new (test) data
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' \href{https://github.com/SantanderMetGroup/downscaleR/wiki/training-downscaling-models}{downscaleR Wiki} for downscaling seasonal forecasting and climate projections.
#' @author J. Bano-Medina
#' @import tensorflow
#' @importFrom stats rgamma runif
#' @export
bernouilliGamma.statistics <- function(p,alpha,beta,
                                       simulate = FALSE,
                                       bias = NULL) {
  p <- redim(p, drop = TRUE) %>% redim(member = TRUE)
  alpha <- redim(alpha, drop = TRUE) %>% redim(member = TRUE)
  beta <- redim(beta, drop = TRUE) %>% redim(member = TRUE)
  n.mem <- getShape(p,"member")
  out <- lapply(1:n.mem, FUN = function(z) {
    p <- subsetGrid(p,members = z)
    alpha <- subsetGrid(alpha,members = z)
    beta <- subsetGrid(beta,members = z)
    amo <- alpha
    dimNames <- attr(p$Data,"dimensions")
    if (isTRUE(simulate)) {
      # p
      s <- p
      s$Data <- array(runif(getShape(p,"time")*getShape(p,"lat")*getShape(p,"lon"),min = 0,max = 1),
                      dim = c(getShape(p,"time"),getShape(p,"lat"),getShape(p,"lon")))
      attr(s$Data,"dimensions") <- c("time","lat","lon")
      p$Data <- (p$Data > s$Data)*1
      attr(p$Data,"dimensions") <- dimNames 
      
      # alpha and beta
      for (zz in 1:getShape(alpha,"lat")) {
        for (zzz in 1:getShape(alpha,"lon")) {
            amo$Data[,zz,zzz] <- rgamma(n = getShape(alpha,"time"), 
                                            shape = exp(alpha$Data[,zz,zzz]), 
                                            scale = exp(beta$Data[,zz,zzz]))
        }
      }
    } else {
      amo$Data <- exp(alpha$Data)*exp(beta$Data)   
    }
    
    if (!is.null(bias)) amo <- amo %>% gridArithmetics(bias,operator = "+")
    makeMultiGrid(p,amo)
  }) %>% bindGrid() %>% redim(drop = TRUE)
  out$Variable$varName <- c("probOfRain","amountOfRain")
  return(out)
}




# # Stochastic Prediction
# simulate_ocu <- function(dat,model,D){
#   ocu <- model$predict(dat)[,1:D,drop = FALSE]
#   sim <- matrix(runif(length(ocu),min = 0,max = 1), nrow = nrow(ocu), ncol = ncol(ocu))
#   cond <- ocu > sim
#   return(cond*1)
# }
# 
# simulate_reg <- function(dat,model,D) {
#   shape <- exp(model$predict(dat)[,(D+1):(D*2),drop = FALSE])
#   scale <- exp(model$predict(dat)[,(D*2+1):(D*3),drop = FALSE])
#   p <- matrix(nrow = nrow(dat),ncol = D)
#   for (i in 1:D) {
#     p[,i] <- rgamma(n = nrow(dat), shape = shape[,i], scale = scale[,i])
#   }
#   return(p)
# }
# predSTO_ocu <- simulate_ocu(x,model,D)
# predSTO_reg <- simulate_reg(x,model,D) + 1