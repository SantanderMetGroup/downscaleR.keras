##     relevanceMaps.R Obtain relevance maps as a climate4R object
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

#' @title Obtain relevance maps as a climate4R object
#' @description Relevance maps are a representation of the influence of the predictor variables on the downscaling over 
#' a particular gridpoint on a certain day. The relevance maps are obtained using 
#' \href{https://arxiv.org/pdf/1702.04595.pdf}{prediction difference analysis}
#' on a trained deep model (see \code{\link[downscaleR.keras]{downscaleTrain.keras}}).
#' @param x A grid (usually a multigrid) of predictor fields.
#' @param obj The object as returned by \code{\link[downscaleR.keras]{prepareData.keras}}.
#' @param model A keras sequential or functional model. 
#' @param C4R.template A climate4R grid that serves as template for the returned prediction object.
#' @param outputCoords A matrix. The coordinates of the predictand(s) gridpoints for whom the
#' prediction difference analysis is desired. Longitudes in the first column and latitudes in the second column.
#' @param parch Possible values are c("all","variable","channel"). Indicates whether we want to marginalize the influence of
#' a certain gridpoint as a whole ("all"), to isolate the influence per variable 
#' ("variable", i.e., for example to isolate the influence of the specific humidity at all levels: hus500,hus700,...) 
#' or of every channel (i.e., "channel") independently.
#' @param k A numeric value. Defines the domain to marginalize around a particular feature with coordinates (i,j). 
#' Therefore the new domain has dimensions (i-k:i+k,j-k:j+k). If k=1, then a region of 3x3 is simultaneously marginalized 
#' at every step.
#' @param l A numeric value. Defines the domain ((2l+1)x(2l+1)) used to infer the conditional multivariate gaussian distribution
#' @param num_samples A numeric value. How many times do we sample from the multivariate gaussian distribution?
#' @param loss Default to NULL. Otherwise a string indicating the loss function used to train the model. This is only
#' relevant where we have used the 2 custom loss functions of this library: "gaussianLoss" or "bernouilliGammaLoss"
#' @details This function relies on keras, which is a high-level neural networks API capable of running on top of tensorflow, CNTK or theano.
#' There are official \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep learning models. We suggest the user, especially the beginners,
#' to consult these tutorials before using the downscaleR.keras package. Moreover, we encourage the reader to consult 
#' the prediction difference analysis technique employed 
#' which is published in this \href{https://arxiv.org/pdf/1702.04595.pdf}{paper}. 
#'  
#' @return A climate 4R object containing the relevance maps as members. The attribute attr(out,"memberCoords") is added to the climate4R output object
#' containing the coordinates in sequential order, such as the relevance map of the first member measures the influence
#' on the first coordinate in attr(out,"memberCoords").
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @author J. Bano-Medina
#' @family downscaling.functions
#' @importFrom MASS mvrnorm ginv
#' @importFrom abind abind
#' @importFrom transformeR aggregateGrid mergeGrid interpGrid subsetDimension
#' @export
relevanceMaps <- function(x,obj,
                          C4R.template,
                          model,
                          outputCoords,
                          parch = c("channel","variable","all"),
                          k = 0, l = 2, num_samples = 10,loss = NULL) {
  
  ntime <- getShape(x,"time")
  nl <- getShape(x,"lat")
  nL <- getShape(x,"lon")
  nv <- getShape(x,"var")
  if (parch == "all") {n <- list(1:nv)}
  if (parch == "channel")  {n <- 1:nv}
  if (parch == "variable") {
    vars <- unique(attr(x$Variable, "description"))
    n <- lapply(vars, FUN = function(z) which(attr(x$Variable, "description") == z))
  }
  
  pKnown <- prepareNewData.keras(x,obj) %>% downscalePredict.keras(model,clear.session = TRUE,C4R.template = C4R.template, loss = loss)  
  if (loss == "bernouilliGammaLoss") {
    pKnown <- lapply(c("p","log_alpha","log_beta"),FUN = function(z) interpGrid(subsetGrid(pKnown,var = z),new.coordinates = list(x = outputCoords[,1],y = outputCoords[,2]))) %>% 
      makeMultiGrid() 
    rainfall <- computeRainfall(log_alpha = subsetGrid(pKnown,var = "log_alpha"),
                                log_beta  = subsetGrid(pKnown,var = "log_beta"))
    pKnown <- gridArithmetics(subsetGrid(pKnown,var = "p"),rainfall)
    rainfall <- NULL
  } else if (loss == "gaussianLoss"){
    pKnown <- subsetGrid(pKnown, var = "mean") %>% interpGrid(new.coordinates = list(x = outputCoords[,1],y = outputCoords[,2]))
  } else {
    pKnown <- interpGrid(pKnown,new.coordinates = list(x = outputCoords[,1],y = outputCoords[,2]))
  }
  for (z in 1:nl) {
    gc()
    for (zz in 1:nL) {
      nn <- 0
      for (zzz in n) {
        nn <- nn + 1
        gc()
        print(paste(z,"out of",length(1:nl)))
        print(paste(zz,"out of",length(1:nL)))
        
        ind_zk <- ((z-k):(z+k))[which((z-k):(z+k) > 0 & (z-k):(z+k) <= nl)]
        ind_zzk <- ((zz-k):(zz+k))[which((zz-k):(zz+k) > 0 & (zz-k):(zz+k) <= nL)]
        ind_zl <- ((z-l):(z+l))[which((z-l):(z+l) > 0 & (z-l):(z+l) <= nl)]
        ind_zzl <- ((zz-l):(zz+l))[which((zz-l):(zz+l) > 0 & (zz-l):(zz+l) <= nL)]
        xk <- x$Data[zzz,,ind_zk,ind_zzk,drop = FALSE] 
        xl <- x$Data[,,ind_zl,ind_zzl,drop = FALSE]
        xw <- rep(list(redim(x,member = TRUE)),num_samples) %>% bindGrid(dimension = "member")
        xw$Data[zzz,1:num_samples,,ind_zk,ind_zzk] <- sampleMultivariateGaussian(xk,xl,num_samples)
        attr(xw$Data,"dimensions") <- c("var","member","time","lat","lon")
        xk <- NULL; xl <- NULL
        gc()
        pUnknown <- prepareNewData.keras(xw,obj) %>%  
          downscalePredict.keras(model,C4R.template,clear.session = TRUE, loss = loss) 
        xw <- NULL
        gc()
        if (loss == "bernouilliGammaLoss") {
          pUnknown <- lapply(c("p","log_alpha","log_beta"),FUN = function(z) interpGrid(subsetGrid(pUnknown,var = z),new.coordinates = list(x = outputCoords[,1],y = outputCoords[,2]))) %>% 
            makeMultiGrid() 
          rainfall <- computeRainfall(log_alpha = subsetGrid(pUnknown,var = "log_alpha"),
                                      log_beta  = subsetGrid(pUnknown,var = "log_beta"))
          pUnknown <- gridArithmetics(subsetGrid(pUnknown,var = "p"),rainfall)
          rainfall <- NULL       
        } else if (loss == "gaussianLoss"){
          pUnknown <- subsetGrid(pUnknown, var = "mean") %>% interpGrid(new.coordinates = list(x = outputCoords[,1],y = outputCoords[,2]))
        } else {
          pUnknown <- interpGrid(pUnknown,new.coordinates = list(x = outputCoords[,1],y = outputCoords[,2]))
        } 
        pUnknown <- aggregateGrid(pUnknown,aggr.mem = list(FUN = "mean", na.rm = TRUE)) %>%
          redim(drop = TRUE)
        infl <- gridArithmetics(pUnknown,pKnown,operator = "-")
        pUnknown <- NULL
        gc()
        out <- subsetGrid(x,var = getVarNames(x)[zzz[1]],
                          latLim = c(x$xyCoords$y[ind_zk][1],x$xyCoords$y[ind_zk][length(ind_zk)]),
                          lonLim = c(x$xyCoords$x[ind_zzk][1],x$xyCoords$x[ind_zzk][length(ind_zzk)])) %>% 
          redim(var = TRUE,member = FALSE)
        out <- lapply(1:nrow(outputCoords),FUN = function(mem) {
          for (zk in 1:length(ind_zk)) {
            for (zzk in 1:length(ind_zzk)) {
              out$Data[1,,zk,zzk] <- subsetDimension(infl,dimension = "loc",indices = mem)$Data
            }
          }
          attr(out$Data,"dimensions") <- c("var","time","lat","lon")
          gc()
          return(out)
        }) %>% bindGrid(dimension = "member")
        gc()
        if (nn < 10) {nnn <- paste0(0,nn)} else {nnn <- nn}
        save(out,file = paste0("./chunk_",z,"_",zz,"_",nnn,".rda"))
        out <- NULL; infl <- NULL
        k_clear_session()
        gc()
      }
      k_clear_session()
      gc()
    } 
    k_clear_session()
    gc()
  }
  k_clear_session()
  gc()
  for (z in 1:nl) {
    for (zz in 1:nL) {
      lf <- list.files(".", pattern =  paste0("chunk_",z,"_",zz,"_"), full.names = TRUE)
      out <- lapply(lf, function(z) mget(load(z))) %>% unlist(recursive = FALSE) %>% makeMultiGrid()
      save(out, file = paste0("chunk_",z,"_",zz,".rda"))
      file.remove(lf)
    }
  }
  
  lf <- list.files(".", pattern =  paste0("chunk_"), full.names = TRUE)
  out <- lapply(lf, function(z) mget(load(z))) %>% unlist(recursive = FALSE)
  out <- lapply(1:ntime, FUN = function(zz){
    lapply(1:length(out), FUN = function(z){
      subsetDimension(out[[z]],dimension = "time", indices = zz)
    }) %>% mergeGrid(aggr.fun = list(FUN = "mean",na.rm = TRUE))
  }) %>% bindGrid(dimension = "time")
  file.remove(lf)
  
  attr(out,"memberCoords") <- list("x" = outputCoords[,1],"y" = outputCoords[,2])
  k_clear_session()
  return(out)
}

#' @title Sample from a multivariate conditional distribution.
#' @description Sample from a multivariate conditional distribution such that p(xk|xl).
#' @param xk Predictors of a domain of size ((2k+1)x(2k+1))
#' @param xl Predictors of a domain of size ((2l+1)x(2l+1))
#' @param num_samples A numeric value. How many times do we sample from the multivariate gaussian distribution? 
#' @return A nested list of 2D matrices with the following structure: sites/members
#' @keywords internal
#' @author J. Baño-Medina    
sampleMultivariateGaussian <- function(xk,xl,num_samples) {
  xk <- aperm(xk,c(2,3,4,1)) 
  dims <- dim(xk)
  dim(xk) <- c(dim(xk)[1],prod(dim(xk)[2:4]))
  xl <- aperm(xl,c(2,3,4,1)) 
  dim(xl) <- c(dim(xl)[1],prod(dim(xl)[2:4]))
  ind_k <- sapply(1:dim(xk)[2], FUN = function(zz) {
    which(sapply(1:dim(xl)[2], FUN = function(z) all(xk[,zz] == xl[,z])))
  })
  ind_l <- setdiff(1:dim(xl)[2],ind_k)
  paramJoint <- jointProbDist(xl)
  paramCond <- condProbDist(ind_k,ind_l,xl,paramJoint)
  n <- dim(paramCond$means)[1]
  xs <- lapply(1:n, FUN = function(z) MASS::mvrnorm(n = num_samples, paramCond$means[z,], paramCond$cov, tol = 10000)) %>% 
    abind(along = 3) %>% aperm(c(1,3,2))
  dim(xs) <- c(num_samples,dims)
  xs <- aperm(xs,c(5,1,2,3,4)) 
  return(xs)
}
#' @title Infer a multivariate conditional distribution.
#' @description Infer a multivariate conditional distribution such that p(xk|xl).
#' @param index_k Indices of the predictors of domain k.
#' @param index_l Indices of the predictors of domain k.
#' @param xl Predictors of domain l.
#' @param paramJoint Parameters from the joint gaussian distribution.
#' @return Parameters (means and covariance matrix) from the multivariate joint gaussian distribution.
#' @keywords internal
#' @author J. Baño-Medina
condProbDist <- function(index_k,index_l,xl,paramJoint) {
  nu1 <- paramJoint$means[index_k]
  nu2 <- paramJoint$means[index_l]
  cov11 <- paramJoint$cov[index_k,index_k]
  cov12 <- paramJoint$cov[index_k,index_l]
  cov21 <- paramJoint$cov[index_l,index_k]
  cov22 <- paramJoint$cov[index_l,index_l]
  gcov22 <- MASS::ginv(cov22)
  a <- xl[,index_l]
  meansCond <- (nu1 + cov12%*%gcov22%*%(t(a - nu2))) %>% t()
  covCond <- cov11 - cov12 %*% gcov22 %*% cov21
  return(list("means" = meansCond, "cov" = covCond))
}
#' @title Infer a multivariate joint distribution.
#' @description Infer a multivariate joint distribution such that p(xl).
#' @param xl Predictors of domain l.
#' @return Parameters (means and covariance matrix) from the multivariate joint gaussian distribution.
#' @keywords internal
#' @importFrom stats cov
#' @author J. Baño-Medina
jointProbDist <- function(xl){
  meansJoint <- apply(xl,MARGIN = 2, mean)
  covJoint <- cov(xl)
  return(list("means" = meansJoint, "cov" = covJoint))
}

