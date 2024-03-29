% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/computeTemperature.R
\name{computeTemperature}
\alias{computeTemperature}
\title{Sample from the daily conditional gaussian distributions.}
\usage{
computeTemperature(mean = NULL, log_var = NULL)
}
\arguments{
\item{mean}{A grid. The mean of the daily gaussian conditional distributions, as returned by downscalePredict.keras when we use the gaussian loss function.}

\item{log_var}{A grid. The log of the variance ("log_beta") of the daily conditional distributions, as returned by downscalePredict.keras when we use the gaussian loss function.}
}
\value{
A climate4R grid with the deterministic or stochastic temporal serie 
depending on whether the \code{simulate} parameter is FALSE or TRUE.
}
\description{
Compute the stochastic temperature given the parameters of conditional gaussian distributions
as returned by \code{\link[downscaleR.keras]{downscalePredict.keras}} on a trained model 
(see \code{\link[downscaleR.keras]{downscaleTrain.keras}})
that optimized the loss function \code{\link[downscaleR.keras]{gaussianLoss}}.
}
\examples{
\donttest{
require(climate4R.datasets)
require(transformeR)
data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
data("VALUE_Iberia_tas")
y <- VALUE_Iberia_tas
inputs <- layer_input(shape = c(getShape(x,"lat"),getShape(x,"lon"),getShape(x,"var")))
hidden <- inputs \%>\% 
  layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') \%>\%  
  layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') \%>\% 
  layer_flatten() \%>\% 
  layer_dense(units = 10, activation = "relu")
outputs1 <- layer_dense(hidden,units = getShape(y,"loc"))
outputs2 <- layer_dense(hidden,units = getShape(y,"loc"))
outputs <- layer_concatenate(list(outputs1,outputs2))
model <- keras_model(inputs = inputs, outputs = outputs)
pred <- downscaleCV.keras(x, y, model,
sampling.strategy = "kfold.chronological", folds = 4, 
scaleGrid.args = list(type = "standardize"),
prepareData.keras.args = list(first.connection = "conv",
                              last.connection = "dense",
                              channels = "last"),
compile.args = list(loss = gaussianLoss(last.connection = "dense"), 
                    optimizer = optimizer_adam()),
fit.args = list(batch_size = 100, epochs = 10, validation_split = 0.1),
                          loss = "gaussianLoss")

# We sample from the daily conditional gaussian distributions
pred <- computeTemperature(mean = subsetGrid(pred,var = "mean"),
                           log_var = subsetGrid(pred,var = "log_var"))
}
}
\seealso{
bernouilliGammaLoss a custom loss function for Bernouilli-Gamma distributions
gaussianLoss a custom loss function for gaussian distributions
downscaleTrain.keras for training a downscaling deep model with keras
downscalePredict.keras for predicting with a keras model
prepareNewData.keras for predictor preparation with new (test) data
}
\author{
J. Bano-Medina
}
