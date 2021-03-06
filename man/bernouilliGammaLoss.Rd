% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bernouilliGammaLoss.R
\name{bernouilliGammaLoss}
\alias{bernouilliGammaLoss}
\title{Custom loss function for Bernouilli-Gamma distributions.}
\usage{
bernouilliGammaLoss(last.connection = NULL)
}
\arguments{
\item{last.connection}{A string with values "conv" o "dense" depending on the type of the net's last connection. DEFAULT is NULL.}
}
\description{
This loss function optimizes the negative log-likelihood of a Bernouill-Gamma distribution. It is a custom
loss function defined according to keras functions.
}
\details{
Note that infering a conditional Bernouilli-Gamma distribution means to estimate their associated parameters:
probability, shape and scale factor. To avoid computational instabilities
we actually estimate log alpha and log beta. Therefore, make sure that your output layers are designed according
to this property of the distribution.
}
\seealso{
bernouilliGammaStatistics for computing the expectance or simulate from the discrete continuous distribution
downscaleTrain.keras for training a downscaling deep model with keras
downscalePredict.keras for predicting with a keras model
prepareNewData.keras for predictor preparation with new (test) data
\href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
\href{https://github.com/SantanderMetGroup/downscaleR/wiki/training-downscaling-models}{downscaleR Wiki} for downscaling seasonal forecasting and climate projections.
}
\author{
J. Bano-Medina
}
