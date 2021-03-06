% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/downscalePredictHidden.keras.R
\name{downscalePredictHidden.keras}
\alias{downscalePredictHidden.keras}
\title{Obtain the output of the hidden layers in a deep learning keras model for downscaling.}
\usage{
downscalePredictHidden.keras(newdata, model, layer, clear.session)
}
\arguments{
\item{newdata}{The grid data. It should be an object as returned by  \code{\link[downscaleR.keras]{prepareNewData.keras}}.}

\item{model}{An object containing the statistical model as returned from  \code{\link[downscaleR.keras]{downscaleTrain.keras}} 
or a list of arguments passed to \code{\link[keras]{load_model_hdf5}}.}

\item{layer}{A string or numeric value. The output of which hidden layer should be returned?}

\item{clear.session}{A logical value. Indicates whether we want to destroy the current tensorflow graph and clear the model from memory.
In particular, refers to whether we want to use the function \code{\link[keras]{k_clear_session}} after training.
If FALSE, model is returned. If TRUE, then k_clear_session() is applied and no model is returned.
Default to FALSE.}
}
\value{
A list containing the filter maps and the associated kernels.
}
\description{
Obtain the output of the hidden layers in a deep learning keras model for downscaling, previously infered with \code{\link[downscaleR.keras]{downscaleTrain.keras}}
Also, returns the kernel of the corresponding hidden layer that generated the output.
}
\details{
This function relies on keras, which is a high-level neural networks API capable of running on top of tensorflow, CNTK or theano.
There are official \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep learning models. We suggest the user, especially the beginners,
to consult these tutorials before using downscalePredictHidden.keras.
}
\seealso{
downscaleTrain.keras for training a downscaling deep model with keras
downscalePredict.keras for predicting with a keras model
prepareNewData.keras for predictor preparation with new (test) data
\href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki}
}
\author{
J. Bano-Medina
}
