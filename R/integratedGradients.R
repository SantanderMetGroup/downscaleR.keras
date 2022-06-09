##     integratedGradients.R Computes integrated gradients for explainability of neural networks.
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

#' @title Computes integrated gradients for a neural network in the C4R framework.  
#' @description Given a neural network, computes integrated gradients for a particular predictand (or 
#' output neuron of the model) w.r.t an input predictand field, to gain explainability. The integrated
#' gradients method is described in Sundarayan et al., 2017 (see References section).
#' @param x The input climate4R object or predictor field.
#' @param model A keras sequential or functional model. 
#' @param baseline The integrated gradients method attributes the prediction 
#' at input 'x' relative to a 'baseline', computing the contribution of 'x'
#' to the prediction. The \code{baseline} parameter defines this baseline, . 
#' Default to NULL which set the baseline to a 0 array. For custom baselines, 
#' input an array with the dimensions matching those of the input layer of
#' the neural network.
#' @param num_steps Number of interpolation steps between the baseline
#' and the input used in the computation of integrated gradients. These
#' steps along determine the integral approximation error. By default,
#' \code{num_steps} is set to 50. The authors suggest an interval from 20 to 300. 
#' @param model.info List of arguments containing metadata of the neural network. 
#' \itemize{
#'  \item @param first.connection A string. Possible values are c("dense","conv") depending on whether 
#'  \item @param last.connection A string. Same as \code{first.connection} but for the last connection
#' (i.e., last hidden layer to output layer).
#'  \item @param channels A string. Possible values are c("first","last") and indicates the dimension of the channels (i.e., climate variables)
#' in the array. If "first" then dimensions = c("channel","latitude","longitude") for regular grids or c("channel","loc") for irregular grids.
#'  \item @param time.frames The number of time frames to build the recurrent neural network. If e.g., time.frame = 2, then the value 
#' y(t) is a function of x(t) and x(t-1). The time frames stack in the input array prior to the input neurons or channels (in conv. layers). 
#' See \code{\link[keras]{layer_simple_rnn}},\code{\link[keras]{layer_lstm}} or \code{\link[keras]{layer_conv_lstm_2d}}. 
#'  \item @param nature An attribute as returned by \code{\link[downscaleR.keras]{prepareData.keras}}.
#'  \item @param ind_TrainingPredictandSites An attribute as returned by \code{\link[downscaleR.keras]{prepareData.keras}}.
#'  \item @param ind_TrainingPredictorSites An attribute as returned by \code{\link[downscaleR.keras]{prepareData.keras}}.
#'  \item @param data.structure An attribute as returned by \code{\link[downscaleR.keras]{prepareData.keras}}.
#'  \item @param coords A data frame containing the 'x' and 'y' coordinates of 
#' all the predictand sites represented in the output layer of the neural network. 
#' }  
#' @param site A data frame containing the 'x' and 'y' coordinates of 
#' the desired site where to compute the gradients.
#' e.g., site = data.frame("x" = -3.82, "y" = 43.46)
#' @param saliency.fun Apply a function to the resulting saliency maps. 
#' e.g., saliency.fun = list(FUN = "mean", na.rm = TRUE). 
#' @param batch An integer indicating the size of the batch. Default to NULL.
#' @details This function relies on keras, which is a high-level neural networks 
#' API capable of running on top of tensorflow, CNTK or theano. There are official 
#' \href{https://keras.rstudio.com/}{keras tutorials} regarding how to build deep 
#' learning models. We suggest the user, especially the beginners,
#' to consult these tutorials before using downscaleR.keras.
#' @references
#' \itemize{
#'  \item Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 
#'  "Axiomatic attribution for deep networks." 
#'  International conference on machine learning. PMLR, 2017. 
#' }
#' @return The integrated gradients in a climate4R object.
#' @seealso 
#' relevanceMaps for computing saliency maps based on prediction difference analysis
#' downscaleTrain.keras to train neural networks in the C4R framework
#' \href{https://github.com/SantanderMetGroup/downscaleR.keras/wiki}{downscaleR.keras Wiki} 
#' @author J. Bano-Medina
#' @import keras
#' @importFrom transformeR climatology
#' @importFrom magrittr %<>% %>%
#' @importFrom abind abind
#' @export
#' @examples \donttest{
#' require(climate4R.datasets)
#' require(transformeR)
#' require(magrittr)
#' 
#' data("NCEP_Iberia_hus850", "NCEP_Iberia_psl", "NCEP_Iberia_ta850")
#' x <- makeMultiGrid(NCEP_Iberia_hus850, NCEP_Iberia_psl, NCEP_Iberia_ta850)
#' data("VALUE_Iberia_tas")
#' y <- VALUE_Iberia_tas
#' 
#' # Preparing the predictors
#' x_scaled <- scaleGrid(x, type = "standardize")
#' data <- prepareData.keras(x = x_scaled, 
#'                           y = y, 
#'                           first.connection = "conv",
#'                           last.connection = "dense",
#'                           channels = "last")
#' 
#' # Defining the keras model.... 
#' # We define 3 hidden layers that consists on 
#' # 2 convolutional steps followed by a dense connection.
#' input_shape  <- dim(data$x.global)[-1]
#' output_shape  <- dim(data$y$Data)[2]
#' inputs <- layer_input(shape = input_shape)
#' hidden <- inputs %>% 
#'   layer_conv_2d(filters = 25, kernel_size = c(3,3), activation = 'relu') %>%  
#'   layer_conv_2d(filters = 10, kernel_size = c(3,3), activation = 'relu') %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 10, activation = "relu")
#' outputs <- layer_dense(hidden,units = output_shape)
#' model <- keras_model(inputs = inputs, outputs = outputs)
#' 
#' # Training model.... 
#' model <- downscaleTrain.keras(obj = data, 
#'                               model = model,
#'                               compile.args = list("loss" = "mse", 
#'                                                   "optimizer" = optimizer_adam(lr = 0.01)),
#'                               fit.args = list("epochs" = 150, "batch_size" = 100), 
#'                               clear.session = FALSE)
#' 
#' # Choose site.... 
#' site <- 5 
#' xCoord <- y$xyCoords$x[site]
#' yCoord <- y$xyCoords$y[site]
#' 
#' # Compute the saliency maps for 
#' # the selected site....
#' saliency_grids <- integratedGradients(x = x_scaled,
#'                                       model = model,
#'                                       baseline = NULL,
#'                                       num_steps = 500,
#'                                       model.info = list(first.connection            = "conv",
#'                                                         last.connection             = "dense",
#'                                                         channels                    = "last",
#'                                                         time.frames                 = NULL,
#'                                                         nature                      = NULL,
#'                                                         ind_TrainingPredictandSites = attr(data, "indices_noNA_y"),
#'                                                         ind_TrainingPredictorSites  = NULL,
#'                                                         data.structure = NULL,
#'                                                         coords = y$xyCoords),
#'                                       site = data.frame("x" = xCoord, "y" = yCoord),
#'                                       saliency.fun = list(FUN = "mean", na.rm = TRUE)
#' )
#' 
#' # Display the saliency maps for 
#' # the selected site....
#' require(visualizeR)
#' require(sp)
#' spatialPlot(climatology(saliency_grids, clim.fun = list(FUN = "abs")), 
#'             backdrop.theme = "coastline",
#'             sp.layout = list(list(SpatialPoints(y$xyCoords[site,]),
#'                                   first = FALSE,
#'                                   col = "black",
#'                                   pch = 16)
#'             )
#' )
#' }
integratedGradients <- function(x = x,
                                model = model,
                                baseline = NULL,
                                num_steps = 50,
                                model.info = list(first.connection            = "conv",
                                                  last.connection             = "dense",
                                                  channels                    = "last",
                                                  time.frames                 = NULL,
                                                  nature                      = NULL,
                                                  ind_TrainingPredictandSites = NULL,
                                                  ind_TrainingPredictorSites  = NULL,
                                                  data.structure = NULL,
                                                  coords = NULL
                                ),
                                site = NULL,
                                saliency.fun = NULL,
                                batch = NULL) {
  
  ### Eliminate the 'member' dimension
  if (getShape(x, "member") > 1) {
    stop("No multi-member grids are allowed. Please consider using subsetGrid.")
  } else {
    x %<>% redim(drop = TRUE, member = FALSE)
  }
  
  ### Prepare input 'x' data based on `model.info` arguments
  data.structure <- list()
  data.structure$x.global <- list()
  attr(data.structure, "first.connection") <- model.info[["first.connection"]]
  attr(data.structure, "last.connection") <- model.info[["last.connection"]]
  attr(data.structure,"time.frames") <- model.info[["time.frames"]]
  attr(data.structure, "channels") <- model.info[["channels"]]
  attr(data.structure, "first.connection") <- model.info[["first.connection"]]
  attr(data.structure, "nature") <- model.info[["nature"]]
  attr(data.structure, "indices_noNA_y") <- model.info[["ind_TrainingPredictandSites"]]
  attr(data.structure, "indices_noNA_x") <- model.info[["ind_TrainingPredictorSites"]]
  attr(data.structure$x.global,"data.structure") <- model.info[["data.structure"]]
  
  if (model.info[["first.connection"]] == "conv") {
    x_input <- prepareNewData.keras(x, data.structure = data.structure)$x.global$member_1
  } else {
    ### to do: for local and spatial predictors in dense networks...
    
  }
  
  ### Match desired 'x,y' coordinated with output neuron
  output_neuron <- if (is.data.frame(site)) { 
    
    if (is.data.frame(model.info[["coords"]])) {### for irregular predictand grids
      coords_trainingSites <- model.info$coords[model.info[["ind_TrainingPredictandSites"]], ]
      ind_x <- which(site$x == coords_trainingSites$x)
      ind_y <- which(site$y == coords_trainingSites$y)
      intersect(ind_x, ind_y)
      
    } else if (is.list(model.info[["coords"]])) { ### for regular predictand grids
      ind_coords_x <- which(site$x == model.info[["coords"]]$x)
      ind_coords_y <- which(site$y == model.info[["coords"]]$y)
      aux_mat <- array(FALSE, dim = c(length(model.info[["coords"]]$x), length(model.info[["coords"]]$y)))
      aux_mat[ind_coords_x, ind_coords_y] <- TRUE
      aux_vector <- as.vector(aux_mat)[model.info[["ind_TrainingPredictandSites"]]]
      out_neuron <- which(aux_vector)
      if (length(out_neuron) == 0) {
        stop("The selected site was not optimized during the training phase and 
            therefore is not represented by any output neuron of the neural
            network.") 
      } else {
        out_neuron 
      }
    }  
    
  } else {
    site
  }
  
  ### Compute the integrated gradients
  array_int_grads <- if (is.null(batch)) {
    get_integrated_gradients(input = x_input, 
                             site = output_neuron, 
                             model = model,
                             baseline = baseline, 
                             num_steps = num_steps) 
  } else {
    samples <- dim(x_input)[1]
    init_batches <- seq(1, samples, batch)
    end_batches <- c(seq(batch, samples, batch), samples) %>% unique()
    mapply(init_batches, end_batches, FUN = function(init_batch, end_batch) {
      print(sprintf("Batch %i/%i", which(init_batch == init_batches), length(init_batches)))
      get_integrated_gradients(input = x_input[init_batch:end_batch,,,], 
                               site = output_neuron, 
                               model = model,
                               baseline = baseline, 
                               num_steps = num_steps)      
    }) %>% abind::abind(along = 1)
  }
  
  ### Verify the axiom of completeness
  # axiom_completeness(input = x_input,
  #                    site = output_neuron,
  #                    model = model,
  #                    baseline = NULL,
  #                    integrated.gradients = array_int_grads)
  
  ### Store the integrated gradients in a climate4R object
  if (getShape(x, "var") > 1) {
    if (length(dim(x_input)) > 3) { ### 3D predictor objects ("var", "lat", "lon")
      if (model.info[["channels"]] == "last") {
        array_int_grads %<>% aperm(c(4,1,2,3))
      } else if (model.info[["channels"]] == "first") {
        array_int_grads %<>% aperm(c(2,1,3,4)) 
      }
    } else { ### 2D predictor objects ("var", "loc")
      if (model.info[["channels"]] == "last") {
        array_int_grads %<>% aperm(c(3,1,2))
      } else if (model.info[["channels"]] == "first") {
        array_int_grads %<>% aperm(c(2,1,3)) 
      }
    }
  }
  
  c4r_int_grads <- x
  c4r_int_grads$Data <- array_int_grads
  attr(c4r_int_grads$Data, "dimensions") <- attr(x$Data, "dimensions")
  
  ### Apply saliency aggregated function
  if (! is.null(saliency.fun)) c4r_int_grads %<>% climatology(clim.fun = saliency.fun)
  
  ### Return
  return(c4r_int_grads)  
}



#' @title Computes the gradients of the neural network.  
#' @description Given a neural network, computes the gradients for a particular predictand (or 
#' output neuron of the model) w.r.t an input predictand field.
#' @param input The predictor field in matrix/array format.
#' @param model A keras sequential or functional model. 
#' @param site A data frame containing the 'x' and 'y' coordinates of 
#' the desired site where to compute the gradients.
#' e.g., site = data.frame("x" = -3.82, "y" = 43.46)
#' @return A matrix/array of the gradients of the predictions w.r.t input
#' @author J. Bano-Medina
#' @import keras
get_gradients <- function(input, site, model) {
  input <- tf$cast(input, tf$float32)
  
  with(tf$GradientTape() %as% tape, {
    tape$watch(input)
    preds <- model(input)[,site]
  })
  
  grads <- tape$gradient(preds, input)
}



#' @title Computes the integrated gradients of the neural network.  
#' @description Given a neural network, computes the integrated gradients for a particular predictand (or 
#' output neuron of the model) w.r.t an input predictand field.
#' @param input The input climate4R object or predictor field.
#' @param model A keras sequential or functional model. 
#' @param baseline The integrated gradients method attributes the prediction 
#' at input 'x' relative to a 'baseline', computing the contribution of 'x'
#' to the prediction. The \code{baseline} parameter defines this baseline, . 
#' Default to NULL which set the baseline to a 0 array. For custom baselines, 
#' input an array with the dimensions matching those of the input layer of
#' the neural network.
#' @param num_steps Number of interpolation steps between the baseline
#' and the input used in the computation of integrated gradients. These
#' steps along determine the integral approximation error. By default,
#' \code{num_steps} is set to 50. The authors suggest an interval from 20 to 300. 
#' @param site A data frame containing the 'x' and 'y' coordinates of 
#' the desired site where to compute the gradients.
#' e.g., site = data.frame("x" = -3.82, "y" = 43.46)
#' @return A matrix/array of the integrated gradients of the predictions w.r.t input
#' @author J. Bano-Medina
#' @import keras
get_integrated_gradients <- function(input, model, baseline = NULL, num_steps = 50, site) {
  # If baseline is not provided, start with a zero field
  # having same dimensions as the input field.
  if (is.null(baseline)) {
    predictor_dims <- dim(input)
    baseline <- baseline_array <- array(0, dim = predictor_dims)
  }
  
  
  # 1. Do interpolation.
  interpolated_input <- list()
  for (step in 1:num_steps) interpolated_input[[step]] <- baseline + (step / num_steps) * (input - baseline)
  
  
  # 2. Get the gradients
  grads = list()
  for (index_int_input in 1:length(interpolated_input))  {
    grads[[index_int_input]] <- get_gradients(input = interpolated_input[[index_int_input]], 
                                              site = site,
                                              model = model)
  }
  grads <- tf$convert_to_tensor(grads, dtype = tf$float32)
  
  # 3. Approximate the integral using the trapezoidal rule
  grads = (grads[1:(num_steps-1),,,,] + grads[2:num_steps,,,,]) / 2.0  ### generalizar esto para inputs que no sean 4D!!!
  avg_grads = tf$reduce_mean(grads, axis = 0L)
  
  # 4. Calculate integrated gradients and return
  integrated_grads <- (input - baseline) * avg_grads
  return(integrated_grads %>% as.array())
}

#' @title Verifies the axiom of completeness.  
#' @description Computes the difference in the prediction 
#' at input 'x' and the prediction at a 'baseline' and compare it
#' with the sum of the integrated gradients.
#' @param input The predictor field in matrix/array format.
#' @param model A keras sequential or functional model. 
#' @param baseline The integrated gradients method attributes the prediction 
#' at input 'x' relative to a 'baseline', computing the contribution of 'x'
#' to the prediction. The \code{baseline} parameter defines this baseline, . 
#' Default to NULL which set the baseline to a 0 array. For custom baselines, 
#' input an array with the dimensions matching those of the input layer of
#' the neural network.
#' @param site A data frame containing the 'x' and 'y' coordinates of 
#' the desired site where to compute the gradients.
#' e.g., site = data.frame("x" = -3.82, "y" = 43.46)
#' @param integrated.gradients An array/matrix of integrated gradients.
#' @return A matrix/array of the gradients of the predictions w.r.t input
#' @author J. Bano-Medina
#' @import keras
axiom_completeness <- function(input, baseline, model, site, integrated.gradients) {
  if (is.null(baseline)) {
    predictor_dims <- dim(input)
    baseline <- baseline_array <- array(0, dim = predictor_dims)
  }
  
  diff <- model$predict(input)[, site] - model$predict(baseline)[, site] 
  apply(integrated.gradients, MARGIN = 1, sum) - diff
}