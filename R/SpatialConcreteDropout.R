##     spatialConcreteDropout.R Custom class layer with concrete dropout.
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

#' @title Custom class layer with spatial concrete dropout.
#' @description This is a keras layer enabling to implement concrete dropout 
#' on the convolutinal layers of the deep learning model. 
#' @details This is an adaptation of the python spatial concrete dropout class
#' to the R software, developed in 
#' https://github.com/yaringal/ConcreteDropout/blob/master/spatial-concrete-dropout-keras.ipynb
#' @author J. Bano-Medina
#' @export
# R6 wrapper class, a subclass of KerasWrapper
spatialConcreteDropout <- R6::R6Class("spatialConcreteDropout",
                               
                               inherit = KerasWrapper,
                               
                               public = list(
                                 weight_regularizer = NULL,
                                 dropout_regularizer = NULL,
                                 init_min = NULL,
                                 init_max = NULL,
                                 is_mc_dropout = NULL,
                                 supports_masking = TRUE,
                                 p_logit = NULL,
                                 p = NULL,
                                 data.format = NULL,
                                 
                                 initialize = function(weight_regularizer,
                                                       dropout_regularizer,
                                                       init_min,
                                                       init_max,
                                                       is_mc_dropout,
                                                       data.format) {
                                   self$weight_regularizer <- weight_regularizer
                                   self$dropout_regularizer <- dropout_regularizer
                                   self$is_mc_dropout <- is_mc_dropout
                                   self$init_min <- k_log(init_min) - k_log(1 - init_min)
                                   self$init_max <- k_log(init_max) - k_log(1 - init_max)
                                   self$data.format <- data.format
                                 },
                                 
                                 build = function(input_shape) {
                                   super$build(input_shape)
                                   
                                   self$p_logit <- super$add_weight(
                                     name = "p_logit",
                                     shape = shape(1),
                                     initializer = initializer_random_uniform(self$init_min, self$init_max),
                                     trainable = TRUE
                                   )
                                   
                                   self$p <- k_sigmoid(self$p_logit)
                                   if (length(input_shape) != 4) stop("This wrapper only supports conv2D layers")
                                   input_dim <- ifelse(self$data.format == "channels_first",input_shape[[2]],input_shape[[4]])
 
                                   weight <- private$py_wrapper$layer$kernel
                                   
                                   kernel_regularizer <- self$weight_regularizer * 
                                     k_sum(k_square(weight)) / 
                                     (1 - self$p)
                                   
                                   dropout_regularizer <- self$p * k_log(self$p)
                                   dropout_regularizer <- dropout_regularizer +  
                                     (1 - self$p) * k_log(1 - self$p)
                                   dropout_regularizer <- dropout_regularizer * 
                                     self$dropout_regularizer * 
                                     k_cast(input_dim, k_floatx())
                                   
                                   regularizer <- k_sum(kernel_regularizer + dropout_regularizer)
                                   super$add_loss(regularizer)
                                 },
                                 
                                 spatial_concrete_dropout = function(x) {
                                   eps <- k_cast_to_floatx(k_epsilon())
                                   temp <- 2/3
                                   
                                   input_shape <- k_shape(x)
                                   if (self$data.format == "channels_first") {
                                     noise_shape <- c(input_shape[[0]],input_shape[[1]],1L,1L)
                                   } else {
                                     noise_shape <- list(input_shape[[0]],1L,1L,input_shape[[3]])
                                   }
                                   unif_noise <- k_random_uniform(shape = noise_shape)
                                   
                                   drop_prob <- k_log(self$p + eps) - 
                                     k_log(1 - self$p + eps) + 
                                     k_log(unif_noise + eps) - 
                                     k_log(1 - unif_noise + eps)
                                   drop_prob <- k_sigmoid(drop_prob / temp)
                                   
                                   random_tensor <- 1 - drop_prob
                                   
                                   retain_prob <- 1 - self$p
                                   x <- x * random_tensor
                                   x <- x / retain_prob
                                   x
                                 },
                                 
                                 call = function(x, mask = NULL, training = NULL) {
                                   if (self$is_mc_dropout) {
                                     super$call(self$spatial_concrete_dropout(x))
                                   } else {
                                     k_in_train_phase(
                                       function()
                                         super$call(self$spatial_concrete_dropout(x)),
                                       super$call(x),
                                       training = training
                                     )
                                   }
                                 }
                               )
)


#' @title Concrete dropout for convolutional keras layers.
#' @description This function is for instantiating custom spatial concrete dropout layer. 
#' It allows to learn the dropout probability for a given `Convolutional 2D' layer.
#' @param object 	Model or layer object
#' @param layer A layer instance.
#' @param weight_regularizer A positive number that penalizes the weights
#' of the layer according to L1 or L2 regularization. 
#' @param dropout_regularizer A positive number adequate to initialize the dropout probability
#' @param init_min .
#' @param init_max .
#' @param is_mc_dropout Wether Monte-Carlo dropout is applied or not. Default to TRUE.
#' @param name 	An optional name string for the layer. 
#' Should be unique in a model (do not reuse the same name twice). 
#' It will be autogenerated if it isn't provided.
#' @param trainable Whether the layer weights will be updated during training.
#' @details This is an adaptation of the python spatial concrete dropout class
#' to the R software, developed in 
#' https://github.com/yaringal/ConcreteDropout/blob/master/spatial-concrete-dropout-keras.ipynb
#' @author J. Bano-Medina
#' @export
# function for instantiating custom wrapper
layer_spatial_concrete_dropout <- function(object, 
                                   layer,
                                   weight_regularizer = 1e-6,
                                   dropout_regularizer = 1e-5,
                                   init_min = 0.1,
                                   init_max = 0.1,
                                   is_mc_dropout = TRUE,
                                   name = NULL,
                                   trainable = TRUE,
                                   data.format = "channels_last") {
  create_wrapper(spatialConcreteDropout, object, list(
    layer = layer,
    weight_regularizer = weight_regularizer,
    dropout_regularizer = dropout_regularizer,
    init_min = init_min,
    init_max = init_max,
    is_mc_dropout = is_mc_dropout,
    name = name,
    trainable = trainable,
    data.format = data.format
  ))
}
