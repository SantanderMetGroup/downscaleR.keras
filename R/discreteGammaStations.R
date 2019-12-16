discreteGammaStations <- custom_metric("custom_loss", function(true, pred){
  K <- backend()
  D <- K$int_shape(pred)[[2]]/3
  occurrence = pred[,1:D]
  shape_parameter = K$exp(pred[,(D+1):(D*2)])
  scale_parameter = K$exp(pred[,(D*2+1):(D*3)])
  bool_rain = K$cast(K$greater(true,0),K$tf$float32)
  epsilon = 0.000001
  return (- K$mean((1-bool_rain)*K$tf$log(1-occurrence+epsilon) + bool_rain*(K$tf$log(occurrence+epsilon) + (shape_parameter - 1)*K$tf$log(true+epsilon) - shape_parameter*K$tf$log(scale_parameter+epsilon) - K$tf$lgamma(shape_parameter+epsilon) - true/(scale_parameter+epsilon))))
})
