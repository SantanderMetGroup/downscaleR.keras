discreteGammaGrid <- custom_metric("custom_loss", function(true, pred){
  K = backend()
  ocurrence = pred[,,,1, drop = FALSE]
  shape_parameter = K$exp(pred[,,,2, drop = FALSE])
  scale_parameter = K$exp(pred[,,,3, drop = FALSE])
  bool_rain = K$cast(K$greater(true,0),K$tf$float32)
  epsilon = 0.000001
  return (- K$mean((1-bool_rain)*K$tf$log(1-ocurrence+epsilon) + bool_rain*(K$tf$log(ocurrence+epsilon) + (shape_parameter - 1)*K$tf$log(true+epsilon) - shape_parameter*K$tf$log(scale_parameter+epsilon) - K$tf$lgamma(shape_parameter+epsilon) - true/(scale_parameter+epsilon))))
})