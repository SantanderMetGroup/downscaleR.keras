# downscaleR.keras
downwscaleR.keras is an R package for statistical downscaling of daily data based on neural networks. This library integrates the deep learning package [Keras](https://keras.rstudio.com/) in the climate4R framework. This permits to incorporate sophisticated convolutional or recurrent neural networks, among others, to the climate4R battery of downscaling methods. Some degree of knowledge regarding the use of keras is required and we refer the reder to specific tutorials to better exploit the benefits of this library. 

The recommended installation procedure is to use the `install_github` command from the devtools R package:

```r
devtools::install_github("SantanderMetGroup/downscaleR.keras")
```

The functions within this package contain illustrative examples, however, we refer the reader to the jupyter notebooks developed for the paper "Configuration and intercomparison of deep learning neural models for statistical downscaling" submitted to the Geoscientific Model and Development journal on September 2019. These can be found in the [github deep notebooks repository of the Santander Meteorology Group](https://github.com/SantanderMetGroup/DeepDownscaling).

**NOTE:** Note that other climate4R libraries (e.g., `transformeR`) as well as `keras (v >= 2.2)` and `tensorflow (v >= 2.0)` libraries are dependencies of this package.
