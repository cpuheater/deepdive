package com.cpuheater.deepdive.optimize

import org.nd4j.linalg.api.ndarray.INDArray

trait BaseOptimizer {

  def optimize(param: INDArray, grad: INDArray, key: String): INDArray

}
