package com.cpuheater.deepdive.activations

import org.nd4j.linalg.api.ndarray.INDArray


trait ActivationFn extends (INDArray => INDArray) {

  def apply(arg: INDArray): INDArray

  def derivative(arg: INDArray): INDArray

}
