package com.cpuheater.deepdive.activations


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Identity extends ActivationFn {

  def apply(a: INDArray): INDArray = a

  def derivative(x: INDArray): INDArray = {
    Nd4j.onesLike(x)
  }
}
