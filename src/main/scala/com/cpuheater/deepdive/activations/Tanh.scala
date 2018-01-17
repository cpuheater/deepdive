package com.cpuheater.deepdive.activations


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.tanh
import org.nd4s.Implicits._


object Tanh extends ActivationFn {

  def apply(a: INDArray): INDArray = tanh(a)

  def derivative(x: INDArray): INDArray = {
    (tanh(x)*tanh(x)).rsub(1)
  }
}
