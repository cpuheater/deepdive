package com.cpuheater.deepdive.activations


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid
import org.nd4s.Implicits._


object Sigmoid extends ActivationFn {

  def apply(a: INDArray): INDArray = sigmoid(a)

  def derivative(x: INDArray): INDArray = {
    sigmoid(x)*sigmoid(x).rsub(1)
  }
}
