package com.cpuheater.deepdive.activations


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object ReLU extends ActivationFn {
  def apply(a: INDArray): INDArray = Transforms.relu(a)

  def derivative(x: INDArray): INDArray =
    Nd4j.getExecutioner.execAndReturn(new RectifedLinear(x).derivative)
}
