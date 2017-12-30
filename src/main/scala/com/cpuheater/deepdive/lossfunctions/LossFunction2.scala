package com.cpuheater.deepdive.lossfunctions

import com.cpuheater.deepdive.activations.ActivationFn
import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction2 {

  def computeGradientAndScore(label: INDArray, output: INDArray, activationFn: ActivationFn) : (Double, INDArray)


}
