package com.cpuheater.deepdive.lossfunctions

import com.cpuheater.deepdive.activations.ActivationFn
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction {

  def computeScoreAndGradient(label: INDArray, output: INDArray) : Float


  def computeGradient(label: INDArray, output: INDArray, activationFn: ActivationFn): INDArray

}