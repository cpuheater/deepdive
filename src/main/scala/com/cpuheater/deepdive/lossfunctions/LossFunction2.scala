package com.cpuheater.deepdive.lossfunctions

import org.nd4j.linalg.api.ndarray.INDArray

trait LossFunction2 {

  def computeGradientAndScore(label: INDArray, output: INDArray) : (Double, INDArray)


}
