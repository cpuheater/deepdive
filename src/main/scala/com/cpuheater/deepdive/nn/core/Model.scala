package com.cpuheater.deepdive.nn.core

import org.nd4j.linalg.api.ndarray.INDArray

trait Model {

  def config : ModelConfig

  def params(): Map[String, INDArray]

}
