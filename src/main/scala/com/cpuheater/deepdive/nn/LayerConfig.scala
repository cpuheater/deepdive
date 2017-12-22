package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.ActivationFn

private trait LayerConfig {

  def name: String

  def nbOutput: Int

  def nbInput: Int

  def activation: ActivationFn

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"


}
