package com.cpuheater.deepdive.layers

import com.cpuheater.deepdive.core.Activation

trait Layer {

  def name: String

  def nbOutput: Int

  def nbInput: Int

  def activationFn: Activation

  override def toString(): String = s"number of input = ${nbInput} number of output = ${nbOutput}"

}


