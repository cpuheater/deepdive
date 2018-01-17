package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}

class RNN(val nbOutput: Int,
             val nbInput: Int,
             val activation: ActivationFn = Identity,
             override val name: String)
  extends LayerConfig {

}

object RNN {
  def apply(nbInput: Int,
            nbOutput: Int,
            activation: ActivationFn = Identity,
            name: String = ""): RNN = {
    new RNN(nbOutput, nbInput, activation = activation, name)
  }
}
