package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.LayerConfig

class Dense(val nbOutput: Int,
            val nbInput: Int,
            val activation: ActivationFn,
            override val name: String)
  extends LayerConfig {


}

object Dense {
  def apply(nbOutput: Int,
            nbInput: Int = 0,
            activation: ActivationFn = Identity,
            name: String = ""): Dense = {
    new Dense(nbOutput, nbInput, activation = activation, name)
  }
}
