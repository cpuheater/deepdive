package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}
import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.LayerConfig
import com.cpuheater.deepdive.nn.core.Activation

class Dense(override val nbOutput: Int,
            override val nbInput: Int,
            override val activation: ActivationFn,
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
