package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.core.Activation

class Dense(override val nbOutput: Int,
            override val nbInput: Int,
            override val activationFn: Activation,
            override val name: String)
  extends Layer {


}

object Dense {
  def apply(nbOutput: Int,
            nbInput: Int = 0,
            activation: Activation = Activation.Identity,
            name: String = ""): Dense = {
    new Dense(nbOutput, nbInput, activationFn = activation, name)
  }
}
