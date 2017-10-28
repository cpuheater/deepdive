package com.cpuheater.deepdive.layers

import com.cpuheater.deepdive.core.Activation
import com.cpuheater.deepdive.lossfunctions.LossFunction

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
