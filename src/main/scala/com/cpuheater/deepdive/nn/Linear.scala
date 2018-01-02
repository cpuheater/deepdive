package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.ActivationFn
import com.cpuheater.deepdive.activations.{ActivationFn, Identity}
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.LayerConfig


class Linear(val nbOutput: Int,
             val nbInput: Int,
             override val activation: ActivationFn = Identity,
             override val name: String)
  extends LayerConfig {

}

object Linear {
  def apply(nbInput: Int,
             nbOutput: Int = 0,
            activation: ActivationFn = Identity,
            name: String = ""): Linear = {
    new Linear(nbOutput, nbInput, activation = activation, name)
  }
}
