package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}

class LSTM(val nbOutput: Int,
          val nbInput: Int,
          val activation: ActivationFn = Identity,
          override val name: String)
  extends LayerConfig {

}

object LSTM {
  def apply(nbInput: Int,
            nbOutput: Int,
            activation: ActivationFn = Identity,
            name: String = ""): LSTM = {
    new LSTM(nbOutput, nbInput, activation = activation, name)
  }
}
