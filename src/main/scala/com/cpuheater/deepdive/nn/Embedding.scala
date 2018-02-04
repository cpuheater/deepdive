package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.ActivationFn
import com.cpuheater.deepdive.activations.{ActivationFn, Identity}
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.LayerConfig


class Embedding(val nbOutput: Int,
                val nbInput: Int,
                override val name: String)
  extends LayerConfig {

}

object Embedding {
  def apply(nbInput: Int,
            nbOutput: Int = 0,
            name: String = ""): Embedding = {
    new Embedding(nbOutput, nbInput, name)
  }
}

