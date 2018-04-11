package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.lossfunctions.LossFunction2
import com.cpuheater.deepdive.nn.{LayerConfig, Optimizer}

case class ModelConfig(layers: List[LayerConfig],
                       loss: LossFunction2,
                       optimizer: Optimizer)
