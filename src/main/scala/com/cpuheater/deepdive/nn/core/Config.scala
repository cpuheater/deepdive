package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.lossfunctions.LossFunction2
import com.cpuheater.deepdive.nn.LayerConfig

case class Config(layers: List[LayerConfig],
                  loss: LossFunction2,
                  lr: Double,
                  batchSize: Int,
                  numOfEpoch:Int = 2)
