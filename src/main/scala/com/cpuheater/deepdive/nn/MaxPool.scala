package com.cpuheater.deepdive.nn

case class MaxPool(poolHeight:Int,
                   poolWidth: Int,
                   stride: Int,
                   override val name: String)
  extends LayerConfig
