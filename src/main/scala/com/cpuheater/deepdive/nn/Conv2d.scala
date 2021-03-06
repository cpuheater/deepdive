package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}


case class Conv2d(height:Int,
                  width: Int,
                  channels: Int,
                  numFilters: Int,
                  filterHeight: Int,
                  filterWidth: Int,
                  stride: Int,
                  padding: Int,
                  val activation: ActivationFn = Identity,
                  override val name: String)
  extends LayerConfig {

  val outHeight = (height + 2 * padding - filterHeight) / stride +1
  val outWidth = (width + 2*padding - filterHeight) / stride + 1


}