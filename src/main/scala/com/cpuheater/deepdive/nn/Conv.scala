package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{ActivationFn, Identity}


case class Conv(height:Int,
                width: Int,
                channels: Int,
                nbOfFilters: Int,
                filterHeight: Int,
                filterWidth: Int,
                stride: Int,
                padding: Int,
                override val activation: ActivationFn = Identity,
                override val name: String)
  extends LayerConfig {


  /**
    * out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    */

  val outHeight = (height + 2 * padding - filterHeight) / stride +1
  val outWidth = (width + 2*padding - filterHeight) / stride + 1


}