package com.cpuheater.deepdive.nn

case class MaxPool(height:Int,
                   width: Int,
                   poolHeight:Int,
                   poolWidth: Int,
                   stride: Int,
                   override val name: String)
  extends LayerConfig {


  val outHeight = 1 + (height - poolHeight) / stride
  val outWidth = 1 + (width - poolWidth) / stride

}
