package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.ActivationFn

case class Linear(override val nbInput: Int,
                  override val nbOutput: Int,
                  override val activation: ActivationFn,
                  override val name: String = "")
  extends LayerConfig
