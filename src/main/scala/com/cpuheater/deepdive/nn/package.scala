package com.cpuheater.deepdive

package object nn {

  trait WeightsInitType

  object WeightsInitType {
    case object XAVIER extends WeightsInitType
    case object UNIFORM extends WeightsInitType

  }

}
