package com.cpuheater.deepdive.nn

package object layers {

  trait CompType

  object CompType {
    case object W extends CompType
    case object B extends CompType
    case object PreOutput extends CompType
    case object X extends CompType
  }

}
