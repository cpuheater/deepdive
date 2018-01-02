package com.cpuheater.deepdive.nn

package object layers {

  trait ParamType

  object ParamType {
    case object W extends ParamType
    case object B extends ParamType
    case object PreOutput extends ParamType
    case object X extends ParamType
    case object X2Cols extends ParamType

    def toString(paramType: ParamType, index:Int) = s"${paramType}$index"
  }

}
