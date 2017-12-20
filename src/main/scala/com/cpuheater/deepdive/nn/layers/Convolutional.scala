package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


class Convolutional(hidden: List[Int], input: Int, numClasses: Int) {

  val numLayers = 1+hidden.length
  var params: Map[String, INDArray] =
    ((input :: hidden) :+ numClasses).sliding(2).zipWithIndex.foldLeft(Map.empty[String, INDArray]){
      case (accum, (List(first, second), index)) =>
        val weight = (s"W${index+1}"-> Nd4j.zeros(Array(first, second): _*) )
        val bias = (s"b${index+1}"->Nd4j.zeros(second))
        accum + weight + bias
    }

  val tmp = scala.collection.mutable.Map[String, INDArray]()


  val w1 = Array(Array( 0.53770564, -0.12171369, -1.04250949 , 0.13828792, -0.22557183, -1.1928829,
  -0.68320696 , 0.19909408 , 0.03070661),
    Array(-0.44972639,  0.14447532, -0.35229594,  0.4882136 , -0.4347099 , -0.28692265,
  -0.84338097 , -0.10827394,  0.85434757),
  Array(-0.90377338, -1.0525584 , -0.30409794,  0.18083726, -0.4125417 ,  1.22913948,
  -0.97791748 , -0.63978524, -0.00880963),
  Array( 0.36213294,  0.35148162, -1.20064035 -0.84272962,  1.61832501, -2.39079478,
  0.88256212  , -1.12082008,  0.12416778),
  Array(-2.43434598, -1.62701704, -1.10613945,  2.00862337,  0.91447992, -0.86943856,
  0.13741017  , -0.72261465, -0.6755518 ),
  Array(-0.04084145, -1.15755894, -0.67498124,  0.44252644, -0.11970425,  0.59700586,
  -2.0399063  , -0.6447033 , -1.61700346),
  Array(-1.85570785,  0.7480524 ,  1.97009461 -0.81967191,  0.42367521, -1.83570698,
  1.00583627  , 0.05834497 , 0.98687019),
  Array( 0.27827513,  0.119619  ,  2.13935149, -0.77429891, -0.42011785, -0.77768316,
  -1.0141586  , -2.30296116, -0.3678455 ),
  Array( 0.59190367, -0.56016357f,  1.03339676, -1.20718575, -0.93426589,  0.92234837,
  0.37364743  , -0.26051388,  0.61250765))



  tmp(s"W${1}") = Nd4j.create(w1).reshape(Array(3,3,3,3):_*)

  tmp(s"b${1}") = Nd4j.create(Array( 0.41794341,  1.39710028, -1.78590431))    
  params = tmp.toMap


  def loss(x: INDArray, y: INDArray)= {

    val cache = scala.collection.mutable.Map[Int, (INDArray, INDArray, INDArray, INDArray)]()


  }



}
