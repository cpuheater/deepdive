package com.cpuheater.deepdive.nn.core

import com.cpuheater.deepdive.activations.ReLU
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import shapeless.HList
import shapeless.ops.hlist.ToList
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._

class MultiLayerNetwork(hidden: List[Int], input: Int, numClasses: Int) {

  val numLayers = 1+hidden.length
  var params: Map[String, INDArray] =
    ((input :: hidden) :+ numClasses).sliding(2).zipWithIndex.foldLeft(Map.empty[String, INDArray]){
      case (accum, (List(first, second), index)) =>
        val weight = (s"W${index+1}"-> Nd4j.zeros(Array(first, second): _*) )
        val bias = (s"b${index+1}"->Nd4j.zeros(second))
        accum + weight + bias
    }

  val tmp = scala.collection.mutable.Map[String, INDArray]()
  tmp(s"W${1}") = Nd4j.create(Array(0.09700684, -0.01866912, -0.16777732, 0.13514824, 0.04932248, -0.13881888, 0.09700684, -0.01866912, -0.16777732)).reshape(3, 3)
  tmp(s"W${2}") = Nd4j.create(Array(-0.07079329, 0.11830491, .13195695, 0.08975121,  0.13044141,  0.05507294, 0.08975121,  0.13044141,  0.05507294f)).reshape(3, 3)
  tmp(s"W${3}") = Nd4j.create(Array( 0.03155537,  0.16341055, -0.08454048, 0.10095577,
  0.00948496,  0.13195695, 0.08975121,  0.13044141,  0.05507294, 0.08975121,  0.13044141,  0.05507294, 0.09700684, -0.01866912, -0.16777732,  0.13514824,
  0.04932248, -0.13881888,
  -0.21343547,  0.08187358,  0.0972448 , -0.07079329,
  0.01285232,  0.11055773,0.11830491,  0.01397892, -0.03056505, 0.02692668,0.02095663,  0.04239022)).reshape(3, 10)
  tmp(s"b${1}") = Nd4j.zeros(3)
  tmp(s"b${2}") = Nd4j.zeros(3)
  tmp(s"b${3}") = Nd4j.zeros(10)
  params = tmp.toMap


  def loss(x: INDArray, y: INDArray)= {

    val cache = scala.collection.mutable.Map[Int, (INDArray, INDArray, INDArray, INDArray)]()
    val hScore = (1 to hidden.length).foldLeft(x){
      case (score, index) =>
        val w = params(s"W${index}")
        val b = params(s"b${index}")
        val (newScore, ac, cx, cw, cb) = reluForward(score, w, b)
        cache(index) = (ac, cx, cw, cb)
        newScore
    }
    val w = params(s"W${hidden.length+1}")
    val b = params(s"b${hidden.length+1}")
    val (preOutput, cx, cw, cb)   = forward(hScore, w, b)

    val (loss, dout) = SoftMaxLoss.computeGradientAndScore(preOutput, y)

    val (dx, dw, db) = backward(dout, cx, cw, cb)

    val grads = scala.collection.mutable.Map[String, INDArray]()
    grads(s"W${hidden.length +1}") = dw
    grads(s"b${hidden.length +1}") = db


    (hidden.length to 1 by -1).foldLeft(dx){
      case (dout, index) =>
        val (ca, cx, cw, cb) = cache(index)
        val (dx, dw, db) = reluBackward(dout, ca, cx, cw, cb)
        grads(s"W${index}") = dw
        grads(s"b${index}") = db
        dx
    }

    (loss, grads)

  }


  def reluBackward(dout: INDArray, ca: INDArray, cx: INDArray, cw: INDArray, cb: INDArray) = {
    val caDupl = ca.dup()
    BooleanIndexing.applyWhere(caDupl, Conditions.lessThan(0), 0)
    val tmp = ReLU.derivative(ca)

    val da = caDupl * dout
    val (dx, dw,db) = backward(da, cx, cw, cb)
    (dx, dw,db)
  }

  def backward(dout: INDArray, cx: INDArray, cw: INDArray, wb: INDArray): (INDArray, INDArray, INDArray) = {
    val dx = dout.dot(cw.T).reshape(cx.shape(): _*)
    val dw = cx.reshape(cx.rows(), -1).T.dot(dout)
    val db = dout.sum(0)
    (dx, dw, db)
  }


  def forward(x: INDArray, w: INDArray, b: INDArray): (INDArray, INDArray, INDArray, INDArray) = {
    val d = x.reshape(x.shape()(0), -1).dot(w)
    val out = x.reshape(x.shape()(0), -1).dot(w).addRowVector(b)
    (out, x, w, b)
  }

  def reluForward(x: INDArray, w: INDArray, b: INDArray):
  (INDArray, INDArray, INDArray, INDArray, INDArray) = {
    val (out1, cx, cw, cb) = forward(x, w, b)
    val out2  = ReLU(out1)
    (out2, out1, cx, cw, cb)
  }


}
