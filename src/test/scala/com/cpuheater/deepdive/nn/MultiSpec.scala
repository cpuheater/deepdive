package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.nn.core.{OldMultiLayerNetwork, OldSolver}
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._



class MultiSpec extends TestSupport {

  it should "cross entropy 0.9838" in {

    val input = 3*32*32
    val hidden1 = 100
    val hidden2 = 100
    val output = 10

    val nbOfEpoch = 30
    val (features1, labels1) = readFromFile
    //val nbOfExamples = features.rows()


    val features =  Nd4j.create(Array(81.46373469,   76.43038776,   74.09187755, -104.66336735, -112.65646939,  -80.41834694)).reshape(2, 3)
    val labels =  Nd4j.create(Array(Array(0,0,0f,0,1,0,0,0,0,0), Array(0,0,0,0f,0,0,0,0,0,1)))


    val dataSet = new DataSet(features, labels)
    val lr = 1e-3
    val batchSize = 2

    val network = new OldMultiLayerNetwork(List(3, 3), input = 3*3, numClasses = 10)
    val solver = new OldSolver(network)
    solver.train(dataSet, lr, batchSize)


    /*val correct = (0 until nbOfExamples).foldLeft(0){
      case (accum, index) =>
        val output = network.predict(features.getRow(index).T)
        val pred: Int = Nd4j.argMax(output).getInt(0) +1
        val target = Nd4j.argMax(labels.getRow(index)).getInt(0) +1
        println(s"pred ${pred}, index ${target}")
        if(pred == target)
          accum +1
        else
          accum
    }

    println(s"ulalala set accuracy: ${correct.toDouble/nbOfExamples} %")*/
  }


  def readFromFile: (INDArray, INDArray) = {

    val oneHotMap = Nd4j.eye(10)
    val lines = io.Source.fromInputStream(getClass.getResourceAsStream("/cifar4.csv")).getLines()

    val (features, labelsArray) = lines.map { x =>
      val value = x.split(",")
      val features = value.reverse.tail
      val label = value.takeRight(1)
      (features.map(_.toFloat), label.map(_.toFloat))
    }.toArray.foldLeft((Array.empty[Array[Float]], Array.empty[Array[Float]])){
      case ((features, labels), (f, l)) =>
        (features :+ f, labels :+ l)
    }

    val labels = Nd4j.create(features.length, 10)
    val labelsNDarray = labelsArray.flatten.zipWithIndex.foreach{ case (value, index) => labels(index, ->) =   oneHotMap.getRow(value.toInt)}
    (Nd4j.create(features), labels)
  }


}
