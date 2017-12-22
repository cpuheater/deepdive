package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.lossfunctions.{CrossEntropyLoss, MSELoss}
import com.cpuheater.deepdive.nn.core.Activation
import com.cpuheater.deepdive.nn.layers.Dense
import com.cpuheater.deepdive.nn.models.Sequential
import com.cpuheater.deepdive.util.TestSupport
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._


class NNSpec extends TestSupport {

/*
  it should "cross entropy 0.9838" in {

    val input = 400
    val hidden = 25
    val output = 10

    val nbOfEpoch = 30

    val learningRate = 0.5
    val batchSize = 10

    val (features, labels) = readFromFile
    val nbOfExamples = features.rows()

    val network = Sequential()

    network.add(Dense(nbOutput = 25, nbInput = 400, activation = Activation.Sigmoid))
    network.add(Dense(nbInput = 25, nbOutput = 10, activation = Activation.Sigmoid))


    val dataSet = new DataSet(features, labels)

    network.compile(CrossEntropyLoss)
    network.fit(dataSet, nbOfEpoch, batchSize, learningRate)

    val correct = (0 until nbOfExamples).foldLeft(0){
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

    println(s"ulalala set accuracy: ${correct.toDouble/nbOfExamples} %")
  }



  it should "accuracy 0.9838" in {

    val input = 400
    val hidden = 25
    val output = 10

    val nbOfEpoch = 30

    val learningRate = 3
    val batchSize = 10

    val (features, labels) = readFromFile

    val nbOfExamples = features.rows()

    val network = Sequential()

    network.add(Dense(nbOutput = 25, nbInput = 400, activation = Activation.Sigmoid))
    network.add(Dense(nbInput = 25, nbOutput = 10, activation = Activation.Sigmoid))


    val dataSet = new DataSet(features, labels)

    network.compile(CrossEntropyLoss)
    network.fit(dataSet, nbOfEpoch, batchSize, learningRate)

    val correct = (0 until nbOfExamples).foldLeft(0){
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

    println(s"ulalala set accuracy: ${correct.toDouble/nbOfExamples} %")
  }



  it should "predict2" in {


    val input = 400
    val hidden = 25
    val output = 10

    val nbOfEpoch = 30

    val learningRate = 3
    val batchSize = 10


    val numLinesToSkip = 0
    val delimiter = ' '

    val labelIndex = 400
    val numClasses = 10

    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("images.txt").getFile))
    val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)

    val network = Sequential()

    network.add(Dense(nbOutput = 25, nbInput = 400, activation = Activation.Sigmoid))
    network.add(Dense(nbInput = 25, nbOutput = 10, activation = Activation.Sigmoid))
    network.compile(MSELoss)


    (0 until nbOfEpoch).map{
      index =>
        println(s"Epoch ${index}")
        network.fit(iterator, learningRate)
    }


    var count = 0
    var total = 0
    while (iterator.hasNext) {
      val next = iterator.next()
      val feature = next.getFeatures
      val labels = next.getLabels
      (0 until feature.rows()).foreach {
        index =>
          val output = network.predict(feature.getRow(index).T)
          val prediction: Int = Nd4j.argMax(output).getInt(0)
          val label = Nd4j.argMax(labels.getRow(index)).getInt(0)
          if (prediction == label)
            count = count + 1
          total = total +1
      }
    }

    println(s"Training set accuracy: ${count.toFloat/total} %")
  }



  def readFromFile: (INDArray, INDArray) = {

    val oneHotMap = Nd4j.eye(10)
    val lines = io.Source.fromInputStream(getClass.getResourceAsStream("/images.txt")).getLines()

    val (features, labelsArray) = lines.map { x =>
      val value = x.split(" ")
      val features = value.reverse.tail
      val label = value.takeRight(1)
      (features.map(_.toFloat), label.map(_.toFloat))
    }.toArray.foldLeft((Array.empty[Array[Float]], Array.empty[Array[Float]])){
      case ((features, labels), (f, l)) =>
        (features :+ f, labels :+ l)
    }

    val labels = Nd4j.create(features.length, 10)
    val labelsNDarray = labelsArray.flatten.zipWithIndex.foreach{ case (value, index) => labels(index, ->) =   oneHotMap.getRow(value.toInt -1)}
    (Nd4j.create(features), labels)
  }*/

}
