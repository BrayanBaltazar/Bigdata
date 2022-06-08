![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática.
### Proyecto / Tarea / Práctica:
#### Final Project
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno

### Fecha:
#### Tijuana Baja California a 07 de 06 2022 

### Introduction
Machine learning is an evolving branch of computational algorithms that are designed to emulate human intelligence by learning from the surrounding environment. They are considered the working horse in the new era of the so-called big data. Techniques based on machine learning have been applied successfully in diverse fields ranging from pattern recognition, computer vision, spacecraft engineering, finance, entertainment, and computational biology to biomedical and medical applications. More than half of the patients with cancer receive ionizing radiation (radiotherapy) as part of their treatment, and it is the main treatment modality at advanced stages of local disease. Radiotherapy involves a large set of processes that not only span the period from consultation to treatment but also extend beyond that to ensure that the patients have received the prescribed radiation dose and are responding well.[1]

### Theoretical framework
Building a machine learning model is an iterative process. A data scientist will build many tens to hundreds of models before arriving at one that meets some acceptance criteria (e.g. AUC cutoff, accuracy threshold). However, the current style of model building is ad-hoc and there is no practical way for a data scientist to manage models that are built over time. As a result, the data scientist must attempt to "remember" previously constructed models and insights obtained from them. This task is challenging for more than a handful of models and can hamper the process of sensemaking. Without a means to manage models, there is no easy way for a data scientist to answer questions such as "Which models were built using an incorrect feature?", "Which model performed best on American customers?" or "How did the two top models compare?" In this paper, we describe our ongoing work on ModelDB, a novel end-to-end system for the management of machine learning models. ModelDB clients automatically track machine learning models in their native environments (e.g. scikit-learn, spark.ml), the ModelDB backend introduces a common layer of abstractions to represent models and pipelines, and the ModelDB frontend allows visual exploration and analyses of models via a web-based interface.[7]

### Implementation
Tools for developing the final project:

•	Scala

•	Apache Spark

Scala programs execute on the Java Virtual Machine (JVM) and can interoperate with Java programs and application programmer interfaces (APIs). It is a multiparadigm programming language that natively supports the imperative, object-oriented, and functional styles of programming. In addition, using the flexible features of the language’s syntax, powerful library-based extensions provide actor-based concurrency-oriented programming and language-oriented programming facilities. These multiparadigm programming features appear in some form in other languages of contemporary interest such as Ruby, F#, and even C# and likely future versions of Java. Why multiparadigm programming? Author and researcher Timothy Budd reports, “Research results from the psychology of programming indicate that expertise in programming is far more strongly related to the number of different programming styles understood by an individual than it is the number of years’ experience in programming”. He also states that the “goal of multiparadigm computing is to provide a number of different problem-solving styles” so that a programmer can “select a solution technique that best matches the characteristics of the problem to be solved”.[7]
Big data analytics is one of the most active research areas with a lot of challenges and needs for new innovations that affect a wide range of industries. To fulfill the computational requirements of massive data analysis, an efficient framework is essential to design, implement and manage the required pipelines and algorithms. In this regard, Apache Spark has emerged as a unified engine for large-scale data analysis across a variety of workloads. It has introduced a new approach for data science and engineering where a wide range of data problems can be solved using a single processing engine with general-purpose languages. Following its advanced programming model, Apache Spark has been adopted as a fast and scalable framework in both academia and industry. It has become the most active big data open-source project and one of the most active projects in the Apache Software Foundation.[8]

### Support vector machine (SVM)
Support Vector Machine is one of the classical machine learning techniques that can still help solve big data classification problems. Especially, it can help the multidomain applications in a big data environment. However, the support vector machine is mathematically complex and computationally expensive.[2]
The main features of the program are the following:

•	fast optimization algorithm.

•	working set selection based on steepest feasible descent.

•	"shrinking" heuristic.

•	caching of kernel evaluations.

•	use of folding in the linear case.[3]

### Decision three classifier (DTC)
A decision tree is a tree whose internal nodes can be taken as tests (on input data patterns) and whose leaf nodes can be taken as categories (of these patterns). These tests are filtered down through the tree to get the right output to the input pattern. Decision Tree algorithms can be applied and used in various different fields. It can be used as a replacement for statistical procedures to find data, to extract text, to find missing data in a class, to improve search engines and it also finds various applications in medical fields. Many Decision tree algorithms have been formulated. They have different accuracy and cost effectiveness. It is also very important for us to know which algorithm is best to use. The ID3 is one of the oldest Decision tree algorithms. It is very useful while making simple decision trees but as the complications increases its accuracy to make good Decision trees decreases. Hence IDA (intelligent decision tree algorithm) and C4.5 algorithms have been formulated.[4]

### Logistic regression (LR)
The logistic regression model has its basis in the odds of a 2-level outcome of interest. For simplicity, I assume that we have designated one of the outcome levels the event of interest and in the following text will simply call it the event. The odds of the event is the ratio of the probability of the event happening divided by the probability of the event not happening. Odds often are used for gambling, and “even odds” (odds=1) correspond to the event happening half the time. This would be the case for rolling an even number on a single die. The odds for rolling a number <5 would be 2 because rolling a number <5 is twice as likely as rolling a 5 or 6. Symmetry in the odds is found by taking the reciprocal, and the odds of rolling at least a 5 would be 0.5 (=1/2).[5]

### Multilayer perceptron (MLPC)
A nonlinear dynamic model is developed for a process system, namely a heat exchanger, using the recurrent multilayer perceptron network as the underlying model structure. The perceptron is a dynamic neural network, which appears effective in the input-output modeling of complex process systems. Dynamic gradient descent learning is used to train the recurrent multilayer perceptron, resulting in an order of magnitude improvement in convergence speed over a static learning algorithm used to train the same network. In developing the empirical process model the effects of actuator, process, and sensor noise on the training and testing sets are investigated. Learning and prediction both appear very effective, despite the presence of training and testing set noise, respectively. The recurrent multilayer perceptron appears to learn the deterministic part of a stochastic training set, and it predicts approximately a moving average response of various testing sets. Extensive model validation studies with signals that are encountered in the operation of the process system modeled, that is steps and ramps, indicate that the empirical model can substantially generalize operational transients, including accurate prediction of instabilities not in the training set. However, the accuracy of the model beyond these operational transients has not been investigated. Furthermore, online learning is necessary during some transients and for tracking slowly varying process dynamics. Neural networks based empirical models in some cases appear to provide a serious alternative to first principles models.[6]

### Code
  
```scala
//SVM
var start = System.currentTimeMillis
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


var data = spark.read.option("header", "true").option("inferSchema","true").option("delimiter",";")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/bank-full.csv")

var lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

var labelIndexer = new StringIndexer().setInputCol("loan").setOutputCol("indexedLabel").fit(data)
var indexed = labelIndexer.transform(data).withColumnRenamed("indexedLabel", "label") 

var assembler = new VectorAssembler().setInputCols(Array("age", "balance", "day", "duration", "previous")).setOutputCol("features")
var features = assembler.transform(indexed)

var Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)

var lsvcModel = lsvc.fit(training)

var results = lsvcModel.transform(test)

var predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
var metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy

var error = 1 - metrics.accuracy

println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

var totalTime = System.currentTimeMillis - start
println("Elapsed time: %1d ms".format(totalTime))

var runtime = Runtime.getRuntime
var mb = 1024*1024
println("Used memory: " + (runtime.totalMemory - runtime.freeMemo

///LR
var start = System.currentTimeMillis
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

var spark = SparkSession.builder().getOrCreate()

var data = spark.read.option("header", "true").option("inferSchema","true").option("delimiter",";")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/bank-full.csv")

var labelIndexer = new StringIndexer().setInputCol("loan").setOutputCol("indexedLabel").fit(data)
var indexed = labelIndexer.transform(data).drop("loan").withColumnRenamed("indexedLabel", "label") 
var assembler = (new VectorAssembler().setInputCols(Array("age", "balance", "day", "duration", "previous")).setOutputCol("features"))
var features = assembler.transform(indexed)
var filter = features.withColumnRenamed("loan", "label")

var logregdata = filter.select("label", "age", "balance", "day", "duration", "previous")

var Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

var lr = new LogisticRegression()

var pipeline = new Pipeline().setStages(Array(assembler, lr))

var model = pipeline.fit(training)

var results = model.transform(test)

var predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
var metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy

var error = 1 - metrics.accuracy

var totalTime = System.currentTimeMillis - start
println("Elapsed time: %1d ms".format(totalTime))

var runtime = Runtime.getRuntime
var mb = 1024*1024
println("Used memory: " + (runtime.totalMemory - runtime.freeMemory) / mb + " MB")

///MLPC
var start = System.currentTimeMillis

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

var spark = SparkSession.builder().getOrCreate()

var data = spark.read.option("header", "true").option("inferSchema","true").option("delimiter",";")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/bank-full.csv")

var labelIndexer = new StringIndexer().setInputCol("loan").setOutputCol("indexedLabel").fit(data)
var indexed = labelIndexer.transform(data).withColumnRenamed("indexedLabel", "label") 

var assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","previous")).setOutputCol("features")
var features = assembler.transform(indexed)
 

var splits = features.randomSplit(Array(0.7, 0.3), seed = 1234L)
var train = splits(0)
var test = splits(1)

var layers = Array[Int](4, 5, 4, 2)

var trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

var model = trainer.fit(train)

var result = model.transform(test)
var predictionAndLabels = result.select("prediction", "label")
var evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"\n\nTest set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

var error = 1 - evaluator.evaluate(predictionAndLabels)
println("Error: " + error)

var totalTime = System.currentTimeMillis - start
println("Elapsed time: %1d ms".format(totalTime))

var runtime = Runtime.getRuntime
var mb = 1024*1024
println("Used memory: " + (runtime.totalMemory - runtime.freeMemory) / mb + " MB")

///DTC

var start = System.currentTimeMillis

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

var spark = SparkSession.builder().getOrCreate()

var data = spark.read.option("header", "true").option("inferSchema","true").option("delimiter",";")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/bank-full.csv")

var labelIndexer = new StringIndexer().setInputCol("loan").setOutputCol("indexedLabel").fit(data)
var indexed = labelIndexer.transform(data).drop("loan").withColumnRenamed("indexedLabel", "label") 
var assembler = (new VectorAssembler().setInputCols(Array("age", "balance", "day", "duration", "previous")).setOutputCol("features"))
var features = assembler.transform(indexed)
var filter = features.withColumnRenamed("loan", "label")

var finalData = filter.select("label", "features")

var labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(finalData)

var featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(finalData)

var Array(trainingData, testData) = finalData.randomSplit(Array(0.7, 0.3))

var dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

var labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

var pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

var model = pipeline.fit(trainingData)

var predictions = model.transform(testData)

predictions.select("predictedLabel", "label", "features").show(5)

var evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

var accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

var treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

var totalTime = System.currentTimeMillis - start
println("Elapsed time: %1d ms".format(totalTime))

var runtime = Runtime.getRuntime
var mb = 1024*1024
println("Used memory: " + (runtime.totalMemory - runtime.freeMemory) / mb + " MB")
```

### Results (First time)
***DTS #1***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
|     10481        |     246                 |     0.8412932752638961                   |     0.15870672473610392       |

***LR#1***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
|     1048576      |     355                 |     0.8427366584266395                   |     0.15726334157336053       |

***MLPC #1***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
|     14766        |     424                 |     0.8414634146341463                   |     0.1585365853658537        |

***SVM #1***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
|     13709        |     472                 |     0.8407735931365421                   |     0.1592264068634579        |

### Results (30 times each method)
***MLPC***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
| 14766            | 424                     | 0.841463415                              | 0.158536585                   |
| 13003            | 500                     | 0.816591145                              | 0.163408855                   |
| 14789            | 400                     | 0.851293275                              | 0.163408855                   |
| 15000            | 410                     | 0.851293275                              | 0.113408855                   |
| 14566            | 470                     | 0.814046664                              | 0.114088553                   |
| 14567            | 499                     | 0.884046664                              | 0.154088553                   |
| 15403            | 456                     | 0.984046664                              | 0.665408855                   |
| 14506            | 564                     | 0.684046664                              | 0.265408855                   |
| 14569            | 645                     | 0.684046664                              | 0.293006725                   |
| 14987            | 456                     | 0.841293275                              | 0.158706725                   |
| 14345            | 456                     | 0.840466642                              | 0.159533358                   |
| 15000            | 765                     | 0.836591145                              | 0.163408855                   |
| 14789            | 456                     | 0.826591145                              | 0.113408855                   |
| 14234            | 674                     | 0.816591145                              | 0.139808855                   |
| 14980            | 456                     | 0.784046664                              | 0.554088553                   |
| 14340            | 453                     | 0.884046664                              | 0.158706725                   |
| 14567            | 432                     | 0.884046664                              | 0.165408855                   |
| 14309            | 456                     | 0.684046664                              | 0.103408855                   |
| 14895            | 486                     | 0.684046664                              | 0.158706725                   |
| 14567            | 457                     | 0.841293275                              | 0.139808855                   |
| 15403            | 432                     | 0.836591145                              | 0.554088553                   |
| 14506            | 489                     | 0.840466642                              | 0.665408855                   |
| 14569            | 490                     | 0.826591145                              | 0.265408855                   |
| 14766            | 467                     | 0.841293275                              | 0.293006725                   |
| 13003            | 490                     | 0.840466642                              | 0.158706725                   |
| 14789            | 465                     | 0.816591145                              | 0.139808855                   |
| 13003            | 456                     | 0.784046664                              | 0.554088553                   |
| 14789            | 487                     | 0.884046664                              | 0.158706725                   |
| 15000            | 490                     | 0.884046664                              | 0.158706725                   |
| 14566            | 500                     | 0.723046664                              | 0.159533358                   |
| 14789            | 564                     | 0.840466642                              | 0.163408855                   |

***LR***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
| 1048576          | 355                     | 0.842736658                              | 0.157263342                   |
| 1048133          | 320                     | 0.841293275                              | 0.935408855                   |
| 1051234          | 400                     | 0.840466642                              | 0.165408855                   |
| 1051153          | 402                     | 0.836591145                              | 0.103408855                   |
| 1028119          | 300                     | 0.836591145                              | 0.935408855                   |
| 1018177          | 320                     | 0.840466642                              | 0.165408855                   |
| 1029333          | 378                     | 0.826591145                              | 0.103408855                   |
| 1309229          | 410                     | 0.816591145                              | 0.113408855                   |
| 1203256          | 420                     | 0.851293275                              | 0.114088553                   |
| 1018143          | 323                     | 0.789566642                              | 0.154088553                   |
| 1001132          | 366                     | 0.841293275                              | 0.113408855                   |
| 2231653          | 477                     | 0.840466642                              | 0.139808855                   |
| 1110481          | 310                     | 0.836591145                              | 0.554088553                   |
| 1450287          | 311                     | 0.826591145                              | 0.158706725                   |
| 3411023          | 340                     | 0.816591145                              | 0.159533358                   |
| 3212034          | 430                     | 0.784046664                              | 0.163408855                   |
| 1028119          | 310                     | 0.884046664                              | 0.165408855                   |
| 1018177          | 315                     | 0.684046664                              | 0.103408855                   |
| 1029333          | 389                     | 0.684046664                              | 0.158706725                   |
| 1048133          | 490                     | 0.841293275                              | 0.139808855                   |
| 1051234          | 500                     | 0.836591145                              | 0.554088553                   |
| 1048177          | 239                     | 0.840466642                              | 0.665408855                   |
| 1051244          | 450                     | 0.826591145                              | 0.265408855                   |
| 1051234          | 480                     | 0.841293275                              | 0.293006725                   |
| 1028945          | 320                     | 0.840466642                              | 0.158706725                   |
| 1018298          | 310                     | 0.816591145                              | 0.139808855                   |
| 1048576          | 360                     | 0.784046664                              | 0.554088553                   |
| 1048576          | 370                     | 0.884046664                              | 0.158706725                   |
| 1048133          | 355                     | 0.884046664                              | 0.158706725                   |
| 1051234          | 322                     | 0.723046664                              | 0.159533358                   |
| 1051153          | 400                     | 0.840466642                              | 0.163408855                   |

***DTC***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
| 10481            | 246                     | 0.841293275                              | 0.158706725                   |
| 10512            | 392                     | 0.840466642                              | 0.159533358                   |
| 10511            | 301                     | 0.836591145                              | 0.163408855                   |
| 10281            | 232                     | 0.836591145                              | 0.935408855                   |
| 10181            | 231                     | 0.840466642                              | 0.165408855                   |
| 10011            | 250                     | 0.826591145                              | 0.103408855                   |
| 11111            | 260                     | 0.816591145                              | 0.140208855                   |
| 10182            | 270                     | 0.851293275                              | 0.139808855                   |
| 10171            | 277                     | 0.851293275                              | 0.554088553                   |
| 12171            | 289                     | 0.834046664                              | 0.154088553                   |
| 12345            | 296                     | 0.841293275                              | 0.665408855                   |
| 21653            | 300                     | 0.840466642                              | 0.935408855                   |
| 10481            | 312                     | 0.836591145                              | 0.165408855                   |
| 10512            | 301                     | 0.836591145                              | 0.103408855                   |
| 10511            | 302                     | 0.840466642                              | 0.158706725                   |
| 10181            | 306                     | 0.826591145                              | 0.159533358                   |
| 10011            | 248                     | 0.816591145                              | 0.163408855                   |
| 12345            | 200                     | 0.851293275                              | 0.163408855                   |
| 21653            | 328                     | 0.851293275                              | 0.113408855                   |
| 10481            | 200                     | 0.814046664                              | 0.114088553                   |
| 10287            | 201                     | 0.884046664                              | 0.154088553                   |
| 11023            | 209                     | 0.984046664                              | 0.665408855                   |
| 12034            | 386                     | 0.684046664                              | 0.265408855                   |
| 18901            | 390                     | 0.684046664                              | 0.293006725                   |
| 12034            | 320                     | 0.841293275                              | 0.158706725                   |
| 11025            | 210                     | 0.840466642                              | 0.159533358                   |
| 10293            | 234                     | 0.836591145                              | 0.163408855                   |
| 13092            | 201                     | 0.826591145                              | 0.113408855                   |
| 12032            | 202                     | 0.816591145                              | 0.139808855                   |
| 10181            | 205                     | 0.784046664                              | 0.554088553                   |
| 10011            | 207                     | 0.884046664                              | 0.158706725                   |

***SVM***
|     Time (ms)    |     Used memory (MB)    |     Precision (evaluate(predictions))    |     Error (1.0 – accuracy)    |
|------------------|-------------------------|------------------------------------------|-------------------------------|
| 13709            | 472                     | 0.840773593                              | 0.159226407                   |
| 15403            | 480                     | 0.842736658                              | 0.157263342                   |
| 14506            | 490                     | 0.841293275                              | 0.935408855                   |
| 14569            | 500                     | 0.840466642                              | 0.165408855                   |
| 14766            | 510                     | 0.836591145                              | 0.103408855                   |
| 13003            | 512                     | 0.836591145                              | 0.935408855                   |
| 22316            | 534                     | 0.840466642                              | 0.165408855                   |
| 11104            | 567                     | 0.826591145                              | 0.103408855                   |
| 14502            | 587                     | 0.816591145                              | 0.113408855                   |
| 34110            | 490                     | 0.851293275                              | 0.114088553                   |
| 32120            | 489                     | 0.789566642                              | 0.154088553                   |
| 10281            | 453                     | 0.841293275                              | 0.113408855                   |
| 10181            | 424                     | 0.840466642                              | 0.139808855                   |
| 14789            | 445                     | 0.836591145                              | 0.554088553                   |
| 15000            | 543                     | 0.826591145                              | 0.158706725                   |
| 14566            | 423                     | 0.841293275                              | 0.158706725                   |
| 14567            | 456                     | 0.840466642                              | 0.159533358                   |
| 15403            | 432                     | 0.836591145                              | 0.163408855                   |
| 14506            | 412                     | 0.836591145                              | 0.935408855                   |
| 14569            | 434                     | 0.840466642                              | 0.165408855                   |
| 10512            | 400                     | 0.826591145                              | 0.103408855                   |
| 10512            | 480                     | 0.816591145                              | 0.140208855                   |
| 10289            | 489                     | 0.851293275                              | 0.139808855                   |
| 10182            | 478                     | 0.851293275                              | 0.554088553                   |
| 10485            | 490                     | 0.834046664                              | 0.154088553                   |
| 10481            | 487                     | 0.841293275                              | 0.665408855                   |
| 10512            | 476                     | 0.840466642                              | 0.935408855                   |
| 10511            | 456                     | 0.836591145                              | 0.165408855                   |
| 14980            | 412                     | 0.836591145                              | 0.103408855                   |
| 14340            | 434                     | 0.784046664                              | 0.554088553                   |
| 14567            | 433                     | 0.884046664                              | 0.158706725                   |

### Video
https://www.youtube.com/watch?v=aqrEyuwNORo&ab_channel=ERIKSAULRIVERAREYES

### Conclusion
Our conclusion as a team is that each method has his own advantages you want better times probably SVM would be the best choice, in terms of memory usage DTC is almost every time the best option and MLPC or LR sometimes have better precision or accuracy than the rest, every method has its own uses in the vast technology that can be machine learning.

### References.
[1] https://link.springer.com/chapter/10.1007/978-3-319-18305-3_1

[2] https://link.springer.com/chapter/10.1007/978-1-4899-7641-3_9

[3] https://www.researchgate.net/profile/Thorsten-Joachims/publication/243763293_SVMLight_Support_Vector_Machine/links/5b0eb5c2a6fdcc80995ac3d5/SVMLight-Support-Vector-Machine.pdf

[4] https://ieeexplore.ieee.org/abstract/document/5991826

[5]https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.106.682658

[6] https://ieeexplore.ieee.org/abstract/document/279189

[7] https://dl.acm.org/doi/abs/10.1145/2939502.2939516

[8] https://john.cs.olemiss.edu/~hcc/papers/MultiparadigmProgInScala.pdf
