![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática.
### Proyecto / Tarea / Práctica:
#### Basic Statistics
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno

### Fecha:
#### Tijuana Baja California a 18 de 05 2022 

# Basic statistics
Basic statistics lays the foundation for further studies in statistics.
It includes lots of ways to classify and sort variables and data so that they can be studied with tools you’ll be introduced to later. 
For example, correlation and hypothesis testing. Perhaps surprisingly, calculus-based statistics is often combined with basic statistics courses. 
Whether you need calculus with statistics depends largely on what your career goals are. 
For example, if you intend on becoming a university researcher, you’ll need calculus; If you’re a nursing student, calculus is optional.

  
### Example
  
```scala
package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.apache.spark.rdd.RDD
// $example off$

object HypothesisTestingExample {

def main() {

    val conf = new SparkConf().setAppName("HypothesisTestingExample")
    val sc = new SparkContext(conf)

    // $example on$
    // a vector composed of the frequencies of events
    val vec: Vector = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25)

    // compute the goodness of fit. If a second vector to test against is not supplied
    // as a parameter, the test runs against a uniform distribution.
    val goodnessOfFitTestResult = Statistics.chiSqTest(vec)
    // summary of the test including the p-value, degrees of freedom, test statistic, the method
    // used, and the null hypothesis.
    println(s"$goodnessOfFitTestResult\n")

    // a contingency matrix. Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    val mat: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

    // conduct Pearson's independence test on the input contingency matrix
    val independenceTestResult = Statistics.chiSqTest(mat)
    // summary of the test including the p-value, degrees of freedom
    println(s"$independenceTestResult\n")

    val obs: RDD[LabeledPoint] =
    sc.parallelize(
        Seq(
        LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
        LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 0.0)),
        LabeledPoint(-1.0, Vectors.dense(-1.0, 0.0, -0.5)
        )
        )
    ) // (label, feature) pairs.

    // The contingency table is constructed from the raw (label, feature) pairs and used to conduct
    // the independence test. Returns an array containing the ChiSquaredTestResult for every feature
    // against the label.
    val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
    featureTestResults.zipWithIndex.foreach { case (k, v) =>
    println(s"Column ${(v + 1)} :")
    println(k)
    }  // summary of the test
    // $example off$

    sc.stop()
    }
}

```

### Code Running
![image](https://user-images.githubusercontent.com/40293937/169226614-e4f49428-b37c-4768-b961-07f043f1c9dc.png)


### Conclusion
Statistics is fundamental to ensuring meaningful, accurate information is extracted from Big Data. The following issues are crucial and are only exacerbated by Big Data: 
o Data quality and missing data o Observational nature of data, so that causal questions such as the comparison of interventions may be subject to confounding. o Quantification of the uncertainty of predictions, forecasts and models 
• The scientific discipline of statistics brings sophisticated techniques and models to bear on these issues 
• Statisticians help translate the scientific question into a statistical question, which includes carefully describing data structure; the underlying system that generated the data (the model); and what we are trying to assess (the parameter or parameters we wish to estimate) or predict




### References.
[1]http://higherlogicdownload.s3.amazonaws.com/AMSTAT/UploadedImages/49ecf7cf-cb26-4c1b-8380-3dea3b7d8a9d/BigDataOnePager.pdf
[2]https://github.com/JoseAguilar9812/datos_masivos-basic_statistics-expo/blob/main/expo.md

