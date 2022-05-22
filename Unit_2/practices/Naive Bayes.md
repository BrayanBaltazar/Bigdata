![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática.
### Proyecto / Tarea / Práctica:
#### Naive Bayes
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno

### Fecha:
#### Tijuana Baja California a 22 de 05 2022 

# Linear Support Vector Machine
Naive Bayes algorithm is a classification algorithm based on Bayes' theorems, and can be used for both exploratory and predictive modeling. The word naïve in the name Naïve Bayes derives from the fact that the algorithm uses Bayesian techniques but does not take into account dependencies that may exist.
This algorithm is less computationally intense than other Microsoft algorithms, and therefore is useful for quickly generating mining models to discover relationships between input columns and predictable columns. You can use this algorithm to do initial exploration of data, and then later you can apply the results to create additional mining models with other algorithms that are more computationally intense and more accurate.

  
### Example
  
```scala
//Importar las librerias necesarias

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

//Cargar los datos especificando la ruta del archivo

val data = spark.read.format("libsvm").load("C:/spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

println ("Numero de lineas en el archivo de datos:" + data.count ())

//Mostrar las primeras 20 líneas por defecto

data.show()

//Divida aleatoriamente el conjunto de datos en conjunto de entrenamiento y conjunto de prueba de acuerdo con los pesos proporcionados. También puede especificar una seed

val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3), 100L)
// El resultado es el tipo de la matriz, y la matriz almacena los datos de tipo DataSet

//Incorporar al conjunto de entrenamiento (operación de ajuste) para entrenar un modelo bayesiano
val naiveBayesModel = new NaiveBayes().fit(trainingData)

//El modelo llama a transform() para hacer predicciones y generar un nuevo DataFrame.

val predictions = naiveBayesModel.transform(testData)

//Salida de datos de resultados de predicción
predictions.show()

//Evaluación de la precisión del modelo

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
// Precisión
val precision = evaluator.evaluate (predictions) 

//Imprimir la tasa de error
println ("tasa de error =" + (1-precision))





```

### Code Running



### Conclusion
Naive Bayes models can be used to tackle large scale classification problems for which the full training set might not fit in memory. 
To handle this case, MultinomialNB, BernoulliNB, and GaussianNB expose a partial_fit method that can be used incrementally as done with other classifiers as demonstrated in Out-of-core classification of text documents. All naive Bayes classifiers support sample weighting.


### References.
[1]https://docs.microsoft.com/en-us/analysis-services/data-mining/microsoft-naive-bayes-algorithm?view=asallproducts-allversions
[2] https://scikit-learn.org/stable/modules/naive_bayes.html



