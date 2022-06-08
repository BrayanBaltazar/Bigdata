![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática.
### Proyecto / Tarea / Práctica:
#### Linear Support Vector Machine
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
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

  
### Example
  
```scala
// Linear Support Vector Machine

// Import the "LinearSVC" library.
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training  = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

//Set the maximum number of iterations and the regularization parameter 
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)


// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")



```


### Conclusion
In the SVM algorithm, we are looking to maximize the margin between the data points and the hyperplane. The loss function that helps maximize the margin is hinge loss.





### References.
[1] https://github.com/uliomar87/Linear-Support-Vector-Machine-/tree/patch-1


