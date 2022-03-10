//PRACTICA 4
C:\Spark\spark-3.2.1-bin-hadoop3.2\bin\spark-shell

//Fibonacci por medio de recursion basica

def fib(n: Int): BigInt = n match {
  case 0 => 0
  case 1 => 1
  case _ => fib(n-2) + fib(n-1)
}

(0 to 10).foreach(n => print(fib(n) + " "))
fib(50)

//Fibnacci usando el metodo "tail"

def fibTR(num: Int): BigInt = {
  @scala.annotation.tailrec
  def fibFcn(n: Int, acc1: BigInt, acc2: BigInt): BigInt = n match {
    case 0 => acc1
    case 1 => acc2
    case _ => fibFcn(n - 1, acc2, acc1 + acc2)
  }
 
  fibFcn(num, 0, 1)
}
fibTR(90)

//Fibonacci usando un mapa mutable
def memoize[K, V](f: K => V): K => V = {
  val cache = scala.collection.mutable.Map.empty[K, V]
  k => cache.getOrElseUpdate(k, f(k))
}
val fibM: Int => BigInt = memoize(n => n match {
  case 0 => 0
  case 1 => 1
  case _ => fibM(n-2) + fibM(n-1)
})
fibM(90)