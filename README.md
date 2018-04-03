# Quantomatic
ZX-calculus diagram to matrix plugin

## Run
To run this version of Quantomatic, you need:
- Python3
- JRE version 7 or newer

For now, if you need only the .jar version and if you can't compile for a reason, ask me directly at quantomatic.matrix@gmail.com

## Compile
To compile Quantomatic, you need:
- Python3
- JDK version 7 or newer
- sbt

To compile on Windows or Linux, make sure that the right line is enabled in `[project directory]/scala/src/main/scala/quanto/gui/GraphEditPanel.scala`
On Linux, you must have, on line 144:
```scala
// For Linux
val command = "python3 " + mainPath + " " + graphPath + " [" + inputList + "] [" + outputList + "]"
// For Windows
// val command = "python " + mainPath + " " + graphPath + " [" + inputList + "] [" + outputList + "]"
```
And on Windows:
```scala
// For Linux
// val command = "python3 " + mainPath + " " + graphPath + " [" + inputList + "] [" + outputList + "]"
// For Windows
val command = "python " + mainPath + " " + graphPath + " [" + inputList + "] [" + outputList + "]"
```
For now, only path without space are supported on Windows

### Run from source
In `[project directory]/scala/src/main/scala/quanto/gui/GraphEditPanel.scala`, line 134, make sure that the run option is enabled:
```scala
// For assembly
// val mainPath = "src/main.py"
// For run
val mainPath = "../src/main.py"
```
~~~~
$ cd [project directory]/scala
$ sbt run
~~~~

### Compile
In `[project directory]/scala/src/main/scala/quanto/gui/GraphEditPanel.scala`, line 134, make sure that the assembly option is enabled:
```scala
// For assembly
val mainPath = "src/main.py"
// For run
// val mainPath = "../src/main.py"
```
~~~~
$ cd [project directory]/scala
$ sbt assembly
~~~~

On linux, you can use the stript `Quantomatic.sh` to run the compiled version

## Disclaimer
This software is a for of the Quantomatic project from akissinger.
Exept from the matrix bit, everything has been done [there](https://github.com/Quantomatic/quantomatic).
This project may eventually be included in the official Quantomatic.
