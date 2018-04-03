# Quantomatic
ZX-calculus diagram to matrix plugin

## Run
To run this version of Quantomatic, you need:
- Python3
- JRE version 7 or newer

For now, if you need only the .jar version if you can't compile for a reason, ask me directly at quantomatic.matrix@gmail.com

## Compile
To compile Quantomatic, you need:
- Python3
- JDK version 7 or newer
- sbt

An issue with sbt caused the impossibility to compile on windows. It needs further testing but for now, linux works just fine.

### Run from source
with sbt installed
~~~~
$ cd [project directory]/scala
$ sbt run
~~~~

### To compile Quantomatic
with sbt installed
~~~~
$ cd [project directory]/scala
$ sbt assembly
~~~~

On linux, you can use the stript `Quantomatic.sh` to run the compiled version

## Disclaimer
This software is a for of the Quantomatic project from akissinger.
Exept from the matrix bit, everything has been done [there](https://github.com/Quantomatic/quantomatic).
This project may eventually be included in the official Quantomatic.
