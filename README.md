# Quantomatic
ZX-calculus diagram to matrix plugin

## Run
To run this version of Quantomatic, you need:
- Python3
- numpy (python package, more info [here](https://scipy.org/install.html))
- JRE version 7 or newer

For now, if you need only the .jar version and if you can't compile for a reason, ask me directly at quantomatic.matrix@gmail.com

## Compile
To compile Quantomatic, you need:
- JDK version 7 or newer
- sbt

### Run from source
Don't forget to install the pieces of software listed in the [#Run] section for this one
~~~~
$ cd [project directory]/scala
$ sbt run
~~~~

### Compile and package in a jar file
~~~~
$ cd [project directory]/scala
$ sbt assembly
~~~~

On linux, you can use the stript `Quantomatic.sh` to run the compiled version.
Use `Quantomatic.bat` instead on Windows.

## Disclaimer
This software is a for of the Quantomatic project from akissinger.
Exept from the matrix bit, everything has been done [there](https://github.com/Quantomatic/quantomatic).
This project may eventually be included in the official Quantomatic.
Run
