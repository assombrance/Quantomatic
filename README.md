# Quantomatic
ZX-calculus diagram to matrix plugin

## Run
To run this version of Quantomatic, you need:
- Python3
- numpy (python package, more info [here](https://scipy.org/install.html))  
tldr : `pip3 install numpy` should do the trick
- JRE version 7 or newer

For now, if you need only the .jar version and if you can't compile for a reason, ask me directly at quantomatic.matrix@gmail.com

## Use
To use the functionnality added by this project, simply open a graph in Quantomatic and clic on the right most button (in shape of a matrix) or press `M`. 

At this point, a prompt will ask you to give the I/O for the compitation, write them with a comma between each (no space).
For example, if the wire are b0, b1, b2 and b3, you could have:
```
Inputs : b3,b1
Outputs : b0,b2
```

Alternatively, you can annotate the edges going to the wire, with `input`/`in`/`i` or `output`/`out`/`o` and the number of the I/O, to follow the previous example, we would have :
<pre>
  b0         b1
   |          |
output0     input1
   |          |
       ...
   |          |
output1     input0
   |          |
  b2          b3
</pre>
If the I/O fields are not completely filled, a last step is executed : if the label of the edges contain `input`/`in`/`i` or `output`/`out`/`o` without any numbers, the arder will be assumed (randomly) and displayed above the result with a warning.

## Compile
To compile Quantomatic, you need:
- JDK version 7 or newer
- sbt

### Run from source
Don't forget to install the pieces of software listed in the [Run](#run) section for this one
~~~~
$ cd [project directory]/scala
$ sbt run
~~~~

### Compile and package in a jar file
~~~~
$ cd [project directory]/scala
$ sbt assembly
~~~~

On linux and Mac OS, you can use the stript `Quantomatic.sh` to run the compiled version.  
Use `Quantomatic.bat` instead on Windows.

## Documentation
You can compile the documentation using Sphinx (I used `Sphinx v1.7.2`).
To compile, simply run :
~~~
$ [project directory]/doc/_templates/make html
~~~

## Disclaimer
This software is a for of the Quantomatic project from akissinger.  
Exept from the matrix bit, everything has been done [there](https://github.com/Quantomatic/quantomatic).  
This project may eventually be included in the official Quantomatic.
