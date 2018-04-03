name := "quanto"

version := "1.0"

scalaVersion := "2.11.8"

scalacOptions ++= Seq("-feature", "-language:implicitConversions")

retrieveManaged := true

fork := true

//seq(com.github.retronym.SbtOneJar.oneJarSettings: _*)

resolvers += "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/"
 
libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.4.10" withSources() withJavadoc()

libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.1.2"

libraryDependencies += "com.fasterxml.jackson.module" % "jackson-module-scala" % "2.1.2"

libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.1" % "test"

libraryDependencies += "org.scala-lang.modules" % "scala-swing_2.11" % "1.0.1"

//libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value

libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "1.0.5"

//EclipseKeys.withSource := true

//exportJars := true

Seq(appbundle.settings: _*)

appbundle.mainClass := Some("quanto.gui.QuantoDerive")

appbundle.javaVersion := "1.7+"

appbundle.screenMenu := true

appbundle.name := "QuantoDerive"

appbundle.normalizedName := "quantoderiveapp"

appbundle.organization := "org.quantomatic"

appbundle.version := "0.2.0"

appbundle.icon := Some(file("../docs/graphics/quantoderive.icns"))

test in assembly := {}

assemblyJarName in assembly := "Quantomatic.jar"

mainClass in assembly := Some("quanto.gui.QuantoDerive")


scalacOptions ++= Seq("-unchecked", "-deprecation")

// to comment out to try out the others main
mainClass in (Compile, run) := Some("quanto.gui.QuantoDerive")
