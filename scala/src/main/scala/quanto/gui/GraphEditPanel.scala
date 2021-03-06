package quanto.gui

import java.awt
import java.awt.{Font, Toolkit}
import java.awt.datatransfer.StringSelection
import java.awt.event.ActionListener

import javax.swing.{ImageIcon, JButton, JLabel, JTextField}
import quanto.data._
import quanto.gui.graphview.GraphView
import quanto.util.swing.ToolBar

import scala.collection.mutable.ArrayBuffer
import scala.swing._
import scala.swing.event._
import scala.sys.process.{Process, ProcessLogger}
import scala.util.Properties

case class MouseStateChanged(m : MouseState) extends Event

class GraphEditControls(theory: Theory) extends Publisher {

  val VertexTypeLabel  = new Label("Vertex Type:  ") { xAlignment = Alignment.Right; enabled = false }
  val vertexOptions : Seq[String] = theory.vertexTypes.keys.toSeq :+ "<wire>"
  val VertexTypeSelect = new ComboBox(vertexOptions) { enabled = false }
  val EdgeTypeLabel    = new Label("Edge Type:  ") { xAlignment = Alignment.Right; enabled = false }
  val edgeOptions : Seq[String] = theory.edgeTypes.keys.toSeq
  val EdgeTypeSelect   = new ComboBox(edgeOptions) { enabled = false }
  val EdgeDirected     = new CheckBox("directed") { selected = false; enabled = false }

  // Bottom panel
  object BottomPanel extends GridPanel(1,5) {
    contents += (VertexTypeLabel, VertexTypeSelect)
    contents += (EdgeTypeLabel, EdgeTypeSelect, EdgeDirected)
  }

  trait ToolButton { var tool: MouseState = SelectTool() }

  val ge = GraphEditor.getClass

//  val icon = new ImageIcon(GraphEditor.getClass.getResource("select-rectangular.png"), "Select")

  val SelectButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("select-rectangular.png"), "Select")
    tool = SelectTool()
    tooltip = "Select (S)"
    selected = true
  }

  val AddVertexButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("draw-ellipse.png"), "Add Vertex")
    tool = AddVertexTool()
    tooltip = "Add Vertex (V)"
  }

  val AddBoundaryButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("draw-ellipse-b.png"), "Add Boundary")
    tool = AddBoundaryTool()
    tooltip = "Add Boundary (I/O)"
  }

  val AddEdgeButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("draw-path.png"), "Add Edge")
    tool = AddEdgeTool()
    tooltip = "Add Edge (E)"
  }

  val AddBangBoxButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("draw-bbox.png"), "Add Bang Box")
    tool = AddBangBoxTool()
    tooltip = "Add Bang Box (B)"
  }

  val AddMatrixButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("matrix.png"), "Compute Matrix")
    tool = MatrixTool()
    tooltip = "Compute Matrix"
  }

  val GraphToolGroup = new ButtonGroup(SelectButton,
    AddVertexButton,
    AddBoundaryButton,
    AddEdgeButton,
    AddBangBoxButton,
    AddMatrixButton)

  def setMouseState(m : MouseState) {
    val previousTool = GraphToolGroup.selected
    publish(MouseStateChanged(m))
    m match {
      case SelectTool() =>
        VertexTypeLabel.enabled = false
        VertexTypeSelect.enabled = false
        EdgeTypeLabel.enabled = false
        EdgeTypeSelect.enabled = false
        EdgeDirected.enabled = false
        GraphToolGroup.select(SelectButton)
      case AddVertexTool() =>
        if(previousTool.nonEmpty && previousTool.get == AddVertexButton){
          //VertexTypeSelect.selection.index = (VertexTypeSelect.selection.index + 1) % vertexOptions.size
        }
        VertexTypeLabel.enabled = true
        VertexTypeSelect.enabled = true
        EdgeTypeLabel.enabled = false
        EdgeTypeSelect.enabled = false
        EdgeDirected.enabled = false
        GraphToolGroup.select(AddVertexButton)
      case AddEdgeTool() =>
        if(previousTool.nonEmpty && previousTool.get == AddEdgeButton){
          //EdgeTypeSelect.selection.index = (EdgeTypeSelect.selection.index + 1) % edgeOptions.size
        }
        VertexTypeLabel.enabled = false
        VertexTypeSelect.enabled = false
        EdgeTypeLabel.enabled = true
        EdgeTypeSelect.enabled = true
        EdgeDirected.enabled = true
        GraphToolGroup.select(AddEdgeButton)
      case AddBangBoxTool() =>
        VertexTypeLabel.enabled = false
        VertexTypeSelect.enabled = false
        EdgeTypeLabel.enabled = false
        EdgeTypeSelect.enabled = false
        EdgeDirected.enabled = false
        GraphToolGroup.select(AddBangBoxButton)
      case AddBoundaryTool() =>
        VertexTypeLabel.enabled = true
        VertexTypeSelect.enabled = true
        EdgeTypeLabel.enabled = false
        EdgeTypeSelect.enabled = false
        EdgeDirected.enabled = false
        GraphToolGroup.select(AddBoundaryButton)
      case MatrixTool() =>
        QuantoDerive.MainTabbedPane.currentContent match {
          case Some(doc: HasDocument) =>
            doc.document.file match {
              case Some(_) =>
              case None    => doc.document.showSaveAsDialog(QuantoDerive.CurrentProject.map(_.rootFolder))
            }
            if (doc.document.unsavedChanges) {
              Dialog.showMessage(title = "Unsaved Changes",
                message = "You need to save the document before",
                messageType = Dialog.Message.Info
              )
            } else {
              def executeMainCommand(_command: String): Unit = {
                var result = ""
                var error = ""
                val logger = ProcessLogger(
                  (o: String) => println(o),
                  (e: String) => error += e)
                var errorOccurred = false
                try
                  result = Process(_command).!!(logger)
                catch {
                  case _: Exception =>
                    errorOccurred = true
                    val errors = error.split(".Error: ")
                    if (errors.length > 1) {
                      result = "Error : " + errors(1)
                    } else {
                      result = error
                    }
                }
                if (errorOccurred) {
                  Dialog.showMessage(title = "error", message = result, messageType = Dialog.Message.Error)
                } else {
                  val resultArray = result.split("_______________")
                  var content: Array[AnyRef] = null

                  def copyButton(buttonName: String, copyText: String): JButton = {
                    val copyButton = new JButton(buttonName)
                    val copyAction = new ActionListener {
                      override def actionPerformed(e: awt.event.ActionEvent): Unit = {
                        val stringSelection = new StringSelection(copyText)
                        val clipboard = Toolkit.getDefaultToolkit.getSystemClipboard
                        clipboard.setContents(stringSelection, null)
                      }
                    }
                    copyButton.addActionListener(copyAction)
                    copyButton
                  }

                  var matrix: String = resultArray(1)
                  // we remove excess line breaks
                  val charArrayMatrix = ArrayBuffer(matrix.toCharArray: _*)
                  while (charArrayMatrix(0) == '\n' || charArrayMatrix(0) == '\r') {charArrayMatrix.remove(0)}
                  while (charArrayMatrix.last == '\n' || charArrayMatrix.last == '\r') {charArrayMatrix.remove(charArrayMatrix.length - 1)}
                  matrix = charArrayMatrix.mkString("")
                  val matrixLabelText: String = "<html>" + matrix.replace("\n", "<br>")
                    .replace(" ", "&nbsp;")
                  val matrixLabel: JLabel = new JLabel(matrixLabelText)
                  matrixLabel.setFont(new Font("monospaced", Font.PLAIN, 12))
                  content = Array(matrixLabel,
                    copyButton("Raw copy", matrix)
                    ,copyButton("Latex copy", latexizeMatrix(matrix))
                    ,copyButton("Sage copy", sageizeMatrix(matrix))
                    )
                  if (resultArray.length != 1) {
                    content = resultArray(0) +: content
                  }
                  Dialog.showMessage(title = "Graph Matrix Result", message = content)
                }
              }
              def latexizeMatrix(rawMatrix: String): String = {
                var latexizedMatrix: String = ""
                val lines: Array[String] = rawMatrix.split("\n")
                val numberOfMatrices = lines.last.split("\\+ *\\[").length
                val matricesLines : Array[Array[String]] = Array.ofDim(numberOfMatrices, lines.length)
                for (lineIndex <- lines.indices) {
                  val lineMatrices : Array[String] = lines(lineIndex).split("\\] *\\[")
                  for (matrixIndex <- lineMatrices.indices) {
                    matricesLines(matrixIndex)(lineIndex) = lineMatrices(matrixIndex).split("\\]")(0).split("\\[").last
                  }
                } // last line need to be treated differently
                val lastLineMatrices : ArrayBuffer[String] = ArrayBuffer(lines.last.split("\\]\\]"): _*)
                lastLineMatrices.remove(lastLineMatrices.length - 1)
                for (matrixIndex <- lastLineMatrices.indices) {
                  var a = lastLineMatrices(matrixIndex).split("\\]")(0)
                  a = a.split("\\[").last
                  matricesLines(matrixIndex)(lines.length - 1) = a
//                  matricesLines(matrixIndex)(lines.length - 1) = lastLineMatrices(matrixIndex).split("\\]")(0).split("\\[").last
                }
                val latexMatrices : Array[String] = Array.ofDim(numberOfMatrices)
                for (matrixIndex <- matricesLines.indices) {
                  var latexMatrix = "\\begin{pmatrix}"
                  for (matrixLine <- matricesLines(matrixIndex)) {
                    latexMatrix += matrixLine.split(" ").filter(_ != "").mkString("&") + "\\\\"
                  }
                  latexMatrix += "\\end{pmatrix}"
                  latexMatrices(matrixIndex) = latexMatrix
                } // now all the matrices are under latex format, all if left is to join them
                if (latexMatrices.length == 4) {
                  val n = rawMatrix.split("2\\^").last.split("\\)")(0)
                  latexizedMatrix = "$\\frac{1}{2^" + n + "}\\left("
                  latexizedMatrix += latexMatrices(0)
                  latexizedMatrix += "+e^{i\\pi/4}" + latexMatrices(1)
                  latexizedMatrix += "+e^{2i\\pi/4}" + latexMatrices(2)
                  latexizedMatrix += "+e^{3i\\pi/4}" + latexMatrices(3)
                  latexizedMatrix += "\\right)$"
                } else { // this case is not handled for now, so only the matrices will be returned
                  latexizedMatrix = latexMatrices.mkString("\\\\")
                }
                latexizedMatrix
              }
              def sageizeMatrix(rawMatrix: String): String = {
                var sageizedMatrix: String = ""
                val lines: Array[String] = rawMatrix.split("\n")
                val numberOfMatrices = lines.last.split("\\+ *\\[").length
                val matricesLines : Array[Array[String]] = Array.ofDim(numberOfMatrices, lines.length)
                for (lineIndex <- lines.indices) {
                  val lineMatrices : Array[String] = lines(lineIndex).split("\\] *\\[")
                  for (matrixIndex <- lineMatrices.indices) {
                    matricesLines(matrixIndex)(lineIndex) = lineMatrices(matrixIndex).split("\\]")(0).split("\\[").last
                  }
                } // last line need to be treated differently
                val lastLineMatrices : ArrayBuffer[String] = ArrayBuffer(lines.last.split("\\]\\]"): _*)
                lastLineMatrices.remove(lastLineMatrices.length - 1)
                for (matrixIndex <- lastLineMatrices.indices) {
                  matricesLines(matrixIndex)(lines.length - 1) = lastLineMatrices(matrixIndex).split("\\]")(0).split("\\[").last
                }
                val sageMatrices : Array[String] = Array.ofDim(numberOfMatrices)
                for (matrixIndex <- matricesLines.indices) {
                  var sageMatrix = "matrix(["
                  for (lineIndex <- matricesLines(matrixIndex).indices) {
                    sageMatrix += "[" + matricesLines(matrixIndex)(lineIndex).split(" ").filter(_ != "").mkString(",") + "]"
                    if (lineIndex != matricesLines(matrixIndex).length -1) {
                      sageMatrix += ","
                    }
                  }
                  sageMatrix += "])"
                  sageMatrices(matrixIndex) = sageMatrix
                } // now all the matrices are under latex format, all if left is to join them
                if (sageMatrices.length == 4) {
                  val n = rawMatrix.split("2\\^").last.split("\\)")(0)
                  sageizedMatrix = "1/2^" + n + "*("
                  sageizedMatrix += sageMatrices(0)
                  sageizedMatrix += "+exp(i*pi/4)*" + sageMatrices(1)
                  sageizedMatrix += "+exp(i*pi/2)*" + sageMatrices(2)
                  sageizedMatrix += "+exp(3*i*pi/4)*" + sageMatrices(3)
                  sageizedMatrix += ")"
                } // the other cases are not handled for now
                sageizedMatrix
              }
              val graphPath = doc.document.file.get.getAbsolutePath
              // the .jar is meant to be run form the scala directory
              val pythonSrcPath = "../src/"
              val mainPath = pythonSrcPath + "main.py"
              val numberedAnnotationsPath = pythonSrcPath + "numbered_annotations.py"
              val OS = Properties.envOrNone("OS")
              var OSValue = ""
              OS match {
                case Some(v) => OSValue = v
                case _ =>
              }
              var command = ""
              if (OSValue.contains("Windows")) {
                command = "python " + numberedAnnotationsPath + " \"" + graphPath + "\""
              } else {
                command = "python3 " + numberedAnnotationsPath + " " + graphPath
              }
              val numberedAnnotationsReturn = Process(command).!!
              val numberedAnnotationsReturnArray = numberedAnnotationsReturn.split(";")
              numberedAnnotationsReturnArray(1) = numberedAnnotationsReturnArray(1).replaceAll(" ", "")
              numberedAnnotationsReturnArray(2) = numberedAnnotationsReturnArray(2).replaceAll(" ", "")
              numberedAnnotationsReturnArray(1) = numberedAnnotationsReturnArray(1).replaceAll("\'", "")
              numberedAnnotationsReturnArray(2) = numberedAnnotationsReturnArray(2).replaceAll("\'", "")
              if (numberedAnnotationsReturnArray(0).contains("True")) {
                var command = ""
                if (OSValue.contains("Windows")) {
                  command = "python " + mainPath + " \"" + graphPath + "\" " + numberedAnnotationsReturnArray(1) + " " +
                    numberedAnnotationsReturnArray(2)
                } else {
                  command = "python3 " + mainPath + " " + graphPath + " " + numberedAnnotationsReturnArray(1) + " " +
                    numberedAnnotationsReturnArray(2)
                }
                command = command.dropRight(2)
                executeMainCommand(command)
              } else {
                val inputs = new JTextField
                val outputs = new JTextField
                val message: Array[AnyRef] = Array("Inputs:", inputs, "Outputs:", outputs)
                val option = Dialog.showConfirmation(message = message, title = "Graph Matrix Dialogue",
                  optionType = Dialog.Options.OkCancel)
                if (option == Dialog.Result.Ok) {
                  val inputList = inputs.getText()
                  val outputList = outputs.getText()
                  var command = ""
                  if (OSValue.contains("Windows")) {
                    command = "python " + mainPath + " \"" + graphPath + "\" [" + inputList + "] [" + outputList + "]"
                  } else {
                    command = "python3 " + mainPath + " " + graphPath + " [" + inputList + "] [" + outputList + "]"
                  }
                  executeMainCommand(command)
                }
              }
              setMouseState(SelectTool())
              }
          case _ =>
        }
      case _ =>
    }
  }

  GraphToolGroup.buttons.foreach(listenTo(_))
  reactions += {
    case ButtonClicked(t: ToolButton) =>
      setMouseState(t.tool)
  }

  val MainToolBar = new ToolBar {
    contents += (SelectButton, AddVertexButton, AddBoundaryButton, AddEdgeButton, AddBangBoxButton, AddMatrixButton)
  }
}


class GraphEditPanel(val theory: Theory, val readOnly: Boolean = false)
extends BorderPanel
with HasDocument
{

  val document = new GraphDocument(this, theory)
//  def graph = document.graph
//  def graph_=(g: Graph) { document.graph = g }

  // GUI components
  val graphView = new GraphView(theory, document) {
    drawGrid = true
    focusable = true
  }

  val controls = new GraphEditControls(theory)

  // alias for graph_=, used in java code
//  def setGraph(g: Graph) { graph_=(g) }

  val graphEditController = new GraphEditController(graphView, document.undoStack, readOnly)
  graphEditController.controlsOpt = Some(controls)

  val GraphViewScrollPane = new ScrollPane(graphView)

  if (!readOnly) {
    add(controls.MainToolBar, BorderPanel.Position.North)
    add(controls.BottomPanel, BorderPanel.Position.South)
  }

  add(GraphViewScrollPane, BorderPanel.Position.Center)


  listenTo(GraphViewScrollPane, controls, document)

  reactions += {
    case UIElementResized(GraphViewScrollPane) =>
      graphView.resizeViewToFit()
      graphView.repaint()
    case MouseStateChanged(m) =>
      graphEditController.mouseState = m
  }
}
