package quanto.gui

import java.awt.Font

import javax.swing.{ImageIcon, JLabel, JTextField}
import quanto.data._
import quanto.gui.graphview.GraphView
import quanto.util.swing.ToolBar

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

  val MatrixButton = new ToggleButton() with ToolButton {
    icon = new ImageIcon(GraphEditor.getClass.getResource("matrix.png"), "Compute Matrix")
    tool = MatrixTool()
    tooltip = "Compute Matrix"
  }

  val GraphToolGroup = new ButtonGroup(SelectButton,
    AddVertexButton,
    AddBoundaryButton,
    AddEdgeButton,
    AddBangBoxButton,
    MatrixButton)

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
              if (OSValue.contains("Windows")){
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
              if (numberedAnnotationsReturnArray(0).contains("True")){
                var command = ""
                if (OSValue.contains("Windows")){
                  command = "python " + mainPath + " \"" + graphPath + "\" " + numberedAnnotationsReturnArray(1) + " " +
                    numberedAnnotationsReturnArray(2)
                } else {
                  command = "python3 " + mainPath + " " + graphPath + " " + numberedAnnotationsReturnArray(1) + " " +
                    numberedAnnotationsReturnArray(2)
                }
                command = command.dropRight(2)
                var result = ""
                var error = ""
                val logger = ProcessLogger(
                  (o: String) => println(o),
                  (e: String) => error += e)
                var errorOccurred = false
                try {
                  result = Process(command).!!(logger)
                } catch {
                  case _ : Exception =>
                    errorOccurred = true
                    val errors = error.split("NameError: ")
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
                  if (resultArray.length == 1){
                    Dialog.showMessage(title = "Graph Matrix Result", message = resultArray(0))
                  } else {
                    var labelText = "<html>" + resultArray(1)
                    labelText = labelText.replace("\n", "<br>").replace(" ", "&nbsp;")
                    val resultLabel = new JLabel(labelText)
                    resultLabel.setFont(new Font("monospaced", Font.PLAIN, 12))
                    val content: Array[AnyRef] = Array(resultArray(0), resultLabel, new JTextField(resultArray(1)))
                    Dialog.showMessage(title = "Graph Matrix Result", message = content)
                  }
                }
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
                  if (OSValue.contains("Windows")){
                    command = "python " + mainPath + " \"" + graphPath + "\" [" + inputList + "] [" + outputList + "]"
                  } else {
                    command = "python3 " + mainPath + " " + graphPath + " [" + inputList + "] [" + outputList + "]"
                  }
                  var result = ""
                  var error = ""
                  val logger = ProcessLogger(
                    (o: String) => println(o),
                    (e: String) => error += e)
                  var errorOccurred = false
                  try {
                    result = Process(command).!!(logger)
                  } catch {
                    case e : Exception =>
                      errorOccurred = true
                      val errors = error.split("ValueError: ")
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
                    var labelText = "<html>" + resultArray(1)
                    labelText = labelText.replace("\n", "<br>").replace(" ", "&nbsp;")
                    val resultLabel = new JLabel(labelText)
                    resultLabel.setFont(new Font("monospaced", Font.PLAIN, 12))
                    if (resultArray.length == 1) {
                      val content: Array[AnyRef] = Array(resultLabel, new JTextField(resultArray(0)))
                      Dialog.showMessage(title = "Graph Matrix Result", message = content)
                    } else {
                      val content: Array[AnyRef] = Array(resultArray(0), resultLabel, new JTextField(resultArray(1)))
                      Dialog.showMessage(title = "Graph Matrix Result", message = content)
                    }
                  }
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
    contents += (SelectButton, AddVertexButton, AddBoundaryButton, AddEdgeButton, AddBangBoxButton, MatrixButton)
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
