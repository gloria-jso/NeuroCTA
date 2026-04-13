import traceback
from pathlib import Path
from typing import Optional

import qt
import slicer
import ctk
import vtk
from slicer.parameterNodeWrapper import parameterNodeWrapper
import numpy as np

from .InstallLogic import InstallLogic, InstallLogicProtocol
from .Parameter import Parameter
from .Logic import Logic, LogicProtocol, SkeletonizationLogic, ClassificationLogic
from .Signal import Signal
from .VesselTableManager import VesselTableManager
from .VesselGNN import VesselGNN, VesselSAGE

import time

@parameterNodeWrapper
class WidgetParameterNode:
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    parameter: Parameter


class Widget(qt.QWidget):
    """
    nnUNet widget containing an install and run settings collapsible areas.
    Allows to run nnUNet model and displays results in the UI.
    Saves the used settings to QSettings for reloading.
    """

    def __init__(
            self,
            logic: Optional[LogicProtocol] = None,
            skeletonizationLogic: Optional[LogicProtocol] = None,
            parent=None
    ):
        super().__init__(parent)

        self.logic = logic or Logic()
        self.skelLogic = skeletonizationLogic or SkeletonizationLogic()
        self.classLogic = ClassificationLogic()
        self.vesselTable = VesselTableManager()


        self._currentDisplayNode = None
        self._pickingSegmentationNode = None
        self._pickingInteractor = None
        self._pickingObserverTag = None
        self._tooltipLabel = None
        self._ijkToRas = None

        # Instantiate widget UI
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ─────────────────── Inputs collapsible ──────────────────────
        # Inputs form layout directly in the main layout
        inputsFormLayout = qt.QFormLayout()
        layout.addLayout(inputsFormLayout)
        inputsFormLayout.setContentsMargins(10,10,10,10)

        # Input Type combo
        self.unsegmentedRadio = qt.QRadioButton("Unsegmented volume")
        self.segmentationRadio = qt.QRadioButton("Segmentation")
        self.unsegmentedRadio.setChecked(True)

        self.inputTypeGroup = qt.QButtonGroup()
        self.inputTypeGroup.addButton(self.unsegmentedRadio)
        self.inputTypeGroup.addButton(self.segmentationRadio)
        
        # Layout to hold them
        inputTypeLayout = qt.QHBoxLayout()
        inputTypeLayout.addWidget(self.unsegmentedRadio)
        inputTypeLayout.addWidget(self.segmentationRadio)

        # Add to form
        inputsFormLayout.addRow("Input type:", inputTypeLayout)

        # ---- Input Volume selector
        self.inputLabel = qt.QLabel("Input Volume:")
        self.inputLabel.setFixedWidth(135)
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        
        self.inputSelector.setMRMLScene(slicer.mrmlScene)

        inputsFormLayout.addRow(self.inputLabel, self.inputSelector)

        # --- Segments Table
        self.vesselTable.setVisible(False)
        inputsFormLayout.addItem(qt.QSpacerItem(0, 6))
        inputsFormLayout.addRow(self.vesselTable)

        self._exportButton = qt.QPushButton("Export Segments Table to CSV")
        self._exportButton.connect("clicked()", self._onExportTable)
        inputsFormLayout.addRow(self._exportButton)

        # ─────────────────── Segmentation collapsible ────────────────────────
        self.segCollapsibleButton = ctk.ctkCollapsibleButton()
        self.segCollapsibleButton.setStyleSheet("""
            ctkCollapsibleButton {
                background-color: #d9dbd9;  /* background color */
                border: 1px solid gray;
                border-radius: 4px;
            }
        """)
        self.segCollapsibleButton.text = "Segmentation"
        self.segCollapsibleButton.collapsed = True
        layout.addWidget(self.segCollapsibleButton)

        segmentationFormLayout = qt.QFormLayout(self.segCollapsibleButton)
        segmentationFormLayout.setContentsMargins(12,12,12,12)

        # nnUNetv2 Configuration
        self.nnUNetOptionsGroupBox = qt.QGroupBox()
        self.nnUNetOptionsLayout = qt.QFormLayout(self.nnUNetOptionsGroupBox)
        self.modelPathEdit = ctk.ctkPathLineEdit()
        self.modelPathEdit.setCurrentPath("/Users/gloriaso/Desktop/BME499/Models/Dataset001_TopBrain")
        self.modelCheckpointEdit = qt.QLineEdit()
        self.modelCheckpointEdit.setText("checkpoint_best.pth")

        # Segmentation method label (own row)
        segmentationFormLayout.addRow(qt.QLabel("Method:"))
        self.segMethods = ["nnUNetv2"]
        self.segMethodGroup = qt.QButtonGroup()
        segMethodLayout = qt.QVBoxLayout()
        segMethodLayout.setAlignment(qt.Qt.AlignLeft)
        segMethodLayout.setContentsMargins(10,0,0,0)

        for method in self.segMethods:
            radio = qt.QRadioButton(method)
            self.segMethodGroup.addButton(radio)
            segMethodLayout.addWidget(radio)

            if method == "nnUNetv2":
                self.nnUNetOptionsLayout.addRow(qt.QLabel("Model path:"), self.modelPathEdit)
                self.nnUNetOptionsLayout.addRow(qt.QLabel("Model checkpoint:"), self.modelCheckpointEdit)
                segMethodLayout.addWidget(self.nnUNetOptionsGroupBox)

        
        self.segMethodGroup.buttons()[0].setChecked(True)
        segmentationFormLayout.addRow(segMethodLayout)

        segmentationFormLayout.addItem(qt.QSpacerItem(0, 5))
        self.runSegPushButton = qt.QPushButton("▶︎ Run Segmentation")
        segmentationFormLayout.addRow("", self.runSegPushButton)


        # ─────────────────── Vessel Classification ────────────────────────
        classCollapsibleButton = ctk.ctkCollapsibleButton()
        classCollapsibleButton.setStyleSheet("""
            ctkCollapsibleButton {
                background-color: #d9dbd9;  /* background color */
                border: 1px solid gray;
                border-radius: 4px;
            }
        """)
        classCollapsibleButton.text = "Vessel Classification"
        classCollapsibleButton.collapsed = True
        layout.addWidget(classCollapsibleButton)

        classificationFormLayout = qt.QFormLayout(classCollapsibleButton)
        classificationFormLayout.setContentsMargins(12, 12, 12, 12)

        # Model path and model type inside a QGroupBox with QFormLayout
        classOptionsGroupBox = qt.QGroupBox()
        classOptionsLayout = qt.QFormLayout(classOptionsGroupBox)

        self.classModelPathEdit = ctk.ctkPathLineEdit()
        self.classModelPathEdit.setCurrentPath("/Users/gloriaso/Desktop/BME499/Models/Graph_Neural_Network/best_model_sage_fold0.pt")
        classOptionsLayout.addRow(qt.QLabel("Model path:"), self.classModelPathEdit)

        # self.segFilePathEdit = ctk.ctkPathLineEdit()
        # self.segFilePathEdit.filters = ctk.ctkPathLineEdit.Files
        # self.segFilePathEdit.setCurrentPath("/Users/gloriaso/Desktop/BME499/Mar4/topcow_ct_024_fullRes-segmentation.seg.nrrd")
        # classOptionsLayout.addRow(qt.QLabel("Segmentation file:"), self.segFilePathEdit)

        # Model type radio buttons
        self.sageRadio = qt.QRadioButton("SAGEConv")
        self.gineRadio = qt.QRadioButton("GINConv")
        self.sageRadio.setChecked(True)
        self.modelTypeGroup = qt.QButtonGroup()
        self.modelTypeGroup.addButton(self.sageRadio)
        self.modelTypeGroup.addButton(self.gineRadio)

        classModelTypeLayout = qt.QHBoxLayout()
        classModelTypeLayout.addWidget(self.sageRadio)
        classModelTypeLayout.addWidget(self.gineRadio)
        classOptionsLayout.addRow(qt.QLabel("Model type:"), classModelTypeLayout)

        classificationFormLayout.addRow(classOptionsGroupBox)

        classificationFormLayout.addItem(qt.QSpacerItem(0, 5))
        self.runInferenceButton = qt.QPushButton("▶︎ Run Vessel Classification")
        classificationFormLayout.addRow("", self.runInferenceButton)
        self.runInferenceButton.connect('clicked()', self.onRunClassInference)


        # ─────────────────── Skeletonization collapsible ───────────────────────
        self.skelCollapsibleButton = ctk.ctkCollapsibleButton()
        self.skelCollapsibleButton.setStyleSheet("""
            ctkCollapsibleButton {
                background-color: #d9dbd9;  /* background color */
                border: 1px solid gray;
                border-radius: 4px;
            }
        """)

        self.skelCollapsibleButton.text = "Skeletonization and Feature Extraction"
        self.skelCollapsibleButton.collapsed = True
        layout.addWidget(self.skelCollapsibleButton)
        skeletonizationFormLayout = qt.QFormLayout(self.skelCollapsibleButton)
        skeletonizationFormLayout.setContentsMargins(12,12,12,12)

        # Skeletonization method radio buttons
        skeletonizationFormLayout.addRow(qt.QLabel("Method:"))
        self.skelMethods = ["Medial axis thinning", "VMTK Extract Centerline"]
        self.skelMethodGroup = qt.QButtonGroup()
        skelMethodLayout = qt.QVBoxLayout()
        skelMethodLayout.setAlignment(qt.Qt.AlignLeft)
        skelMethodLayout.setContentsMargins(10,0,0,0)

        for method in self.skelMethods:
            radio = qt.QRadioButton(method)
            self.skelMethodGroup.addButton(radio)
            skelMethodLayout.addWidget(radio)

        self.skelMethodGroup.buttons()[0].setChecked(True)
        skeletonizationFormLayout.addRow(skelMethodLayout)

        skeletonizationFormLayout.addItem(qt.QSpacerItem(0, 5))
        self.runSkelPushButton = qt.QPushButton("▶︎ Run Skeletonization")
        skeletonizationFormLayout.addRow("", self.runSkelPushButton)

        self.showBranchingPointsCheckBox = qt.QCheckBox("Show branching points")
        self.showBranchingPointsCheckBox.setChecked(True)
        self.showBranchingPointsCheckBox.setEnabled(False)
        skeletonizationFormLayout.addRow("", self.showBranchingPointsCheckBox)

        layout.addStretch(1)


        # ── Connections ─────────────────────────────────────────────────────────
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        # self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.unsegmentedRadio.connect("toggled(bool)", self.onInputTypeChanged)
        self.segmentationRadio.connect("toggled(bool)", self.onInputTypeChanged)

        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputNodeChanged)

        self.segMethodGroup.connect("buttonClicked(QAbstractButton*)", self.onSegMethodChanged)
        self.skelMethodGroup.connect("buttonClicked(QAbstractButton*)", self.onSkelMethodChanged)

        self.runSegPushButton.connect("clicked(bool)", self.onApplySegButton)
        self.runSkelPushButton.connect("clicked(bool)", self.onApplySkelButton)
        self.showBranchingPointsCheckBox.connect("toggled(bool)", self.onToggleBranchingPoints)


        # self.runFeatExPushButton.connect("clicked(bool)", self.onApplyFeatExButton)

        # Make sure parameter node is initialized (needed for module reload)
        # self.initializeParameterNode()

        # Logic connections
        self.logic.progressInfo.connect(self.onProgressInfo)

        # Skel connections
        self.skelLogic.progressInfo.connect(self.onProgressInfo)
        self.skelLogic.errorOccurred.connect(self.onSkelError)
        self.skelLogic.skelFinished.connect(
            lambda *args: self.onSkelFinished()
        )

        self.skelLogic.progressUpdated.connect(self.onSkelProgress)


        # Create parameter node and connect GUI
        self._parameterNode = self._createParameterNode()
        self._parameterNode.parameter = Parameter.fromSettings()
        self._parameterNode.connectParametersToGui(
            {

                "parameter.modelPath": self.modelPathEdit,
                "parameter.modelCheckpointName": self.modelCheckpointEdit,
                # "parameter.device": self.ui.deviceComboBox,
                # "parameter.stepSize": self.ui.stepSizeSlider,
                # "parameter.checkPointName": self.ui.checkPointNameLineEdit,
                # "parameter.folds": self.ui.foldsLineEdit,
                # "parameter.nProcessPreprocessing": self.ui.nProcessPreprocessingSpinBox,
                # "parameter.nProcessSegmentationExport": self.ui.nProcessSegmentationExportSpinBox,
                # "parameter.disableTta": self.ui.disableTtaCheckBox
            }
        )
    

    @staticmethod
    def _createParameterNode() -> WidgetParameterNode:
        moduleName = "NeuroCTA"
        parameterNode = slicer.mrmlScene.GetSingletonNode(moduleName, "vtkMRMLScriptedModuleNode")
        if not parameterNode:
            parameterNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            parameterNode.SetName(slicer.mrmlScene.GenerateUniqueName(moduleName))

        parameterNode.SetAttribute("ModuleName", moduleName)
        return WidgetParameterNode(parameterNode)

    # Node handling
    def onInputTypeChanged(self) -> None:
        """Update the input selector label based on the selected input type."""

        if self.unsegmentedRadio.isChecked():  # Unsegmented Volume
            # self._stopSegmentPicking()
            self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            self.inputLabel.setText("Input volume:")
            self._setSectionState(self.segCollapsibleButton,   enabled=True,  collapsed=False)
            self._setSectionState(self.skelCollapsibleButton,  enabled=False, collapsed=True)

        else:  # Segmentation
            self.inputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
            self.inputLabel.setText("Input segmentation:")
            self._setSectionState(self.segCollapsibleButton,   enabled=False, collapsed=True)
            self._setSectionState(self.skelCollapsibleButton,  enabled=True,  collapsed=False)
        
        self.inputSelector.setMRMLScene(slicer.mrmlScene)

        # Force selection of first available node
        node = self.inputSelector.currentNode()
        if not node:
            node = slicer.mrmlScene.GetFirstNodeByClass(self.inputSelector.nodeTypes[0])
            if node:
                self.inputSelector.setCurrentNode(node)

    def onInputNodeChanged(self, node) -> None:
        self._stopSegmentPicking()

        # Hide previously shown nodes
        if hasattr(self, '_currentDisplayNode') and self._currentDisplayNode:
            self._currentDisplayNode.SetVisibility(False)
        if hasattr(self, '_currentVolumeRenderingDisplayNode') and self._currentVolumeRenderingDisplayNode:
            self._currentVolumeRenderingDisplayNode.SetVisibility(False)

        self._currentDisplayNode = None
        self._currentVolumeRenderingDisplayNode = None

        if node is None:
            return

        if node.IsA("vtkMRMLScalarVolumeNode"):
            self._segmentationNode = None
            volRenLogic = slicer.modules.volumerendering.logic()
            vrDisplayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(node)
            if not vrDisplayNode:
                volRenLogic.CreateDefaultVolumeRenderingNodes(node)
                vrDisplayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(node)
            vrDisplayNode.SetVisibility(True)
            self._currentVolumeRenderingDisplayNode = vrDisplayNode
            self.vesselTable.setVisible(False)

        elif node.IsA("vtkMRMLSegmentationNode"):
            self._segmentationNode = node
            if not node.GetDisplayNode():
                node.CreateDefaultDisplayNodes()
            if not node.GetSegmentation().ContainsRepresentation("Closed surface"):
                node.CreateClosedSurfaceRepresentation()
            displayNode = node.GetDisplayNode()
            displayNode.SetVisibility(True)
            displayNode.SetVisibility3D(True)
            displayNode.SetVisibility2DFill(True)
            segmentation = node.GetSegmentation()
            for i in range(segmentation.GetNumberOfSegments()):
                displayNode.SetSegmentVisibility(segmentation.GetNthSegmentID(i), True)
            self._currentDisplayNode = displayNode

            self.vesselTable.populateSegments(node)
            self.vesselTable.setVisible(True)

            # Start ray cast picking
            # self._startSegmentPicking(node)

        # Reset 3D view
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.rotateToViewAxis(3)  # look from anterior direction
          # reset the 3D view cube size and center it

        # Set camera angle
        renderWindow = threeDView.renderWindow()
        renderer = renderWindow.GetRenderers().GetFirstRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetPosition(-1300, 900, 1500)
        camera.SetFocalPoint(-25, 133, 1500)
        camera.SetViewUp(-0.0032600386718052606, -0.050379519790934586, 0.998724825031146)
        camera.Dolly(4)
        renderer.ResetCameraClippingRange()
        renderWindow.Render()
        threeDView.resetFocalPoint()

    def getCurrentVolumeNode(self):
        return self.inputSelector.currentNode()   

    def _onExportTable(self):
        self.vesselTable.export()     


    # Segmentation
    def onSegMethodChanged(self, button) -> None:
        method = button.text
        if method == "nnUNetv2":
            self.nnUNetOptionsGroupBox.setVisible(True)
        else:
            self.nnUNetOptionsGroupBox.setVisible(False)


    def onApplySegButton(self):
        """
        Run processing when user clicks "Run Segmentation" button.
        """

        # Get selected volume node
        volumeNode = self.inputSelector.currentNode()
        if volumeNode is None:
            slicer.util.errorDisplay("No volume selected.")
            return
        
        # Check segmentation method

        # TODO if nnunet:
        try:
            from SlicerNNUNetLib import SegmentationLogic, Parameter
        except ImportError:
            slicer.util.errorDisplay("Slicer nnU-Net extension is not installed.")
            return
        
        self._parameterNode.parameter.toSettings()
        self.logic.startNnUNetSegmentation(
            self.getCurrentVolumeNode(),
            parameter=self._parameterNode.params
        )

    # Classification

    def onRunClassInference(self):
        inputNode = self.inputSelector.currentNode()

        if inputNode is None:
            slicer.util.errorDisplay("No input selected.")
            return
        
        # load model lazily on first run
        if self.classLogic.model is None:
            path = self.classModelPathEdit.currentPath
            if not path:
                slicer.util.errorDisplay("No model path set.")
                return
            if self.sageRadio.isChecked():
                model = VesselSAGE(
                    in_channels=5,
                    hidden_channels=128,
                    num_classes=40,
                    n_layers=4,
                    dropout=0.3
                )
            else:
                model = VesselGNN(
                    in_channels=5,
                    edge_dim=6,
                    hidden_channels=128,
                    num_classes=40,
                    n_layers=4,
                    dropout=0.3
                )
            try:
                self.classLogic.loadModel(
                    path,
                    model
                )
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to load model: {e}")
                return

        # reload if path changed
        elif self.classModelPathEdit.currentPath != self.classLogic.loadedModelPath:
            path = self.classModelPathEdit.currentPath

            if self.sageRadio.isChecked():
                model = VesselSAGE(
                    in_channels=5,
                    hidden_channels=128,
                    num_classes=40,
                    n_layers=4,
                    dropout=0.3
                )
            else:
                model = VesselGNN(
                    in_channels=5,
                    edge_dim=6,
                    hidden_channels=128,
                    num_classes=40,
                    n_layers=4,
                    dropout=0.3
                )

            self.classLogic.loadModel(path, model)
        
        # load directly from file — bypasses Slicer's export geometry issues
        segFilePath = self.segFilePathEdit.currentPath
        if not segFilePath:
            slicer.util.errorDisplay("Please set segmentation file path.")
            return

        try:
            seg_vol, affine, zooms = self.classLogic.loadVolume(segFilePath)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load file: {e}")
            return

        binary_mask  = (seg_vol > 0).astype(np.uint8)


        modelType = "SAGE" if self.sageRadio.isChecked() else "GINE"

        # if modelType == "SAGE":
        #     voxelSpacing = [float(zooms[2]), float(zooms[1]), float(zooms[0])]
        # else:
        voxelSpacing = [float(zooms[0]), float(zooms[1]), float(zooms[2])]

        print("Shape:", binary_mask.shape)
        print("Zooms:", zooms)
        print("Spacing passed to inference:", voxelSpacing)
        print("Affine:\n", affine)

        ras_coords, preds = self.classLogic.runClassInference(
            binary_mask, None, voxelSpacing, affine,
            modelType
        )
        self._displayClassificationResults(ras_coords, preds)

    def _displayClassificationResults(self, ras_coords, preds):
        idx_to_label = {v: k for k, v in self.classLogic.LABEL_TO_IDX.items()}

        # fixed colormap — index matches LABEL_TO_IDX (0-indexed, label 1 = idx 0)
        COLORS = [
            (255,  0,182), (  0,159,255), (154, 77, 66), (  0,255,190),
            (120, 63,193), ( 31,150,152), (255,172,253), (177,204,113),
            (241,  8, 92), (254,143, 66), (221,  0,255), ( 77, 62,  2),
            (255,  0,  0), (  0,255,  0), (  2,173, 36), (  0,  0,255),
            (255,255,  0), (  0,255,255), (255,  0,255), (255,239,213),
            (  0,  0,205), (205,133, 63), (210,180,140), (102,205,170),
            (  0,  0,128), (  0,139,139), ( 46,139, 87), (255,228,225),
            (106, 90,205), (221,160,221), (233,150,122), (165, 42, 42),
            (255,250,250), (147,112,219), (218,112,214), ( 75,  0,130),
            (255,182,193), ( 60,179,113), (255,235,205), (255,228,196),
        ]
        # normalize to 0-1
        colors = {i: tuple(c/255.0 for c in COLORS[i]) for i in range(len(COLORS))}

        shNode     = slicer.mrmlScene.GetSubjectHierarchyNode()
        rootFolder = shNode.CreateFolderItem(shNode.GetSceneItemID(), "Classification")

        for idx in np.unique(preds):
            if idx < 0:
                continue
            mask      = preds == idx
            name      = idx_to_label.get(idx, f"cls_{idx}")
            class_pts = ras_coords[mask]
            color     = colors.get(idx, (1.0, 1.0, 1.0))

            pts   = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            for i, pt in enumerate(class_pts):
                pts.InsertNextPoint(*pt.tolist())
                cells.InsertNextCell(1)
                cells.InsertCellPoint(i)

            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetVerts(cells)

            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Class_{name}")
            node.SetAndObserveMesh(pd)
            node.CreateDefaultDisplayNodes()
            dn = node.GetDisplayNode()
            dn.SetColor(*color)
            dn.SetPointSize(5)
            dn.SetVisibility(True)
            shNode.SetItemParent(shNode.GetItemByDataNode(node), rootFolder)

        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()

    # Skeletonization
    def onSkelMethodChanged(self, button) -> None:
        pass

    def onApplySkelButton(self) -> None:
        """
        Run processing when user clicks "Run Skeletonization" button.
        """
        self._currentSkelMethod = self.skelMethodGroup.checkedButton().text

        inputNode = self.inputSelector.currentNode()
        if inputNode is None:
            slicer.util.errorDisplay("No input selected.")
            return
        
        if self._hasExistingResults(inputNode):
            if not slicer.util.confirmYesNoDisplay(
                f"Results already exist for {inputNode.GetName()}. Re-run and overwrite?"
            ):
                return

        self.runSkelPushButton.setText("Running... 0%")
        self.runSkelPushButton.setEnabled(False)

        method = self.skelMethodGroup.checkedButton().text        
        if method == "VMTK Extract Centerline":
            self.skelLogic.startVMTKCenterlineExtraction(inputNode)
        elif method == "Medial axis thinning":
            self._runMedialAxisSkel(inputNode)

    def _runMedialAxisSkel(self, inputNode):
        labelmapNode = None
        volumeArray = None

        if self.unsegmentedRadio.isChecked():
            volumeArray = slicer.util.arrayFromVolume(inputNode)
            referenceNode = inputNode
            print(f"Using scalar volume: {inputNode.GetName()}")
        else:
            name = f"{inputNode.GetName()}"
            labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                inputNode, labelmapNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
            )

        volumeArray = slicer.util.arrayFromVolume(labelmapNode)
        referenceNode = labelmapNode
        print(f"Using segmentation: {inputNode.GetName()}")

        self._ijkToRas = vtk.vtkMatrix4x4()
        referenceNode.GetIJKToRASMatrix(self._ijkToRas)
        segmentation = inputNode.GetSegmentation()

        # extract spacing here, before creating the logic
        voxelSpacing = [
            np.linalg.norm([self._ijkToRas.GetElement(r, 0) for r in range(3)]),
            np.linalg.norm([self._ijkToRas.GetElement(r, 1) for r in range(3)]),
            np.linalg.norm([self._ijkToRas.GetElement(r, 2) for r in range(3)]),
        ]

        # set up logic
        self._setButtonProgress(self.runSkelPushButton, 0, 1, "Skeletonizing...")
        self.skelLogic.startMedialAxisSkel(volumeArray, segmentation, self._ijkToRas, voxelSpacing)

        # Delete temp labelmap node
        if labelmapNode:
            slicer.mrmlScene.RemoveNode(labelmapNode)


    def onSkelProgress(self, current, total):
        self._setButtonProgress(self.runSkelPushButton, current, total, "Skeletonizing...")
        
    def onSkelFinished(self):
        try:
            if self._currentSkelMethod == "VMTK Extract Centerline":
                skeletons = self.skelLogic.loadResults(self._segmentationNode)
            else:
                skeletons = self.skelLogic.loadResults(self._segmentationNode, self._ijkToRas)
        except RuntimeError as e:
            slicer.util.errorDisplay(str(e))
            return
        finally:
            self._resetButton(self.runSkelPushButton, "▶︎ Run Skeletonization")

        self.vesselTable.setSkeletons(skeletons)
        self.vesselTable.removeCLColumns()
        self.vesselTable.populateCLColumn()
        self.vesselTable.populateFeatureColumns()
        self.showBranchingPointsCheckBox.setEnabled(True)

    def _hasExistingResults(self, segmentationNode) -> bool:
        ref = segmentationNode.GetNodeReferenceID("SkeletonFolder")
        if not ref:
            return False
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        itemId = int(ref)
        return itemId != 0 and shNode.GetItemName(itemId) != ""

    def onSkelError(self, errorMsg):
        print(f"Error: {errorMsg} ")
        # if self.isStopping:
        #     return

        # self._setApplyVisible(True)
        # if isinstance(errorMsg, Exception):
        #     errorMsg = str(errorMsg)
        # self._reportError("Encountered error during inference :\n" + errorMsg, doTraceback=False)

    # Segment picking
    def _startSegmentPicking(self, segmentationNode):
        self._stopSegmentPicking()

        self._pickingSegmentationNode = segmentationNode

        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        self._pickingInteractor = threeDWidget.threeDView().interactor()

        self._pickingObserverTag = self._pickingInteractor.AddObserver(
            vtk.vtkCommand.LeftButtonPressEvent,
            self._onPickClick,
            1.0
        )

    def _stopSegmentPicking(self):
        if self._tooltipLabel:
            self._tooltipLabel.hide()
            
        if hasattr(self, '_pickingInteractor') and self._pickingInteractor and \
        hasattr(self, '_pickingObserverTag') and self._pickingObserverTag:
            self._pickingInteractor.RemoveObserver(self._pickingObserverTag)
        self._pickingInteractor = None
        self._pickingObserverTag = None
        self._pickingSegmentationNode = None

    def _onPickClick(self, caller, event):
        if self._tooltipLabel:
            self._tooltipLabel.hide()

        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()

        # Get click position in display coordinates
        interactor = threeDView.interactor()
        x, y = interactor.GetEventPosition()

        # Use hardware picker to find the picked cell
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        renderWindow = threeDView.renderWindow()
        renderer = renderWindow.GetRenderers().GetFirstRenderer()
        picker.Pick(x, y, 0, renderer)

        pickedPosition = picker.GetPickPosition()

        result = picker.Pick(x, y, 0, renderer)
        print(f"Picker result: {result}, cellId: {picker.GetCellId()}, position: {picker.GetPickPosition()}")


        # Find which segment contains this RAS point
        segmentation = self._pickingSegmentationNode.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            segmentId = segmentation.GetNthSegmentID(i)
            segmentName = segmentation.GetNthSegment(i).GetName()

            polyData = vtk.vtkPolyData()
            self._pickingSegmentationNode.GetClosedSurfaceRepresentation(segmentId, polyData)

            if polyData.GetNumberOfPoints() == 0:
                continue

            # Check if picked position is close to this segment's surface
            pointLocator = vtk.vtkPointLocator()
            pointLocator.SetDataSet(polyData)
            pointLocator.BuildLocator()
            closestPointId = pointLocator.FindClosestPoint(pickedPosition)
            closestPoint = polyData.GetPoint(closestPointId)

            dist = vtk.vtkMath.Distance2BetweenPoints(pickedPosition, closestPoint)
            if dist < 10:  # threshold in mm², adjust as needed
                print(f"Clicked segment: {segmentName} (ID: {segmentId})")

                self._showPickLabel(segmentName, threeDView, x, y)

                slicer.util.showStatusMessage(f"Segment: {segmentName}", 3000)
                return
            
        print("No segment matched distance threshold")

    def _showPickLabel(self, text, threeDView, x, y):
        if self._tooltipLabel is None:
            self._tooltipLabel = qt.QLabel(threeDView)
            self._tooltipLabel.setStyleSheet("""
                QLabel {
                    background-color: #cce7ff;
                    border: 1px solid #888;
                    padding: 3px 6px;
                    font-size: 20px;
                    color: #000;
                }
            """)
            self._tooltipLabel.setWindowFlags(qt.Qt.SubWindow)
            self._tooltipTimer = qt.QTimer()
            self._tooltipTimer.setSingleShot(True)
            self._tooltipTimer.connect("timeout()", self._tooltipLabel.hide)

        self._tooltipLabel.setText(text)
        self._tooltipLabel.adjustSize()

        flippedY = threeDView.height - y
        self._tooltipLabel.move(x + 10, flippedY - 20)
        self._tooltipLabel.show()
        self._tooltipLabel.raise_()

        self._tooltipTimer.start(1000)

    
    # Buttons & UI Helpers
    def onToggleBranchingPoints(self, checked: bool) -> None:
        if not hasattr(self.vesselTable, '_skeletonsBySegment'):
            return
        for segmentName, data in self.vesselTable._skeletonsBySegment.items():
            # medial axis nodes by name
            for prefix in ["BP_", "EP_"]:
                node = slicer.mrmlScene.GetFirstNodeByName(f"{prefix}{segmentName}")
                if node and node.GetDisplayNode():
                    node.GetDisplayNode().SetVisibility(checked)
            # VMTK endpoint node by ID
            ep_node_id = data.get("ep_node_id")
            if ep_node_id:
                node = slicer.mrmlScene.GetNodeByID(ep_node_id)
                if node and node.GetDisplayNode():
                    node.GetDisplayNode().SetVisibility(checked)

    def _setSectionState(self, collapsibleButton, enabled, collapsed):
        collapsibleButton.setEnabled(enabled)
        collapsibleButton.collapsed = collapsed

    def _setButtonProgress(self, button, value, maximum, text="Running..."):
        pct = int(value / maximum * 100) if maximum > 0 else 0
        button.setText(f"{text} {pct}%")
        button.setStyleSheet(f"""
            QPushButton {{
                text-align: center;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90d9,
                    stop:{pct/100:.3f} #4a90d9,
                    stop:{min(pct/100 + 0.001, 1.0):.3f} #e8e8e8,
                    stop:1 #e8e8e8
                );
                border: 1px solid #aaa;
            }}
        """)
        button.setEnabled(False)
        slicer.app.processEvents()


    def _resetButton(self, button, originalText):
        button.setText(originalText)
        button.setStyleSheet("")
        button.setEnabled(True)

    # Progress and Logging
    def onProgressInfo(self, infoMsg):
        print(infoMsg)
        # self.ui.logTextEdit.insertPlainText(self._formatMsg(infoMsg) + "\n")
        # self.moveTextEditToEnd(self.ui.logTextEdit)
        # slicer.app.processEvents()

    @staticmethod
    def _formatMsg(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines()])

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)
