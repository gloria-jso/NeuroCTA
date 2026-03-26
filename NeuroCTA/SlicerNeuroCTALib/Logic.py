import os
import sys
from pathlib import Path
from typing import Protocol, List, Optional, Callable
from SlicerNNUNetLib import SegmentationLogic, Parameter

import qt
import slicer
import vtk
import numpy as np

from .Parameter import Parameter
from .Signal import Signal
import time
from ExtractCenterline import ExtractCenterlineLogic
from skimage.morphology import skeletonize



class LogicProtocol(Protocol):
    """
    Segmentation interface class.
    This class is used as interface which is expected by the NNUnet Widget.
    Any instance implementing this interface is compatible with the NNUnet Widget.
    """
    inferenceFinished: Signal
    errorOccurred: Signal
    progressInfo: Signal

    def setParameter(self, nnUNetParam: Parameter) -> None:
        pass

    def startSegmentation(
            self,
            volumeNode: "slicer.vtkMRMLScalarVolumeNode"
    ) -> None:
        pass

    def stopSegmentation(self):
        pass

    def waitForSegmentationFinished(self):
        pass

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        pass


class ProcessProtocol(Protocol):
    """
    Interface for NNUnet process runner.
    Process is responsible for running the NNUnet prediction and reporting progress / errors.
    """

    errorOccurred: Signal
    finished: Signal
    readInfo: Signal

    def start(self, program: str, args: List[str]) -> None:
        pass

    def stop(self) -> None:
        pass

    def waitForFinished(self) -> None:
        pass


class Logic:
    r"""
    Segmentation logic for nnUNet based segmentations.

    This class is responsible for writing a volume file with correct nnUNet formatting, calling the nnUNet detection
    on the given folder with user inputs and user model and allowing to load the generated segmentation afterwards.

    At the start of the nnUNet detection, a temporary folder is created with the selected input volume.
    The nnUNet detection QProcess is called with user parameters.

    Once the QProcess processing is done (either successfully or not), the Segmentation Logic triggers
    the inferenceFinished call.

    During execution, the standard output is returned using the progressInfo signal. If any error occurs, the
    errorOccurred sigal is used.

    The loadSegmentation method can be called to load the segmentation results. If the segmentation failed, this method
    call will raise a RuntimeError Exception.

    Usage example :
    >>> from SlicerNNUNetLib import SegmentationLogic, Parameter
    >>>
    >>> # Create instance
    >>> logic = SegmentationLogic()
    >>>
    >>> # Connect progress and error logging
    >>> logic.progressInfo.connect(print)
    >>> logic.errorOccurred.connect(slicer.util.errorDisplay)
    >>>
    >>> # Connect processing done signal
    >>> logic.inferenceFinished.connect(logic.loadSegmentation)
    >>>
    >>> # Start segmentation on given volume node on 5 folds
    >>> param = Parameter(modelPath=Path(r"C:\<PATH_TO>\NNUnetModel\Dataset123_456MRI"), folds = "0,1,2,3,4")
    >>> import SampleData
    >>> volumeNode = SampleData.downloadSample("MRHead")
    >>> logic.setParameter(param)
    >>> logic.startSegmentation(volumeNode)
    """

    def __init__(self, process: Optional[ProcessProtocol] = None):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")

        self.inferenceProcess = process or Process(qt.QProcess.MergedChannels)
        self.inferenceProcess.finished.connect(self.inferenceFinished)
        self.inferenceProcess.errorOccurred.connect(self.errorOccurred)
        self.inferenceProcess.readInfo.connect(self.progressInfo)

        self._nnUNet_predict_path = None
        self._nnUNetParam: Optional[Parameter] = None
        self._tmpDir = qt.QTemporaryDir()

    def __del__(self):
        self.stopSegmentation()

    def setParameter(self, nnUnetConf: Parameter):
        self._nnUNetParam = nnUnetConf

    def startSegmentation(self, volumeNode: "slicer.vtkMRMLScalarVolumeNode") -> None:
        """Run the segmentation on a slicer volumeNode, get the result as a segmentationNode"""


        # Check the nnUNet parameters are correct
        try:
            self._getNNUNetParamArgsOrRaise()
        except RuntimeError as e:
            self.errorOccurred(e)
            return

        # Prepare the inference directory
        if not self._prepareInferenceDir(volumeNode):
            self.errorOccurred(f"Failed to export volume node to {self.nnUNetInDir}")
            return

        # Launch the nnUNet processing
        self._startInferenceProcess()

    def stopSegmentation(self):
        self.inferenceProcess.stop()

    def waitForSegmentationFinished(self):
        self.inferenceProcess.waitForFinished()

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        try:
            segmentationNode = slicer.util.loadSegmentation(self._outFile)
            self._renameSegments(segmentationNode)
            return segmentationNode
        except StopIteration:
            raise RuntimeError(
                "Failed to load the segmentation.\n"
                "Something went wrong during the nnUNet processing.\n"
                "Please check the logs for potential errors and contact the library maintainers."
            )

    def _renameSegments(self, segmentationNode: "slicer.vtkMRMLSegmentationNode") -> None:
        """
        Rename loaded segments with dataset file labels dict.
        """
        labels = self._nnUNetParam.readSegmentIdsAndLabelsFromDatasetFile()
        if labels is None:
            return

        for segmentId, label in labels:
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            if segment is None:
                continue
            segment.SetName(label)

    @staticmethod
    def _nnUNetPythonDir():
        return Path(sys.executable).parent.joinpath("..", "lib", "Python")

    @classmethod
    def _findUNetPredictPath(cls):
        # nnUNet install dir depends on OS. For Windows, install will be done in the Scripts dir.
        # For Linux and MacOS, install will be done in the bin folder.
        nnUNetPaths = ["Scripts", "bin"]
        for path in nnUNetPaths:
            predict_paths = list(sorted(cls._nnUNetPythonDir().joinpath(path).glob("nnUNetv2_predict*")))
            if predict_paths:
                return predict_paths[0].resolve()

        return None

    def _startInferenceProcess(self):
        """
        Run the nnU-Net V2 inference script
        """
        # Check the nnUNet predict script is correct
        nnUnetPredictPath = self._findUNetPredictPath()
        if not nnUnetPredictPath:
            self.errorOccurred("Failed to find nnUNet predict path.")
            return

        # Get the nnUNet parameters as arg list
        try:
            args = self._getNNUNetParamArgsOrRaise()
        except RuntimeError as e:
            self.errorOccurred(e)
            return

        # setup environment variables
        # not needed, just needs to be an existing directory
        os.environ['nnUNet_preprocessed'] = self._nnUNetParam.modelFolder.as_posix()

        # not needed, just needs to be an existing directory
        os.environ['nnUNet_raw'] = self._nnUNetParam.modelFolder.as_posix()
        os.environ['nnUNet_results'] = self._nnUNetParam.modelFolder.as_posix()

        argListStr = ' '.join(str(a) for a in args)
        self.progressInfo(
            "Starting nnUNet with the following parameters:\n"
            f"\n{nnUnetPredictPath} {argListStr}\n\n"
            "JSON parameters :\n"
            f"{self._nnUNetParam.debugString()}\n"
        )
        self.progressInfo("nnUNet preprocessing...\n")
        self.inferenceProcess.start(nnUnetPredictPath, args, qt.QProcess.Unbuffered | qt.QProcess.ReadOnly)

    def _getNNUNetParamArgsOrRaise(self):
        return self._nnUNetParam.asArgList(self.nnUNetInDir, self.nnUNetOutDir)

    @property
    def _outFile(self) -> str:
        return next(file for file in self.nnUNetOutDir.rglob(f"*{self._fileEnding}")).as_posix()

    def _prepareInferenceDir(self, volumeNode) -> bool:
        self._tmpDir.remove()
        self.nnUNetOutDir.mkdir(parents=True)
        self.nnUNetInDir.mkdir(parents=True)

        # Name of the volume should match expected nnUNet conventions
        self.progressInfo(f"Transferring volume to nnUNet in {self._tmpDir.path()}\n")
        volumePath = self.nnUNetInDir.joinpath(f"volume_0000{self._fileEnding}")
        slicer.util.exportNode(volumeNode, volumePath)
        return volumePath.exists()

    @property
    def _fileEnding(self):
        return self._nnUNetParam.readFileEndingFromDatasetFile() if self._nnUNetParam else ".nii.gz"

    @property
    def nnUNetOutDir(self):
        return Path(self._tmpDir.path()).joinpath("output")

    @property
    def nnUNetInDir(self):
        return Path(self._tmpDir.path()).joinpath("input")
    


    def startNnUNetSegmentation(self, volumeNode: "slicer.vtkMRMLScalarVolumeNode"):
        print("I'm in startNnUNetSegmentation")

        # --- Create nnU-Net logic ---
        logic = SegmentationLogic()

        # Log progress to console
        # logic.progressInfo.connect(lambda txt: print(txt))
        logic.errorOccurred.connect(lambda err: slicer.util.errorDisplay(str(err)))

        # Log to the button bar too
        def onProgressInfo(txt):
            print(txt)
            import re
            match = re.search(r'(\d+)%', txt)
            if match:
                pct = int(match.group(1))
                self._setButtonProgress(self.runSegPushButton, pct, 100, f"Segmenting... {pct}%")

        logic.progressInfo.connect(onProgressInfo)

        # Load segmentation automatically when done
        def onFinished(arg):
            try:
                segNode = logic.loadSegmentation()
                slicer.util.setSliceViewerLayers(label=segNode)
                print("Segmentation loaded successfully.")
                self._resetButton(self.runSegPushButton, "▶︎ Run Segmentation")
                self._setSectionState(self.segCollapsibleButton,  enabled=True, collapsed=True)
                self._setSectionState(self.skelCollapsibleButton, enabled=True, collapsed=False)
            except Exception as e:
                slicer.util.errorDisplay(str(e))

        logic.inferenceFinished.connect(onFinished)

        modelFolder = Path(
            self.modelPathEdit.currentPath
        )

        param = Parameter(
            modelPath=modelFolder,
            folds="0",  # or "0,1,2,3,4"
            checkPointName=self.modelCheckpointEdit.text,
        )

        logic.setParameter(param)
        self._setButtonProgress(self.runSegPushButton, 0, 100, "Segmenting... 0%")
        logic.startSegmentation(volumeNode)

        print("nnU-Net inference started...")

    def startSkeletonization(
            self,
            inputNode,
            method
        ):       
        if method == "VMTK Extract Centerline":
            self._runVMTKCenterlineExtraction(inputNode)
        elif method == "Medial axis thinning":
            self._runVoxelSkeletonization(inputNode)


class Process:
    """
    Convenience wrapper around QProcess run.
    Forwards the process read / error events when read.
    Kills process on stop if process is running.
    """

    def __init__(self, channelMode: qt.QProcess.ProcessChannelMode):
        self.errorOccurred = Signal("str")
        self.finished = Signal()
        self.readInfo = Signal("str")

        self.process = qt.QProcess()
        self.process.setProcessChannelMode(channelMode)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self._onErrorOccurred)
        self.process.readyRead.connect(self._onReadyRead)

    def stop(self):
        if self.process.state() == self.process.Running:
            self.readInfo("Killing process.")
            self.process.kill()

    def start(self, program, args, openMode: qt.QIODevice.OpenMode):
        self.stop()
        self.process.start(program, args, openMode)

    def waitForFinished(self, timeOut_ms: Optional[int] = None):
        self.process.waitForFinished(timeOut_ms if timeOut_ms is not None else -1)

    def _onReadyRead(self):
        self._report(self.process.readAll(), self.readInfo)

    def _onErrorOccurred(self, *_):
        self._report(self.process.readAllStandardError(), self.errorOccurred)

    @staticmethod
    def _report(stream: "qt.QByteArray", outSignal: Callable[[str], None]) -> None:
        info = qt.QTextCodec.codecForUtfText(stream).toUnicode(stream)
        if info:
            outSignal(info)


# SkeletonizationLogic.py
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional

import qt
import slicer
import vtk

from .Signal import Signal


class SkeletonizationLogic:
    """
    Segmentation logic for voxel skeletonization.
    Mirrors the nnUNet SegmentationLogic pattern — serializes volume data
    to a temp directory, runs skeletonize_worker.py as a QProcess subprocess,
    then deserializes results back into Slicer model nodes.

    Usage:
    >>> logic = SkeletonizationLogic()
    >>> logic.progressInfo.connect(print)
    >>> logic.errorOccurred.connect(slicer.util.errorDisplay)
    >>> logic.finished.connect(lambda: logic.loadResults(segmentation, ijkToRas))
    >>> logic.startSkeletonization(volumeArray, segmentation, ijkToRas)
    """

    def __init__(self, process: Optional[ProcessProtocol]=None):
        self.skelFinished      = Signal("int", "int")
        self.errorOccurred = Signal("str")
        self.progressInfo  = Signal("str")
        self.progressUpdated = Signal("int", "int")

        self.skeletonizationProcess = process or Process(qt.QProcess.MergedChannels)
        self.skeletonizationProcess.finished.connect(lambda *args: self.skelFinished())
        self.skeletonizationProcess.errorOccurred.connect(self.errorOccurred)
        self.skeletonizationProcess.readInfo.connect(self._onProgressInfo)


        self._tmpDir = qt.QTemporaryDir()
        self._ijkToRas = None

    def __del__(self):
        self.stop()

    def stop(self):
        self.skeletonizationProcess.stop()

    def waitForFinished(self):
        self.skeletonizationProcess.waitForFinished()

    def _onProgressInfo(self, msg: str):
        for line in msg.strip().splitlines():
            if line.startswith("PROGRESS:"):
                parts = line.split("\t")
                progress, readable = parts[0], parts[1] if len(parts) > 1 else ""
                tokens = progress.split(":")
                if len(tokens) == 3:
                    _, current, total = tokens
                    self.progressUpdated(int(current), int(total))
                if readable:
                    self.progressInfo(readable)  # logs cleanly
            else:
                self.progressInfo(line)


    def startSkeletonization(
        self,
        volumeArray: np.ndarray,
        segmentation: "slicer.vtkSegmentation",
        ijkToRas: "vtk.vtkMatrix4x4",
        voxelSpacing
    ) -> None:
        self._ijkToRas = ijkToRas  # stash for loadResults
        self._voxelSpacing = voxelSpacing


        if not self._prepareWorkDir(volumeArray, segmentation):
            self.errorOccurred("Failed to write inputs to temp directory.")
            return

        self._startSkelProcess()

    def loadResults(
        self,
        segmentation: "slicer.vtkSegmentation",
        ijkToRas: "vtk.vtkMatrix4x4"
    ) -> dict:
        """
        Called after finished signal fires.
        Reads worker output and creates vtkMRMLModelNodes.
        Returns dict of {segmentName: modelNode}.
        """
        output_path = Path(self._tmpDir.path()) / "results.json"
        if not output_path.exists():
            raise RuntimeError(
                "Worker output not found. Check logs for errors."
            )

        with open(output_path) as f:
            results = json.load(f)

        m = np.array([
            [ijkToRas.GetElement(r, c) for c in range(4)]
            for r in range(4)
        ])

        skeletonsBySegment = {}

        for segmentName, data in results.items():
            coords = np.array(data["coords"])   # (N, 3) IJK
            r, g, b = data["color"]

            if len(coords) == 0:
                continue

            # IJK -> RAS
            ijk_h  = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0], np.ones(len(coords))])
            ras_h  = (m @ ijk_h.T).T
            ras_pts = ras_h[:, :3]

            # Build polydata
            points = vtk.vtkPoints()
            points.SetData(vtk.util.numpy_support.numpy_to_vtk(ras_pts, deep=True))

            nPts  = points.GetNumberOfPoints()
            cells = np.hstack([
                np.ones((nPts, 1), dtype=np.int64),
                np.arange(nPts, dtype=np.int64).reshape(-1, 1)
            ])
            verts = vtk.vtkCellArray()
            verts.SetCells(nPts, vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells.ravel()))

            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetVerts(verts)

            # Create / replace model node — safe, we're back on main thread
            existing = slicer.mrmlScene.GetFirstNodeByName(f"Skeleton_{segmentName}")
            if existing:
                slicer.mrmlScene.RemoveNode(existing)

            modelNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", f"Skeleton_{segmentName}"
            )
            modelNode.SetAndObserveMesh(polyData)
            modelNode.CreateDefaultDisplayNodes()

            displayNode = modelNode.GetDisplayNode()
            displayNode.SetColor(r, g, b)
            displayNode.SetPointSize(3)
            displayNode.SetLineWidth(3)
            displayNode.SetVisibility(True)

            skeletonsBySegment[segmentName] = {
                "modelNode": modelNode,
                "features":  data.get("features", {})
            }

            # Add branch points
            bp_coords = data.get("features", {}).get("branch_point_coords", [])
            if bp_coords:
                bp_coords = np.array(bp_coords)

                # IJK -> RAS
                ijk_h  = np.column_stack([bp_coords[:, 2], bp_coords[:, 1], bp_coords[:, 0], np.ones(len(bp_coords))])
                ras_pts = (m @ ijk_h.T).T[:, :3]

                existing = slicer.mrmlScene.GetFirstNodeByName(f"BranchPoints_{segmentName}")
                if existing:
                    slicer.mrmlScene.RemoveNode(existing)

                bpNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"BranchPoints_{segmentName}")
                bpNode.CreateDefaultDisplayNodes()
                bpNode.GetDisplayNode().SetGlyphScale(1.0)
                bpNode.GetDisplayNode().SetSelectedColor(r, g, b)
                bpNode.GetDisplayNode().SetColor(r, g, b)

                for pt in ras_pts:
                    bpNode.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))

        return skeletonsBySegment

    # ── Private ────────────────────────────────────────────────────────────────

    def _prepareWorkDir(self, volumeArray, segmentation) -> bool:
        """Serialize volume array + label metadata to the temp directory."""
        tmp = Path(self._tmpDir.path())

        np.save(tmp / "volume.npy", volumeArray)

        uniqueLabels = np.unique(volumeArray)
        uniqueLabels = uniqueLabels[uniqueLabels > 0]

        labels_info = []
        for label in uniqueLabels:
            segmentId = segmentation.GetNthSegmentID(int(label) - 1)
            segment   = segmentation.GetSegment(segmentId)
            r, g, b   = segment.GetColor()
            labels_info.append({
                "label": int(label),
                "name":  segment.GetName(),
                "color": [r, g, b]
            })

        with open(tmp / "labels.json", "w") as f:
            json.dump(labels_info, f)

        return (tmp / "volume.npy").exists()

    def _startSkelProcess(self):
        tmp         = Path(self._tmpDir.path())
        worker_path = Path(__file__).parent / "skeletonize_worker.py"

        if not worker_path.exists():
            self.errorOccurred(f"Worker script not found at {worker_path}")
            return

        args_dict = {
            "volume_path": str(tmp / "volume.npy"),
            "labels_path": str(tmp / "labels.json"),
            "output_path": str(tmp / "results.json"),
            "voxel_spacing": self._voxelSpacing,
        }

        self.skeletonizationProcess.start(
            sys.executable,
            [str(worker_path), json.dumps(args_dict)],
            qt.QProcess.Unbuffered | qt.QProcess.ReadOnly
        )