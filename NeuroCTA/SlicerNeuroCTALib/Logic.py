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

    def loadResults(self, segmentationNode, ijkToRas):
        output_path = Path(self._tmpDir.path()) / "results.json"
        if not output_path.exists():
            raise RuntimeError("Worker output not found. Check logs for errors.")

        with open(output_path) as f:
            results = json.load(f)

        m = np.array([
            [ijkToRas.GetElement(r, c) for c in range(4)]
            for r in range(4)
        ])

        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        # --- Clean up old folder if exists ---
        existingRef = segmentationNode.GetNodeReferenceID("SkeletonFolder")
        if existingRef:
            existingItem = shNode.GetItemByUID(slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyUIDByName("SkeletonFolder"), existingRef)
            oldItem = int(existingRef) if existingRef.isdigit() else 0
            if oldItem and shNode.GetItem(oldItem):
                shNode.RemoveItem(oldItem)

        # --- Create fresh folder hierarchy ---
        segNodeName = segmentationNode.GetName()
        rootFolderItemId = shNode.CreateFolderItem(
            shNode.GetSceneItemID(), f"{segNodeName}"
        )
        segmentationNode.AddNodeReferenceID("SkeletonFolder", str(rootFolderItemId))

        skelFolder = shNode.CreateFolderItem(rootFolderItemId, "Skeletons")
        bpFolder   = shNode.CreateFolderItem(rootFolderItemId, "Branch Points")

        skeletonsBySegment = {}

        for segmentName, data in results.items():
            coords = np.array(data["coords"])
            r, g, b = data["color"]

            if len(coords) == 0:
                continue

            # IJK -> RAS
            ijk_h   = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0], np.ones(len(coords))])
            ras_pts = (m @ ijk_h.T).T[:, :3]

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

            # Remove old skeleton node if exists
            existing = slicer.mrmlScene.GetFirstNodeByName(f"Skeleton_{segmentName}")
            if existing:
                slicer.mrmlScene.RemoveNode(existing)

            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Skeleton_{segmentName}")
            modelNode.SetAndObserveMesh(polyData)
            modelNode.CreateDefaultDisplayNodes()
            shNode.SetItemParent(shNode.GetItemByDataNode(modelNode), skelFolder)

            displayNode = modelNode.GetDisplayNode()
            displayNode.SetColor(r, g, b)
            displayNode.SetPointSize(3)
            displayNode.SetLineWidth(3)
            displayNode.SetVisibility(True)

            skeletonsBySegment[segmentName] = {
                "modelNode": modelNode,
                "features":  data.get("features", {})
            }

            # --- Branch points ---
            bp_coords = data.get("features", {}).get("branch_point_coords", [])
            if bp_coords:
                bp_coords = np.array(bp_coords)
                ijk_h   = np.column_stack([bp_coords[:, 2], bp_coords[:, 1], bp_coords[:, 0], np.ones(len(bp_coords))])
                ras_pts = (m @ ijk_h.T).T[:, :3]

                existing = slicer.mrmlScene.GetFirstNodeByName(f"BP_{segmentName}")
                if existing:
                    slicer.mrmlScene.RemoveNode(existing)

                bpNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"BP_{segmentName}")
                bpNode.CreateDefaultDisplayNodes()
                bpNode.GetDisplayNode().SetGlyphScale(1.0)
                bpNode.GetDisplayNode().SetSelectedColor(r, g, b)
                bpNode.GetDisplayNode().SetColor(r, g, b)
                shNode.SetItemParent(shNode.GetItemByDataNode(bpNode), bpFolder)

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


# Classification Logic
import torch
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt

class ClassificationLogic:
    def __init__(self):
        self.model = None
        self.device = torch.device("cpu")
        self.LABEL_MAP = {
            0: "background", 1: "BA", 2: "R-P1P2", 3: "L-P1P2",
            4: "R-ICA", 5: "R-M1", 6: "L-ICA", 7: "L-M1",
            8: "R-Pcom", 9: "L-Pcom", 10: "Acom", 11: "R-A1A2",
            12: "L-A1A2", 13: "R-A3", 14: "L-A3", 15: "3rd-A2",
            16: "3rd-A3", 17: "R-M2", 18: "R-M3", 19: "L-M2",
            20: "L-M3", 21: "R-P3P4", 22: "L-P3P4", 23: "R-VA",
            24: "L-VA", 25: "R-SCA", 26: "L-SCA", 27: "R-AICA",
            28: "L-AICA", 29: "R-PICA", 30: "L-PICA", 31: "R-AChA",
            32: "L-AChA", 33: "R-OA", 34: "L-OA", 35: "VoG",
            36: "StS", 37: "ICVs", 38: "R-BVR", 39: "L-BVR", 40: "SSS"
        }
        self.loadedModelPath = None

        # exclude background
        self.ARTERY_NAMES = [v for k, v in self.LABEL_MAP.items() if k > 0]
        self.LABEL_TO_IDX = {name: i for i, name in enumerate(self.ARTERY_NAMES)}
        self.NUM_CLASSES = len(self.ARTERY_NAMES)

    def loadModel(self, modelPath, modelInstance):
        modelInstance.load_state_dict(
            torch.load(modelPath, map_location=self.device, weights_only=False)
        )
        self.model = modelInstance
        self.model.eval()
        self.loadedModelPath = modelPath

    def runClassInference(self, binary_mask, _, voxel_spacing, affine, modelType):

        # ── Skeleton + distance map ───────────────────────────────────────────────
        skel = skeletonize(binary_mask)

        dist = distance_transform_edt(
            binary_mask,
            sampling=voxel_spacing  # IMPORTANT: spacing applied here
        )

        skel_obj = Skeleton(skel, spacing=voxel_spacing)
        stats = summarize(skel_obj, separator='-')

        coords = skel_obj.coordinates  
        degrees = skel_obj.degrees

        # ── Node features ─────────────────────────────────────────────────────────
        shape = np.array(binary_mask.shape, dtype=float)
        norm_coords = coords / shape

        node_radii = dist[coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)]
        degrees_norm = degrees / (degrees.max() + 1e-6)

        node_feats = np.column_stack([norm_coords, degrees_norm, node_radii])  # (N, 5)

        # ── Edge features ─────────────────────────────────────────────────────────
        edge_src, edge_dst, edge_feats = [], [], []

        for row_idx, row in stats.iterrows():
            src = int(row["node-id-src"])
            dst = int(row["node-id-dst"])

            path = skel_obj.path_coordinates(int(row_idx)).astype(int)

            radii = dist[path[:, 0], path[:, 1], path[:, 2]]

            euc = max(float(row["euclidean-distance"]), 1e-6)

            feats = [
                float(row["branch-distance"]),
                float(row["branch-distance"]) / euc,
                float(radii.mean()),
                float(radii.min()),
                float(radii.max()),
                float(radii.std()),
            ]

            # undirected graph (duplicate edges)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_src.append(dst)
            edge_dst.append(src)

            edge_feats.append(feats)
            edge_feats.append(feats)

        x = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

        print(f"[Slicer] Graph: {x.shape[0]} nodes, {len(edge_src)//2} edges")


        # ── Inference ─────────────────────────────────────────────────────────────
        model = self.model
        model.eval()

        with torch.no_grad():
            if modelType == "SAGE":
                out = model(x, edge_index)
            else:
                edge_attr = np.array(edge_feats, dtype=np.float32)
                mean = edge_attr.mean(axis=0)
                std = edge_attr.std(axis=0) + 1e-6
                edge_attr = torch.tensor((edge_attr - mean) / std, dtype=torch.float)
                
                out = model(x, edge_index, edge_attr)

            preds = out.argmax(dim=1).cpu().numpy()

        # ── Convert to RAS ─────────────────────────────────────────────────────────
        # coords are ZYX → convert to XYZ before affine
        ras_coords = nib.affines.apply_affine(affine, coords)  

        print("Skel points:", coords.shape)
        print("Coords min:", coords.min(axis=0))
        print("Coords max:", coords.max(axis=0))
        print("RAS coords min:", ras_coords.min(axis=0))
        print("RAS coords max:", ras_coords.max(axis=0))

        return ras_coords, preds

    def voxelToRAS(self, norm_coords, affine, shape):
        vox_coords = norm_coords * (shape - 1)      # denormalize
        vox_xyz = vox_coords[:, ::-1]            # ZYX -> XYZ
        return nib.affines.apply_affine(affine, vox_xyz)
        
    
    def _build_graph(self, binary_mask, label_vol, voxel_spacing):
        skel = skeletonize(binary_mask.astype(np.uint8))
        dist = distance_transform_edt(binary_mask, sampling=voxel_spacing)
        skel_obj = Skeleton(skel, spacing=voxel_spacing)
        stats = summarize(skel_obj, separator='-')

        # ── Node features ────────────────────────────────────────────────────────
        coords = skel_obj.coordinates                        # (N, 3)
        degrees = skel_obj.degrees                            # (N,)

        # normalise position to [0,1] within volume
        shape = np.array(binary_mask.shape, dtype=float)
        norm_coords = coords / shape

        node_feats = np.column_stack([norm_coords, degrees])  # (N, 4)

        # ── Node labels ──────────────────────────────────────────────────────────
        node_labels = np.array([
            self._assign_node_label(coords[i], label_vol)
            for i in range(len(coords))
        ])

        # ── Edges ────────────────────────────────────────────────────────────────
        edge_src, edge_dst, edge_feats = [], [], []

        for row_idx, row in stats.iterrows():
            src = int(row["node-id-src"])
            dst = int(row["node-id-dst"])

            path = skel_obj.path_coordinates(int(row_idx)).astype(int)
            radii = dist[path[:, 0], path[:, 1], path[:, 2]]
            euc = max(float(row["euclidean-distance"]), 1e-6)

            edge_src.append(src)
            edge_dst.append(dst)
            edge_src.append(dst)
            edge_dst.append(src)  # undirected

            feats = [
                float(row["branch-distance"]),
                float(row["branch-distance"]) / euc,  # tortuosity
                float(radii.mean()),
                float(radii.min()),
                float(radii.max()),
                float(radii.std()),
            ]
            edge_feats.append(feats)
            edge_feats.append(feats)  # both directions

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float)

        return Data(
            x = torch.tensor(node_feats,   dtype=torch.float),
            edge_index = edge_index,
            edge_attr = edge_attr,
            y = torch.tensor(node_labels,  dtype=torch.long),
            num_nodes = len(coords),
        )

    def _assign_node_label(self, coord, label_vol):
        """
        Assign artery label to a node by majority vote
        in a small neighbourhood around the coordinate.
        """
        r = 2  # neighbourhood radius in voxels
        z, y, x = [int(c) for c in coord]
        patch = label_vol[
            max(0, z-r):z+r+1,
            max(0, y-r):y+r+1,
            max(0, x-r):x+r+1
        ]
        vals, counts = np.unique(patch[patch > 0], return_counts=True)
        if len(vals) == 0:
            return -1  # background node, will be masked in training
        return self.LABEL_TO_IDX.get(self.LABEL_MAP.get(int(vals[counts.argmax()]), ""), -1)
    

    def loadVolume(self, path):
        import nibabel as nib
        path = Path(path)

        if path.suffix in [".nii", ".gz"] or str(path).endswith(".nii.gz"):
            nii = nib.load(str(path))
            data = nii.get_fdata().astype(np.uint8)
            affine = nii.affine
            zooms = nii.header.get_zooms()

        elif path.suffix == ".nrrd":
            import nrrd
            data, header = nrrd.read(str(path))
            data = data.astype(np.uint8)
            if "space directions" in header:
                zooms = [np.linalg.norm(v) for v in header["space directions"]]
            else:
                zooms = [1.0, 1.0, 1.0]
            affine = np.eye(4)
            affine[:3, :3] = np.diag(zooms)

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return data, affine, zooms


