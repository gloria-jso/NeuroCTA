import qt
import slicer
import vtk

class VesselTableManager(qt.QTableWidget):
    def __init__(self):
        super().__init__()
        self._buildTable()
        self._segmentationNode = None

    def _buildTable(self):
        self.segmentsTableWidget = qt.QTableWidget() 
        self.setColumnCount(3)
        self.setEditTriggers(qt.QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setMinimumHeight(150)
        self.setSelectionBehavior(qt.QTableWidget.SelectRows)
        self.verticalHeader().setDefaultSectionSize(24)

         # add button in header to hide/show all segments
        self._allSegmentVisibleState = True
        self._segVisHeaderButton = qt.QPushButton(self.horizontalHeader())
        self._segVisHeaderButton.setIcon(qt.QIcon(":/Icons/Small/SlicerVisible.png"))
        self._segVisHeaderButton.setFlat(True)
        self._segVisHeaderButton.setFixedSize(20, 20)
        self._segVisHeaderButton.setIconSize(qt.QSize(16, 16))
        self._segVisHeaderButton.setVisible(False)
        self.setColumnWidth(0, 24)
        self._segVisHeaderButton.connect("clicked()", self._onToggleAllSegmentVisibility)

        # segment colour
        self.setHorizontalHeaderLabels(["", "", "Segment"])
        self.setColumnWidth(1, 24)
        self.connect("cellClicked(int,int)", self._onSegmentsTableCellClicked) # open QColorDialog

        # column resizing -- fix segment visibility and colour, stretch Segment
        self.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.Stretch)

        # Centreline -- only show after skeletonization
        self._allCenterlineVisibleState = True
        self._clVisHeaderButton = qt.QPushButton(self.horizontalHeader())
        self._clVisHeaderButton.setIcon(qt.QIcon(":/Icons/Small/SlicerVisible.png"))
        self._clVisHeaderButton.setFlat(True)
        self._clVisHeaderButton.setFixedSize(20, 20)
        self._clVisHeaderButton.setIconSize(qt.QSize(16, 16))
        self._clVisHeaderButton.setVisible(False)
        self._clVisHeaderButton.connect("clicked()", self._onToggleAllCenterlineVisibility)

        self.horizontalHeader().connect(
            "sectionResized(int,int,int)", self._repositionCLHeaderButton)
        self.horizontalHeader().connect(
            "sectionMoved(int,int,int)", self._repositionCLHeaderButton)

        # only turn on Segment table if input type is Segmentation
        self.setVisible(False)


    def populateSegments(self, segmentationNode):
        self._segmentationNode = segmentationNode
        segmentation = segmentationNode.GetSegmentation()
        nSegments = segmentation.GetNumberOfSegments()

        self.setRowCount(nSegments)

        visIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")
        invisIcon = qt.QIcon(":/Icons/Small/SlicerInvisible.png")
        displayNode = segmentationNode.GetDisplayNode()

        for i in range(nSegments):
            segmentId = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segmentId)
            r, g, b = segment.GetColor()

            # Visibility toggle
            isVisible = displayNode.GetSegmentVisibility3D(segmentId)
            visButton = qt.QPushButton()
            visButton.setIcon(visIcon if isVisible else invisIcon)
            visButton.setFlat(True)
            visButton.setFixedSize(20, 20)
            visButton.setIconSize(qt.QSize(16, 16))
            visButton.setProperty("segmentId", segmentId)
            visButton.setProperty("segmentVisible", isVisible)
            visButton.connect("clicked()", lambda sid=segmentId, btn=visButton: self._onSegmentVisibilityToggled(sid, btn))
            self.setCellWidget(i, 0, visButton)

            # Color swatch
            colorItem = qt.QTableWidgetItem()
            colorItem.setBackground(qt.QColor(int(r*255), int(g*255), int(b*255)))
            colorItem.setFlags(qt.Qt.ItemIsEnabled)
            self.setItem(i, 1, colorItem)

            # Segment name
            self.setItem(i, 2, qt.QTableWidgetItem(segment.GetName()))
       
        colPos = self.columnViewportPosition(0)
        colWidth = self.columnWidth(0)
        self._segVisHeaderButton.move(colPos + (colWidth - 24) // 2, 2)        
        self._segVisHeaderButton.setVisible(True)
        self._syncToSlicerTableNode()

    def populateFeatureColumns(self):
        # --- Remove existing metric columns ---
        metricColNames = ["Length (mm)", "Tortuosity", "Mean Radius (mm)"]
        headers = [self.horizontalHeaderItem(c).text()
                for c in range(self.columnCount)]
        for colName in metricColNames:
            if colName in headers:
                self.removeColumn(headers.index(colName))
                headers = [self.horizontalHeaderItem(c).text()
                        for c in range(self.columnCount)]

        # --- Add metric columns ---
        for colName in metricColNames:
            col = self.columnCount
            self.insertColumn(col)
            item = qt.QTableWidgetItem(colName)
            font = item.font()
            font.setPointSize(9)
            item.setFont(font)
            self.setHorizontalHeaderItem(col, item)
            self.horizontalHeader().setSectionResizeMode(col, qt.QHeaderView.Stretch)

        # --- Populate values ---
        headers = [self.horizontalHeaderItem(c).text()
                for c in range(self.columnCount)]

        metricToHeader = {
            "length":      "Length (mm)",
            "tortuosity":  "Tortuosity",
            "mean_radius": "Mean Radius (mm)",
        }

        for i in range(self.rowCount):
            segmentation = self._segmentationNode.GetSegmentation()
            segmentId    = segmentation.GetNthSegmentID(i)
            segmentName  = segmentation.GetSegment(segmentId).GetName()

            data     = self._skeletonsBySegment.get(segmentName, {})
            features = data.get("features", {}) if isinstance(data, dict) else {}

            for featureKey, headerName in metricToHeader.items():
                if headerName not in headers:
                    continue
                col   = headers.index(headerName)
                value = features.get(featureKey, "N/A")
                text  = f"{value:.3f}" if isinstance(value, float) else str(value)
                self.setItem(i, col, qt.QTableWidgetItem(text))
        self._syncToSlicerTableNode()


    def setSkeletons(self, skeletonsBySegment) -> None:
        self._skeletonsBySegment = skeletonsBySegment

    def _onToggleAllSegmentVisibility(self):
        self._allSegmentVisibleState = not self._allSegmentVisibleState
        visIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")
        invisIcon = qt.QIcon(":/Icons/Small/SlicerInvisible.png")
        icon = visIcon if self._allSegmentVisibleState else invisIcon
        self._segVisHeaderButton.setIcon(icon)

        segmentationNode = self._segmentationNode
        if not segmentationNode:
            return

        segmentation = segmentationNode.GetSegmentation()
        displayNode = segmentationNode.GetDisplayNode()

        for i in range(self.rowCount):
            segmentId = segmentation.GetNthSegmentID(i)
            displayNode.SetSegmentVisibility(segmentId, self._allSegmentVisibleState)
            btn = self.cellWidget(i, 0)
            if btn:
                btn.setIcon(icon)
                btn.setProperty("segmentVisible", self._allSegmentVisibleState)

    def _onSegmentVisibilityToggled(self, segmentId, button):
        segmentationNode = self._segmentationNode
        if not segmentationNode:
            return

        displayNode = segmentationNode.GetDisplayNode()
        isVisible = not button.property("segmentVisible")
        displayNode.SetSegmentVisibility(segmentId, isVisible)
        button.setProperty("segmentVisible", isVisible)

        visIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")
        invisIcon = qt.QIcon(":/Icons/Small/SlicerInvisible.png")
        button.setIcon(visIcon if isVisible else invisIcon)

    
    def _onSegmentsTableCellClicked(self, row, col):
        if col != 1:  # only color swatch column
            return

        segmentationNode = self._segmentationNode
        if not segmentationNode:
            return

        segmentation = segmentationNode.GetSegmentation()
        segmentId = segmentation.GetNthSegmentID(row)
        segment = segmentation.GetSegment(segmentId)
        r, g, b = segment.GetColor()

        currentColor = qt.QColor(int(r*255), int(g*255), int(b*255))
        newColor = qt.QColorDialog.getColor(currentColor, self.parent, f"Pick colour for {segment.GetName()}")

        if not newColor.isValid():
            return

        # Update segment color
        segment.SetColor(newColor.red()/255.0, newColor.green()/255.0, newColor.blue()/255.0)

        # Update swatch in table
        colorItem = self.item(row, 1)
        colorItem.setBackground(newColor)

        # Update skeleton color if it exists
        skelNode = self._skeletonsBySegment.get(segment.GetName()) if hasattr(self, '_skeletonsBySegment') else None
        if skelNode and skelNode.GetDisplayNode():
            skelNode.GetDisplayNode().SetColor(newColor.red()/255.0, newColor.green()/255.0, newColor.blue()/255.0)

    def populateCLColumn(self):
        visIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")

        # Check if columns already exist
        headers = [self.horizontalHeaderItem(c).text() 
                   for c in range(self.columnCount)]

        if "CL Points" not in headers:
            clPtsCol = self.columnCount
            self.insertColumn(clPtsCol)
            self.setHorizontalHeaderItem(clPtsCol, qt.QTableWidgetItem("CL Points"))
            self.setColumnWidth(clPtsCol, 110)
        else:
            clPtsCol = headers.index("CL Points")

        segmentationNode = self._segmentationNode
        segmentation = segmentationNode.GetSegmentation()

        for i in range(self.rowCount):
            segmentId = segmentation.GetNthSegmentID(i)
            segmentName = segmentation.GetSegment(segmentId).GetName()
            skelNode = self._skeletonsBySegment.get(segmentName, {}).get("modelNode")
            nPts = skelNode.GetPolyData().GetNumberOfPoints() if skelNode else 0

            # Reuse existing button if column already existed
            existingWidget = self.cellWidget(i, clPtsCol)
            if existingWidget:
                btn = existingWidget.findChild(qt.QPushButton)
            else:
                btn = None

            if btn is None:
                cellWidget = qt.QWidget()
                cellLayout = qt.QHBoxLayout(cellWidget)
                cellLayout.setContentsMargins(2, 0, 2, 0)

                clVisBtn = qt.QPushButton()
                clVisBtn.setIcon(visIcon)
                clVisBtn.setFlat(True)
                clVisBtn.setFixedSize(20, 20)
                clVisBtn.setIconSize(qt.QSize(16, 16))
                clVisBtn.setProperty("segmentName", segmentName)
                clVisBtn.setProperty("clVisible", True)
                clVisBtn.connect("clicked()", lambda btn=clVisBtn, name=segmentName: self._onCenterlineVisibilityToggled(name, btn))

                ptsLabel = qt.QLabel(str(nPts))
                cellLayout.addWidget(clVisBtn)
                cellLayout.addWidget(ptsLabel)
                cellLayout.addStretch()
                self.setCellWidget(i, clPtsCol, cellWidget)
            else:
                # Just update the label
                label = existingWidget.findChild(qt.QLabel)
                if label:
                    label.setText(str(nPts))

        # Position header eye button over CL Points column
        headers = [self.horizontalHeaderItem(c).text()
                   for c in range(self.columnCount)]
        clPtsCol = headers.index("CL Points")
        colPos = self.columnViewportPosition(clPtsCol)
        self._clVisHeaderButton.move(colPos + 2, 2)
        self._clVisHeaderButton.setVisible(True)
        self._syncToSlicerTableNode()

    def removeCLColumns(self):
        headers = [self.horizontalHeaderItem(c).text()
                   for c in range(self.columnCount)]
        for colName in ["CL Points"]:
            if colName in headers:
                self.removeColumn(headers.index(colName))
                headers = [self.horizontalHeaderItem(c).text()
                           for c in range(self.columnCount)]
        self._clVisHeaderButton.setVisible(False)
        self._allCenterlineVisibleState = True
        self._syncToSlicerTableNode()


    def _onCenterlineVisibilityToggled(self, segmentName, button):
        skelNode = self._skeletonsBySegment.get(segmentName, {}).get("modelNode")
        if not skelNode:
            return

        isVisible = not button.property("clVisible")
        skelNode.GetDisplayNode().SetVisibility(isVisible)
        button.setProperty("clVisible", isVisible)

        visIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")
        invisIcon = qt.QIcon(":/Icons/Small/SlicerInvisible.png")
        button.setIcon(visIcon if isVisible else invisIcon)

    def _onToggleAllCenterlineVisibility(self):
        self._allCenterlineVisibleState = not self._allCenterlineVisibleState
        visIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")
        invisIcon = qt.QIcon(":/Icons/Small/SlicerInvisible.png")
        self._clVisHeaderButton.setIcon(visIcon if self._allCenterlineVisibleState else invisIcon)

        headers = [self.horizontalHeaderItem(c).text()
                   for c in range(self.columnCount)]
        if "CL Points" not in headers:
            return
        clVisCol = headers.index("CL Points")

        for i in range(self.rowCount):
            cellWidget = self.cellWidget(i, clVisCol)
            btn = cellWidget.findChild(qt.QPushButton) if cellWidget else None
            if not btn:
                continue
            segmentName = btn.property("segmentName")
            skelNode = self._skeletonsBySegment.get(segmentName, {}).get("modelNode")
            if skelNode:
                skelNode.GetDisplayNode().SetVisibility(self._allCenterlineVisibleState)
            btn.setIcon(visIcon if self._allCenterlineVisibleState else invisIcon)
            btn.setProperty("clVisible", self._allCenterlineVisibleState)

    def _repositionCLHeaderButton(self, *args):
        if not self._clVisHeaderButton.isVisible():
            return
        for c in range(self.columnCount):
            item = self.horizontalHeaderItem(c)
            if item and item.text() == "CL Points":
                colPos = self.columnViewportPosition(c)
                self._clVisHeaderButton.move(colPos + 2, 2)
                return
            
    def export(self):
        import csv

        filePath = qt.QFileDialog.getSaveFileName(
            self, "Export Table", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filePath:
            return
        if not filePath.endswith(".csv"):
            filePath += ".csv"

        with open(filePath, "w", newline="") as f:
            writer = csv.writer(f)

            # --- Build header list, skipping vis/color cols (headers "" or " ") ---
            headers = []
            for col in range(self.columnCount):
                item = self.horizontalHeaderItem(col)
                headers.append(item.text() if item else "")
            exportCols = [c for c in range(self.columnCount)
                        if headers[c] not in ("", " ")]
            writer.writerow([headers[c] for c in exportCols])

            # --- Write data rows ---
            skeletons = getattr(self, "_skeletonsBySegment", {})
            segmentation = self._segmentationNode.GetSegmentation() if self._segmentationNode else None

            for row in range(self.rowCount):
                # Get segment name for this row (used as fallback for widget cells)
                segmentName = ""
                if segmentation:
                    segmentId = segmentation.GetNthSegmentID(row)
                    segmentName = segmentation.GetSegment(segmentId).GetName()

                rowData = []
                for col in exportCols:
                    item = self.item(row, col)
                    if item is not None:
                        rowData.append(item.text())
                    else:
                        # Widget cell — derive value from source data instead of parsing the widget
                        colName = headers[col]
                        if colName == "CL Points" and segmentName:
                            skelNode = skeletons.get(segmentName, {}).get("modelNode")
                            nPts = skelNode.GetPolyData().GetNumberOfPoints() if skelNode else 0
                            rowData.append(str(nPts))
                        else:
                            rowData.append("")
                writer.writerow(rowData)

        qt.QMessageBox.information(self, "Export Complete", f"Table exported to:\n{filePath}")

    def _syncToSlicerTableNode(self):
        skeletons = getattr(self, "_skeletonsBySegment", {})
        segmentation = self._segmentationNode.GetSegmentation() if self._segmentationNode else None
        if not segmentation:
            return

        # Always recreate for a clean slate
        tableNode = slicer.mrmlScene.GetFirstNodeByName("VesselMetrics")
        if tableNode:
            slicer.mrmlScene.RemoveNode(tableNode)
        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "VesselMetrics")

        # --- Build exportable column list ---
        headers = []
        for col in range(self.columnCount):
            item = self.horizontalHeaderItem(col)
            headers.append(item.text() if item else "")
        exportCols = [c for c in range(self.columnCount)
                    if headers[c] not in ("", " ")]

        # --- Create vtk columns ---
        for col in exportCols:
            arr = vtk.vtkStringArray()
            arr.SetName(headers[col])
            tableNode.AddColumn(arr)

        # --- Fill rows ---
        table = tableNode.GetTable()
        for row in range(self.rowCount):
            segmentId = segmentation.GetNthSegmentID(row)
            segmentName = segmentation.GetSegment(segmentId).GetName()
            table.InsertNextBlankRow()

            for i, col in enumerate(exportCols):
                item = self.item(row, col)
                if item is not None:
                    value = item.text()
                else:
                    colName = headers[col]
                    if colName == "CL Points" and segmentName:
                        skelNode = skeletons.get(segmentName, {}).get("modelNode")
                        nPts = skelNode.GetPolyData().GetNumberOfPoints() if skelNode else 0
                        value = str(nPts)
                    else:
                        value = ""
                table.GetColumn(i).SetValue(row, value)

        table.Modified()