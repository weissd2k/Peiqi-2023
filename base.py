# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'base.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 680)
        font = QtGui.QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(449, 210, 101, 31))
        self.start.setObjectName("start")
        self.concentration_of_absorbate = QtWidgets.QLineEdit(self.centralwidget)
        self.concentration_of_absorbate.setGeometry(QtCore.QRect(110, 160, 271, 31))
        self.concentration_of_absorbate.setObjectName("concentration_of_absorbate")
        self.concentration_of_absorbent = QtWidgets.QLineEdit(self.centralwidget)
        self.concentration_of_absorbent.setGeometry(QtCore.QRect(570, 160, 271, 31))
        self.concentration_of_absorbent.setObjectName("concentration_of_absorbent")
        self.input3 = QtWidgets.QLabel(self.centralwidget)
        self.input3.setGeometry(QtCore.QRect(70, 160, 31, 31))
        self.input3.setObjectName("input3")
        self.input4 = QtWidgets.QLabel(self.centralwidget)
        self.input4.setGeometry(QtCore.QRect(530, 160, 31, 31))
        self.input4.setObjectName("input4")
        self.input1 = QtWidgets.QLabel(self.centralwidget)
        self.input1.setGeometry(QtCore.QRect(70, 110, 31, 31))
        self.input1.setObjectName("input1")
        self.time = QtWidgets.QTextEdit(self.centralwidget)
        self.time.setGeometry(QtCore.QRect(110, 100, 271, 41))
        self.time.setObjectName("time")
        self.amount_of_absorbate = QtWidgets.QTextEdit(self.centralwidget)
        self.amount_of_absorbate.setGeometry(QtCore.QRect(570, 100, 271, 41))
        self.amount_of_absorbate.setObjectName("amount_of_absorbate")
        self.input2 = QtWidgets.QLabel(self.centralwidget)
        self.input2.setGeometry(QtCore.QRect(530, 110, 31, 31))
        self.input2.setObjectName("input2")
        self.output = QtWidgets.QLabel(self.centralwidget)
        self.output.setGeometry(QtCore.QRect(70, 260, 71, 21))
        self.output.setObjectName("output")
        self.result_plot = QtWidgets.QLabel(self.centralwidget)
        self.result_plot.setGeometry(QtCore.QRect(60, 290, 431, 311))
        self.result_plot.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.result_plot.setText("")
        self.result_plot.setScaledContents(True)
        self.result_plot.setObjectName("result_plot")
        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setGeometry(QtCore.QRect(600, 560, 101, 31))
        self.save.setObjectName("save")
        self.clean = QtWidgets.QPushButton(self.centralwidget)
        self.clean.setGeometry(QtCore.QRect(740, 560, 101, 31))
        self.clean.setObjectName("clean")
        self.result_data = QtWidgets.QTextBrowser(self.centralwidget)
        self.result_data.setGeometry(QtCore.QRect(510, 290, 421, 251))
        self.result_data.setObjectName("result_data")
        self.help = QtWidgets.QPushButton(self.centralwidget)
        self.help.setGeometry(QtCore.QRect(140, 260, 21, 23))
        self.help.setObjectName("help")
        self.method = QtWidgets.QLabel(self.centralwidget)
        self.method.setGeometry(QtCore.QRect(70, 10, 851, 71))
        self.method.setObjectName("method")
        self.t_unit = QtWidgets.QLineEdit(self.centralwidget)
        self.t_unit.setGeometry(QtCore.QRect(390, 100, 71, 41))
        self.t_unit.setObjectName("t_unit")
        self.qt_unit = QtWidgets.QLineEdit(self.centralwidget)
        self.qt_unit.setGeometry(QtCore.QRect(850, 100, 71, 41))
        self.qt_unit.setObjectName("qt_unit")
        self.C0_unit = QtWidgets.QLineEdit(self.centralwidget)
        self.C0_unit.setGeometry(QtCore.QRect(390, 160, 71, 31))
        self.C0_unit.setObjectName("C0_unit")
        self.Cs_unit = QtWidgets.QLineEdit(self.centralwidget)
        self.Cs_unit.setGeometry(QtCore.QRect(850, 160, 71, 31))
        self.Cs_unit.setObjectName("Cs_unit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Adsorption Model Calculator"))
        self.start.setText(_translate("MainWindow", "Start"))
        self.input3.setText(_translate("MainWindow", "C0"))
        self.input4.setText(_translate("MainWindow", "Cs"))
        self.input1.setText(_translate("MainWindow", "t"))
        self.input2.setText(_translate("MainWindow", "qt"))
        self.output.setText(_translate("MainWindow", "Results"))
        self.save.setText(_translate("MainWindow", "Save"))
        self.clean.setText(_translate("MainWindow", "Clean"))
        self.help.setText(_translate("MainWindow", "?"))
        self.method.setText(_translate("MainWindow", "Enter experimental data in the first box spliting with \',\' and its unit in the second box.\n"
"For t and qt, start a new line if there are multiple datasets."))
        self.t_unit.setPlaceholderText(_translate("MainWindow", "min"))
        self.qt_unit.setPlaceholderText(_translate("MainWindow", "mg/g"))
        self.C0_unit.setPlaceholderText(_translate("MainWindow", "mg/L"))
        self.Cs_unit.setPlaceholderText(_translate("MainWindow", "g/L"))
