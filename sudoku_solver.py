import numpy as np
import sudoku as sd
import sys
from functools import partial
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
t=np.array([-1]*81).reshape((9,9))
class MainWindow(QMainWindow):
    def __init__(self,*args,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)
        self.setWindowTitle('Sudoku Solver')
        self.setFixedSize(500,400)
        layout=QGridLayout()
        for i in range(9):
            for j in range(9):
                if i%3==0 and j%3==0 and (i+j==3 or i+j==9):
                    layout.addWidget(Color('gray'),i-i%3,j-j%3,3,3)
        self.w={}
        for i in range(9):
            for j in range(9):
                a=i-i%3
                b=j-j%3
                self.w['widget'+str(i)+str(j)]=QLineEdit()
                self.w['widget'+str(i)+str(j)].setMaxLength(1)
                self.w['widget'+str(i)+str(j)].setPlaceholderText(' ')
                self.w['widget'+str(i)+str(j)].setTextMargins(2,2,2,2)
                self.w['widget'+str(i)+str(j)].setFixedWidth(30)
                self.w['widget'+str(i)+str(j)].setAlignment(Qt.AlignCenter)
                palette = self.w['widget'+str(i)+str(j)].palette()
                palette.setColor(QPalette.Active, QPalette.Text, QColor('red'))
                if (a+b==3 or a+b==9):
                    palette.setColor(QPalette.Active, QPalette.Base, QColor('gray'))
                self.w['widget'+str(i)+str(j)].setPalette(palette)
                layout.addWidget(self.w['widget'+str(i)+str(j)],i,j)
        for i in range(9):
            for j in range(9):
                self.w['widget'+str(i)+str(j)].textEdited.connect(partial(self.text_edited,i,j))

        layout2=QVBoxLayout()

        btn1=QPushButton('SOLVE')
        btn1.setStatusTip('Click to Solve')
        btn1.pressed.connect(self.solve)
        layout2.addWidget(btn1)

        btn2=QPushButton('RESET')
        btn2.setStatusTip('Click to reset Sudoku grid')
        btn2.pressed.connect(self.cleardisp)
        layout2.addWidget(btn2)

        self.label=QLabel('')
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        layout2.addWidget(self.label)

        layout.addLayout(layout2,0,9,9,4)

        l2=QLabel('Made by Pecacio')
        self.statusbar=QStatusBar()
        self.statusbar.addPermanentWidget(l2)
        self.setStatusBar(self.statusbar)
        wid=QWidget()
        wid.setLayout(layout)
        self.setCentralWidget(wid)
    def text_edited(self,a,b,s):
        s=s.strip()
        if s not in ['1','2','3','4','5','6','7','8','9']:
            self.w['widget'+str(a)+str(b)].clear()
            self.w['widget'+str(a)+str(b)].setFocus()
        else:
            t[a][b]=int(s)
    def solve(self):
        start=datetime.datetime.now()
        self.label.setText('Solving...')
        output=sd.dfs(t)
        stop=datetime.datetime.now()
        delta=(stop-start).total_seconds()
        if type(output)==int:
            if output==-1:
                self.label.setText('No Solution exists')
            if output==-2:
                self.label.setText('Invalid Entry')
        else:
            for i in range(9):
                for j in range(9):
                    self.w['widget'+str(i)+str(j)].setText(str(output[i][j]))
            self.label.setText('Solved within '+str(delta)+' seconds')
    def cleardisp(self):
        global t
        for i in range(9):
            for j in range(9):
                self.w['widget'+str(i)+str(j)].clear()
                self.w['widget'+str(i)+str(j)].setFocus()
        t=np.array([-1]*81).reshape((9,9))
        self.label.setText('')
class Color(QWidget):
    def __init__(self,color,*args,**kwargs):
        super(Color,self).__init__(*args,*kwargs)
        self.setAutoFillBackground(True)
        palette=self.palette()
        palette.setColor(QPalette.Window,QColor(color))
        self.setPalette(palette)
def main():
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())
