#importing the necessary modules
import numpy as np
import sudoku as sd
import sys
from functools import partial
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

#initializing the sudoku grid
t=np.array([-1]*81).reshape((9,9))

#Extending the QMainWindow class of PyQt5
class MainWindow(QMainWindow):
    def __init__(self,*args,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)
        self.setWindowTitle('Sudoku Solver')
        self.setFixedSize(500,400)
        
        #creating the main layout
        layout=QGridLayout()
        
        #adding color to the sudoku inout boxes
        for i in range(9):
            for j in range(9):
                if i%3==0 and j%3==0 and (i+j==3 or i+j==9):
                    layout.addWidget(Color('gray'),i-i%3,j-j%3,3,3)
        self.w={}
        #adding the QlineEdit widget to input the sudoku
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
        #connecting the input boxes to the 'text_edited' function
        for i in range(9):
            for j in range(9):
                self.w['widget'+str(i)+str(j)].textEdited.connect(partial(self.text_edited,i,j))
        
        #adding layout for the solve button and reset button
        layout2=QVBoxLayout()
        
        #adding solve button to layout2
        btn1=QPushButton('SOLVE')
        btn1.setStatusTip('Click to Solve')
        btn1.pressed.connect(self.solve)
        layout2.addWidget(btn1)
        
        #adding reset button to layout2
        btn2=QPushButton('RESET')
        btn2.setStatusTip('Click to reset Sudoku grid')
        btn2.pressed.connect(self.cleardisp)
        layout2.addWidget(btn2)
        
        #adding label to show the time required to solve the sudoku
        self.label=QLabel('')
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        layout2.addWidget(self.label)
        
        #adding the layout2 to layout
        layout.addLayout(layout2,0,9,9,4)
        
        #adding the my name tag
        l2=QLabel('Made by Pecacio')
        self.statusbar=QStatusBar()
        self.statusbar.addPermanentWidget(l2)
        self.setStatusBar(self.statusbar)
        wid=QWidget()
        wid.setLayout(layout)
        self.setCentralWidget(wid)
    
    #function to set the corresponding number input given by the user the sudoku matrix
    def text_edited(self,a,b,s):
        s=s.strip()
        if s not in ['1','2','3','4','5','6','7','8','9']:
            self.w['widget'+str(a)+str(b)].clear()
            self.w['widget'+str(a)+str(b)].setFocus()
        else:
            t[a][b]=int(s)
     
    #function to solve the sudoku upon pressing the 'SOLVE' button
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
    
    #function to reset the sudoku grid and clear the texts upon pressing the 'RESET' button
    def cleardisp(self):
        global t
        for i in range(9):
            for j in range(9):
                self.w['widget'+str(i)+str(j)].clear()
                self.w['widget'+str(i)+str(j)].setFocus()
        t=np.array([-1]*81).reshape((9,9))
        self.label.setText('')

#Extending the QWidget class to add color to the background
class Color(QWidget):
    def __init__(self,color,*args,**kwargs):
        super(Color,self).__init__(*args,*kwargs)
        self.setAutoFillBackground(True)
        palette=self.palette()
        palette.setColor(QPalette.Window,QColor(color))
        self.setPalette(palette)

#main function to execute the code
def main():
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())
