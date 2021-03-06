import numpy as np
import datetime

#Set N=9 for a 9x9 sudoku board
N=9

#A function to check whether the input at 'pos' is a valid move
def checkvalid(board,value,pos):
  x,y=pos
  col=board[:,y].copy()
  if value in col:
    return False
  row=board[x,:].copy()
  if value in row:
    return False
  x0=x-x%3
  y0=y-y%3
  box=board[x0:x0+3,y0:y0+3].copy()
  if value in box:
    return False
  return True

#funtion to check whether the board is solved
def is_solved(board):
  ind=np.argwhere(board<0)
  if len(ind)==0:
    return True
  return False

#funtion to print the board in the format of sudoku
def print_board(board):
  global N
  t=board.astype(str)
  t=np.where(t=='-1',' ',t)
  txt=''
  for i in range(N):
    for j in range(N):
      if j%3==0 and j>0:
        txt+=' | '
      txt+=' '+t[i,j]
    txt+='\n'
    if i%3==2 and i>0 and i!=N-1:
      txt+=''.join(['-']*24)
      txt+='\n'
  print(txt)

#function to check whether the board given as input is valid or not
def is_valid_board(board):
  ind=np.argwhere(board>0)
  for i in ind:
    x,y=i
    row=board[x,:].copy()
    col=board[:,y].copy()
    row=np.delete(row,y)
    col=np.delete(col,x)
    if (board[x,y] in row) or (board[x,y] in col):
      return False
  return True

#function to implement the depth-first-search algorithm to solve the sudoku
def dfs(board,depth=0):
  ind=np.argwhere(board<0)
  if depth==0:
    if is_valid_board(board)==False:
      return -2
  if is_solved(board):
    return board
  indx,indy=ind[0][0],ind[0][1]
  check=False
  for i in range(1,10):
    check=checkvalid(board,i,ind[0])
    if check and len(ind)!=0:
      board[indx,indy]=i
      dfs(board,depth+1)
      if is_solved(board):
        return board
      board[indx,indy]=-1
  if check==False:
    if depth==0:
      return -1
    return board

#function to input the board and get the sudoku solution
def solver(board):
  t=np.array(board)
  start=datetime.datetime.now()
  ans=dfs(t)
  stop=datetime.datetime.now()
  if type(ans)==int:
    if ans==-1:
      print('No solution exists')
    if ans==-2:
      print('Invalid entry')
    print_board(t)
    return
  delta=stop-start
  tot_time=delta.total_seconds()
  print('Solved within '+str(tot_time)+' seconds')
  print_board(ans)
