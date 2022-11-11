from tkinter import *

root = Tk()
root.geometry('800x500')
root.title('Application')
root.resizable(False,False)
#Tap Bar
frame1 = Frame(root,width=800,height=50,bg='white') 
frame1.pack()
DeleteButton = Button(frame1,text='Delete',bg='grey',width=10,height=3)
DeleteButton.place(x=680,y=25,anchor='c')
ConfirmButton = Button(frame1,text='Confirm',bg='grey',width=10,height=3)
ConfirmButton.place(x=760,y=25,anchor='c')
#Icon Bar
frame2 = Frame(root,width=800,height=450,bg='coral') 
frame2.pack()

root.mainloop()