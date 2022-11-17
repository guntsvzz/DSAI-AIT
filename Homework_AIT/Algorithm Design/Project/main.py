from tkinter import *
    
root = Tk()
root.geometry('800x500')
root.title('Autistic Application')
root.resizable(False,False)

txt_input = StringVar(value="0")
text_list = list()

def addText(text):
    global text_list 
    text_list.append(text)
    content = " ".join(text_list)
    txt_input.set(content)
    return text_list

def deleteText():
    global text_list
    text_list.pop(-1)
    content = " ".join(text_list)
    txt_input.set(content)
    return text_list
    
def convertText():
    word_list = " ".join(text_list)
    return word_list

#Tap Bar
frame1 = Frame(root,width=800,height=50,bg='grey') 
frame1.pack()
displayEntry = Entry(frame1,font=('arial',30,'bold'),width=35,fg="white",bg="black",textvariable=txt_input,justify="right")
displayEntry.place(x=250,y=25,anchor='c')
deleteButton = Button(frame1,text='Delete',bg='grey',width=10,height=3,command=deleteText)
deleteButton.place(x=680,y=25,anchor='c')
confirmButton = Button(frame1,text='Confirm',bg='grey',width=10,height=3,command=convertText)
confirmButton.place(x=760,y=25,anchor='c')
#Icon Bar
frame2 = Frame(root,width=800,height=450,bg='coral') 
frame2.pack()


for x in range(3):
    Grid.columnconfigure(frame2,x,weight=1)
for y in range(3):
    Grid.columnconfigure(frame2,x,weight=1)

A = Button(frame2,text='A',bg='orange',command=lambda:addText("A"))
B = Button(frame2,text='B',bg='orange',command=lambda:addText("B"))
C = Button(frame2,text='C',bg='orange',command=lambda:addText("C"))
D = Button(frame2,text='D',bg='orange',command=lambda:addText("D"))
E = Button(frame2,text='E',bg='orange',command=lambda:addText("E"))
F = Button(frame2,text='F',bg='orange',command=lambda:addText("F"))
G = Button(frame2,text='G',bg='orange',command=lambda:addText("G"))
H = Button(frame2,text='H',bg='orange',command=lambda:addText("H"))
I = Button(frame2,text='I',bg='orange',command=lambda:addText("I"))

A.grid(row=0,column=0,sticky='NSEW')
B.grid(row=0,column=1,sticky='NSEW')
C.grid(row=0,column=2,sticky='NSEW')
D.grid(row=1,column=0,sticky='NSEW')
E.grid(row=1,column=1,sticky='NSEW')
F.grid(row=1,column=2,sticky='NSEW')
G.grid(row=2,column=0,sticky='NSEW')
H.grid(row=2,column=1,sticky='NSEW')
I.grid(row=2,column=2,sticky='NSEW')

root.mainloop()