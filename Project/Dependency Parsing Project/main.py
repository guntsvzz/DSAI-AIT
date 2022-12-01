from tkinter import *
from tkinter import messagebox
from utils import *

def mainScreen():
    global text_list
    location = [50,150,250,350,450,550]
    list1 = ['I' ,'you'  ,'it'   ,'this','that' ,'we']
    list2 = ['is','am'   ,'are'  ,'do'  ,'have' ,'can']
    list3 = ['go', 'want','like' ,'get' ,'help' ,'eat']
    list4 = ['to','in'   ,'for'  ,'for' ,'up'   ,'with']
    list5 = ['apple','banana','orange','grape','durian','Kiwi']
    list6 = ['what','how','when','where','who','why']
    list7 = ['sleep','talk','swim','write','read','drink']
    list8 = ['and','or','because','but','so','if'] #,'than','as','else','until','a','the']

    window = Tk()
    window.title('Test')
    window.geometry('800x700')
    window.resizable(False,False)

    def addText(text):
        print('added : ',text)
        global text_list 
        text_list.append(text)
        content = " ".join(text_list)
        txt_input.set(content)
        return text_list

    def deleteText():
        global text_list
        remove = text_list.pop(-1)
        print('remove : ',remove)
        content = " ".join(text_list)
        txt_input.set(content)
        return text_list
        
    def convertText():
        global s,word_list,graph,new_config
        word_list = " ".join(text_list)
        # print(word_list)
        # return word_list
        word_list = convertToSpacy(word_list)
        #Stack
        s = Stack()
        for i in range(len(word_list)):
            s.push(word_list[i][1])
        print(s.container)
        #StackandBuffer
        new_config = SaB(word_list)
        graph = DependencyGraph(word_list,new_config)
        print(graph)
        #BFS
        BFS(graph,'0')
        #DFS
        DFS(graph,'0')
        #Comparision
        comparision(word_list,s,graph)


    txt_input = StringVar(value="0")
    text_list = []
    #top
    display = Frame(window,width=800,height=50,bg='grey') 
    display.pack(side=TOP)
    sub = Frame(window,width=800,height=50,bg='white') 
    sub.pack(side=TOP) 
    #bottom
    frame1 = Frame(window, width=100, height=600, bg = 'orange')
    frame1.pack(side=LEFT)
    frame2 = Frame(window, width=100, height=600, bg = 'plum3')
    frame2.pack(side=LEFT)
    frame3 = Frame(window, width=100, height=600, bg = 'plum2')
    frame3.pack(side=LEFT)
    frame4 = Frame(window, width=100, height=600, bg = 'olive drab1')
    frame4.pack(side=LEFT)
    frame5 = Frame(window, width=100, height=600, bg = 'light blue')
    frame5.pack(side=LEFT)
    frame6 = Frame(window, width=100, height=600, bg = 'ivory3')
    frame6.pack(side=LEFT)
    frame7 = Frame(window, width=100, height=600, bg = 'rosy brown1')
    frame7.pack(side=LEFT)
    frame8 = Frame(window, width=100, height=600, bg = 'light salmon')
    frame8.pack(side=LEFT)
    #Button
    displayEntry = Entry(display,font=('arial',30,'bold'),width=35,fg="white",bg="black",textvariable=txt_input,justify="right")
    displayEntry.place(x=250,y=25,anchor='c')
    deleteButton = Button(display,text='Delete',bg='grey',width=10,height=3,command=deleteText)
    deleteButton.place(x=680,y=25,anchor='c')
    confirmButton = Button(display,text='Confirm',bg='grey',width=10,height=3,command=convertText)
    confirmButton.place(x=760,y=25,anchor='c')
    frame_list = [frame1,frame2,frame3,frame4,frame5,frame6,frame7,frame8]
    color_list = ['orange','plum3','plum2','olive drab1','light blue','ivory3','rosy brown1','light salmon']
    #loop orange
    j = 0
    bt = [i for i in range(36)]

    #List1
    bt1 = Button(frame1,text=f'{list1[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[0]}',command=lambda:addText(f'{list1[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame1,text=f'{list1[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[0]}',command=lambda:addText(f'{list1[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame1,text=f'{list1[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[0]}',command=lambda:addText(f'{list1[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame1,text=f'{list1[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[0]}',command=lambda:addText(f'{list1[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame1,text=f'{list1[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[0]}',command=lambda:addText(f'{list1[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame1,text=f'{list1[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[0]}',command=lambda:addText(f'{list1[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List2
    bt1 = Button(frame2,text=f'{list2[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[1]}',command=lambda:addText(f'{list2[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame2,text=f'{list2[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[1]}',command=lambda:addText(f'{list2[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame2,text=f'{list2[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[1]}',command=lambda:addText(f'{list2[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame2,text=f'{list2[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[1]}',command=lambda:addText(f'{list2[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame2,text=f'{list2[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[1]}',command=lambda:addText(f'{list2[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame2,text=f'{list2[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[1]}',command=lambda:addText(f'{list2[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List3
    bt1 = Button(frame3,text=f'{list3[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[2]}',command=lambda:addText(f'{list3[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame3,text=f'{list3[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[2]}',command=lambda:addText(f'{list3[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame3,text=f'{list3[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[2]}',command=lambda:addText(f'{list3[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame3,text=f'{list3[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[2]}',command=lambda:addText(f'{list3[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame3,text=f'{list3[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[2]}',command=lambda:addText(f'{list3[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame3,text=f'{list3[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[2]}',command=lambda:addText(f'{list3[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List4
    bt1 = Button(frame4,text=f'{list4[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[3]}',command=lambda:addText(f'{list4[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame4,text=f'{list4[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[3]}',command=lambda:addText(f'{list4[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame4,text=f'{list4[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[3]}',command=lambda:addText(f'{list4[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame4,text=f'{list4[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[3]}',command=lambda:addText(f'{list4[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame4,text=f'{list4[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[3]}',command=lambda:addText(f'{list4[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame4,text=f'{list4[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[3]}',command=lambda:addText(f'{list4[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List5
    bt1 = Button(frame5,text=f'{list5[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[4]}',command=lambda:addText(f'{list5[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame5,text=f'{list5[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[4]}',command=lambda:addText(f'{list5[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame5,text=f'{list5[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[4]}',command=lambda:addText(f'{list5[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame5,text=f'{list5[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[4]}',command=lambda:addText(f'{list5[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame5,text=f'{list5[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[4]}',command=lambda:addText(f'{list5[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame5,text=f'{list5[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[4]}',command=lambda:addText(f'{list5[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List6
    bt1 = Button(frame6,text=f'{list6[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[5]}',command=lambda:addText(f'{list6[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame6,text=f'{list6[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[5]}',command=lambda:addText(f'{list6[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame6,text=f'{list6[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[5]}',command=lambda:addText(f'{list6[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame6,text=f'{list6[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[5]}',command=lambda:addText(f'{list6[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame6,text=f'{list6[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[5]}',command=lambda:addText(f'{list6[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame6,text=f'{list6[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[5]}',command=lambda:addText(f'{list6[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List7
    bt1 = Button(frame7,text=f'{list7[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[6]}',command=lambda:addText(f'{list7[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame7,text=f'{list7[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[6]}',command=lambda:addText(f'{list7[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame7,text=f'{list7[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[6]}',command=lambda:addText(f'{list7[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame7,text=f'{list7[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[6]}',command=lambda:addText(f'{list7[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame7,text=f'{list7[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[6]}',command=lambda:addText(f'{list7[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame7,text=f'{list7[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[6]}',command=lambda:addText(f'{list7[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')
    #List8
    bt1 = Button(frame8,text=f'{list8[0]}',font = (None , 20),width=5,height=1,bg=f'{color_list[7]}',command=lambda:addText(f'{list8[0]}'))
    bt1.place(x=50,y=f'{location[0]}',anchor='c')
    bt2 = Button(frame8,text=f'{list8[1]}',font = (None , 20),width=5,height=1,bg=f'{color_list[7]}',command=lambda:addText(f'{list8[1]}'))
    bt2.place(x=50,y=f'{location[1]}',anchor='c')
    bt3 = Button(frame8,text=f'{list8[2]}',font = (None , 20),width=5,height=1,bg=f'{color_list[7]}',command=lambda:addText(f'{list8[2]}'))
    bt3.place(x=50,y=f'{location[2]}',anchor='c')
    bt4 = Button(frame8,text=f'{list8[3]}',font = (None , 20),width=5,height=1,bg=f'{color_list[7]}',command=lambda:addText(f'{list8[3]}'))
    bt4.place(x=50,y=f'{location[3]}',anchor='c')
    bt5 = Button(frame8,text=f'{list8[4]}',font = (None , 20),width=5,height=1,bg=f'{color_list[7]}',command=lambda:addText(f'{list8[4]}'))
    bt5.place(x=50,y=f'{location[4]}',anchor='c')
    bt6 = Button(frame8,text=f'{list8[5]}',font = (None , 20),width=5,height=1,bg=f'{color_list[7]}',command=lambda:addText(f'{list8[5]}'))
    bt6.place(x=50,y=f'{location[5]}',anchor='c')

    window.mainloop()

if __name__ == '__main__':
    mainScreen()