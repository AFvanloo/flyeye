#Minimal version of the program for measuring eye sizes. If you would like more
#options and more configurability, use the 'main.py' function that should be 
# in the same folder as this file. 

#This file uses settings in fly_config.py

#Written by Arjan van Loo, in case of questions email arjanvanloo@gmail.com

#For this program to run, the PC needs to have Python installed, including
# external libraries installation including Tkinter, numpy, matplotlib,
#openCV and PIL

#Program is written for 1600 x 1200 RGB tiff images!

from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk  #ImageTk is part of the PIL package
import os
import time
import numpy as np
import cv, cv2
#import matplotlib
import flyeye as fly
import csv



class Interface(Frame):

    def __init__(self, parent):
        '''
        Initializing the window
        '''

        Frame.__init__(self, parent)
        self.parent = parent
        self.grid()

        self.impath = ''
        
        #HSV threshold values - later loaded form config
        self.vh = 0
        self.vs = 0
        self.vv = 0
        self.magicnumber = 0

        self.saved = True

        self.adap=IntVar()

        #
        self.tr1 = 0

        #list of possible image processes
        self.processes = ['color thresholding', 'lowpass filter', 'close image', \
                'open image', 'scalar threshold', 'gradient', 'detect edge', \
                'find contour']

        self.tkimsize = (480,360)

        #paths
        self.instpath = os.getcwd()
        self.cfgpath = self.instpath+'/fly_config.txt'

        #default image
        im =  cv2.imread(self.instpath+'/ey1.tif')
        self.tkim = self.convertim(im)

        #default processed image
        pim = np.zeros((480,360,3))
        self.tkpim = self.convertim(pim)


#============ #TO BE COPIED TO FULL VERSION===============================
        
        self.name = ''
        #temp current result variable
        self.curarea = 0
        self.curcontour = ''

        self.area = 0
        
        #Allowed image formats
        self.formats = ["jpg","tif",'TIF','TIFF','png']

        #itertor for imlist
        self.imi = 0

        #all will be saved here
        self.results = []

        self.stopped=False
        self.atEnd = False
       
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       
        #plt.imshow(self.im),plt.show()

        self.verbose = True
        if self.verbose:
            print 'now running get_config and initUI'
        self.get_config()

        self.initUI()
        #self.do_processes()

    
    def initUI(self):
        '''
        create widgets
        All features in the main GUI window are initialized here
        '''

        if self.verbose:
            print 'Initializing the GUI'
   
        #right button column
        rbutton = 5 
        lrow = 17

        #entry for folder plus label, plus load button
        self.pathlabel = Label(self, text= 'current folder: ')
        self.pathlabel.grid(row=0,column=0, sticky=W)

        self.pathEntry = Entry(self,width=60)
        self.pathEntry.insert(0,self.instpath+'/eyes/bla6')
        self.pathEntry.grid(row=0,column=1,columnspan=3)
        
        #self.loadbutton = Button(self,text='load',fg='Blue',command=self.loadfolder)
        #self.loadbutton.grid(row=0,column=rbutton, sticky=E)

        self.openDirButton = Button(self,text='open directory', fg='Blue',\
                command = self.openDir)
        self.openDirButton.grid(row=0,column=rbutton,sticky=E)

        self.openSingleButton = Button(self,text='open single image', \
                fg = 'Blue', command = self.openSingle)
        self.openSingleButton.grid(row=0,column=rbutton-1,sticky=E)

        #images
        self.im1labelframe = LabelFrame(self,text = 'Original image'+self.name,labelanchor='n')
        self.im1labelframe.grid(row=1,column=0,columnspan=3,rowspan=4)
        self.im1label = Label(self.im1labelframe, image=self.tkim)
        self.im1label.grid(row=0, column=0,columnspan=3,rowspan=4)

        self.im2labelframe = LabelFrame(self,text = 'Processed image'+self.name,labelanchor='n')
        self.im2labelframe.grid(row=1,column=3,columnspan=3,rowspan=4)
        self.im2label = Label(self.im2labelframe,image=self.tkpim)
        self.im2label.grid(row=0,column=0,columnspan=3,rowspan=4)

        #Navigation buttons
        self.nextbutton = Button(self, text= 'Next >',fg='blue',command=self.nextButton)
        self.nextbutton.grid(row=5,column=rbutton,sticky=E)
        
        
        self.saveresbutton = Button(self, text= 'Save result image',fg='blue',command=self.saveImage)
        self.saveresbutton.grid(row=5,column=rbutton-1,sticky=E)

        #save and quit
        self.sqbutton = Button(self, text='Quit', fg = 'red',\
                command = self.saveQuitButton)
        self.sqbutton.grid(row=lrow,column=rbutton, sticky=E)


#=====================TO BE COPIED========================================
        self.resultsframe = LabelFrame(self, text='last few results',labelanchor='n',\
                        width=400,height=80)
        self.resultsframe.grid(row=7,rowspan=4,column=0,columnspan=3)

        self.imnamelabel = Label(self.resultsframe,text = 'Image name')
        self.imnamelabel.grid(row=0,column=0,padx=10)

        self.imfolderlabel = Label(self.resultsframe,text = 'Current Folder')
        self.imfolderlabel.grid(row=0,column=1, padx=40)


        self.imarealabel = Label(self.resultsframe,text = 'Area Found')
        self.imarealabel.grid(row=0,column=2, padx=10)

        self.name1label = Label(self.resultsframe, text='')
        self.name1label.grid(row=1,column=0)
        self.name2label = Label(self.resultsframe, text='')
        self.name2label.grid(row=2,column=0)
        self.name3label = Label(self.resultsframe, text='')
        self.name3label.grid(row=3,column=0)

        self.folder1label = Label(self.resultsframe, text='')
        self.folder1label.grid(row=1,column=1)
        self.folder2label = Label(self.resultsframe, text='')
        self.folder2label.grid(row=2,column=1)
        self.folder3label = Label(self.resultsframe, text='')
        self.folder3label.grid(row=3,column=1)


        self.res1label = Label(self.resultsframe, text='')
        self.res1label.grid(row=1,column=2)
        self.res2label = Label(self.resultsframe, text='')
        self.res2label.grid(row=2,column=2)
        self.res3label = Label(self.resultsframe, text='')
        self.res3label.grid(row=3,column=2)
        '''
        The magic number is 'mingaus', the constant value subtracted from 
        the threshold in adaptive thresholding for the hue color channel.
        It is the only value I have not been able to reliably get from the 
        image statistics
        '''
        self.magicLabel = Label(self,text='magic number: \n \
                (for adaptive method)')
        self.magicLabel.grid(row=5,column=0)
        self.magicEntry = Entry(self, width=10)
        self.magicEntry.insert(0,self.magicnumber)
        self.magicEntry.grid(row=5,column=1)
        self.magicButton = Button(self,text = 'recalculate', command=self.recalculate)
        self.magicButton.grid(row=5,column=2)

        #reset results
        self.resetButton = Button(self, text='empty results',\
                command=self.resetButton)
        self.resetButton.grid(row=7,column=rbutton,sticky=E)

        #histogram plotter
        self.histButton = Button(self, text = 'plot histogram', command = \
                self.showHistogram)
        self.histButton.grid(row=8,column=rbutton,sticky=E)

        self.measureSaveAllButton = Button(self, text= 'measure all',\
                command  = self.MeasureAll)
        self.measureSaveAllButton.grid(row=9,column=rbutton-1, sticky=E)

        self.stopButton = Button(self,text = 'stop', command=self.stop)
        self.stopButton.grid(row=9,column=rbutton,sticky=E)
        
        #adaptive thresholding
        self.adaptiveMethod = Checkbutton(self, text='adaptive method',\
                variable=self.adap)
        self.adaptiveMethod.grid(row=5,column=3)
        self.adap.set(1)

        #convex hull
        self.convex = IntVar()
        self.convexButton = Checkbutton(self, text = 'convex hull',\
                variable = self.convex)
        self.convexButton.grid(row=6,column=3)
        self.convex.set(1)

        #Saving results
        self.savelabel = Label(self, text='save results in folder: ')
        self.savelabel.grid(row=lrow-1,column=0,sticky=W)
        self.saveEntry = Entry(self,width=70)
        self.saveEntry.insert(0,self.instpath+'/eyes/bla3')
        self.saveEntry.grid(row=lrow-1,column=1,columnspan=4)
        self.saveAsButton = Button(self, text = 'save as', command = self.saveAs)
        self.saveAsButton.grid(row=lrow-1,column=rbutton-1, sticky=E)
        self.saveButton = Button(self,text=  'save',command = self.save)
        self.saveButton.grid(row=lrow-1,column=rbutton, sticky=E)
        


#=============== Config Loading And Saving ===============================

    def get_config(self):
        '''
        load settings from the config file
        '''
        #open the config file
        cfgfile = open(self.cfgpath)
        cfg = eval(cfgfile.read())

        #HSV thresholds
        self.vh = cfg.get('vh')
        self.vs = cfg.get('vs')
        self.vv = cfg.get('vv')

        self.tr1 = cfg.get('tr1')
        self.magicnumber = cfg.get('magicnumber')
        self.tkimsize = cfg.get('imsize')
        
        self.formats = cfg.get('formats')

        #GUI settings
        self.owner = cfg.get('owner')
        cfgfile.close()

        if self.verbose:
            print 'config values: h, s and v are: ', self.vh, self.vs, self.vv
            print 'config values: tr1 is: ', self.tr1


    def openDir(self):
        '''
        opens directory name
        '''

        newpath = tkFileDialog.askdirectory()
        self.pathEntry.delete(0,END)
        self.pathEntry.insert(0,newpath)
        self.loadfolder()

    def openSingle(self):

        #get filename
        newfile = tkFileDialog.askopenfilename(defaultextension='.tif')
        print 'newfile is ', newfile
        self.imleft = cv2.imread(newfile)

        #set attributes
        pathparts = newfile.split('/')
        print 'pathparts are', pathparts
        self.name = pathparts[-1].split('.')[0]
        gah = ''
        self.impath = '/'.join(pathparts[:-1])
        print 'name is: ', self.name
        print 'impath is ', self.impath
        
        #get area and contour
        self.magicnumber = float(self.magicEntry.get())
        self.getarea_default()

        #store and show results
        self.results.append([self.name,self.impath,self.area])
        if self.verbose:
            print 'current results data : ', self.results

        self.showresults()
        self.saveImage()



    def loadfolder(self):
        '''
        get the path from the entry form, get a list of images, and show the
        first one 
        '''
       

        self.imi = -1
        self.impath = self.pathEntry.get()
        
        #update the savepath
        self.saveEntry.delete(0,END)
        self.saveEntry.insert(0,self.impath)
        
        if self.verbose:
            print 'Loading data from ', self.impath
        #check if folder exists
        if not os.path.isdir(self.impath):
            self.show_messg('Check if entered path contains typos')

        #Get a path walker construction
        #The first entry in the path walker will be the current directory,
        #The second are child directories, and the third are files in this
        #directory
        self.pathwalker = os.walk(self.impath)
        next(self.pathwalker)
        self.get_im_list()

        self.atEnd = False
        
        #go back to standard dir
        #os.chdir('self.instpath')
        self.nextButton()
        #empty results?
        #save results?

    def stop(self):
        self.stopped = True
            
    def nextButton(self,showMessage=True):
        '''
        loads picture, gets contour, shows results
        '''

        self.stopped =  False

        if self.imi == len(self.imlist)-1:
            try: 
                curwalk = next(self.pathwalker)
                print 'current walk is ', curwalk
                os.chdir(curwalk[0])
                print 'change to directory ', os.getcwd()
                self.impath = os.getcwd()
                self.get_im_list()
                #update path Entry
                self.pathEntry.delete(0,END)
                self.pathEntry.insert(0,self.impath)
                #reset iterator
                self.imi = -1
            except StopIteration:
                if showMessage==True:
                    self.show_messg('Already at last image file')
                self.atEnd = True
                return

        self.imi += 1
           
        if self.verbose:
            print 'iterator is at :', self.imi
        #get the filename before the extension
        self.name = self.imlist[self.imi].split('.')[0]
        print 'name is: ', self.name
        self.imleft = self.loadim()
        self.magicnumber = float(self.magicEntry.get())

        self.getarea_default()
        self.results.append([self.name,self.impath,self.area])
        if self.verbose:
            print 'current results data : ', self.results

        self.showresults()
        self.saveImage()

        #new results arent saved yet, so set the saved qualifier to False
        self.saved = False

    def resetButton(self):
        '''
        empty the results table and associated labels
        '''
        self.results = []
        self.imlist = []

        #reset labels
        self.name1label.configure(text='')
        self.name2label.configure(text='')
        self.name3label.configure(text='')
        self.folder1label.configure(text='')
        self.folder2label.configure(text='')
        self.folder3label.configure(text='')
        self.res1label.configure(text='')
        self.res2label.configure(text='')
        self.res3label.configure(text='')
        
    def MeasureAll(self):

        self.stopped = False

        #check if imlist exists
        try:
            bla  = self.imlist
        except AttributeError:
            print 'nothing loaded yet...loading'
            self.loadfolder()
            self.saveImage(showMessage=False)
            self.showresults()

        if self.imlist == []:
            next(self.pathwalker)
            self.loadfolder()

        while self.atEnd == False and self.stopped==False:
            self.nextButton(showMessage=False)
            time.sleep(.1)
            self.saveImage(showMessage=False)
            self.showresults()
            time.sleep(.1)


    def showHistogram(self):
        results = self.reshapeList()
        fly.showHistGUI(results) 

    def convertim(self,image):
        '''
        convert image to Tkinter displayable image
        '''
        tkim1 = Image.fromarray(image[:,:,::-1],'RGB')
        tkim2 = tkim1.resize(self.tkimsize)
        tkim = ImageTk.PhotoImage(tkim2)
        return tkim

    def saveImage(self,showMessage=False):
        '''
        save the image with the contour in it
        '''
        print 'saving image'
        smallim = cv2.resize(self.imright,self.tkimsize)
        curpath = self.pathEntry.get()
        curname = self.name+'_contour.png'
        cv2.imwrite(curpath+'/'+curname, smallim)
        
        if showMessage == True:
            self.show_messg('Image saved at \n '+curpath+'/'+curname)


    def get_im_list(self):
        '''
        get an image list from the folder
        load and display image
        '''

        #empty imlist
        self.imlist = [] 

        try:
            os.chdir(self.impath)
        except TypeError:
            self.show_messg('Error in path')

        filelist = list(next(os.walk('.'))[2])
        for f in filelist:
            for formt in self.formats:
                if f.count(formt) > 0:
                    self.imlist.append(f)
        self.imlist.sort()

        #if len(self.imlist)==0:
        #    self.show_messg('No image in folder of allowed types \n \
        #            '+str(self.formats))
        if self.verbose:
            print 'list: ', self.imlist


    def getarea_default(self):
        '''
        This function calls a series of functions from the fly eye module, and
        returns the processed image and area of measured eye
        '''
        if self.adap.get() == 1:
            print 'adaptive is on \n'
            self.imright, cont, self.area = fly.do_default2(
                    self.imleft,tr=self.tr1,magicnumber=self.magicnumber,\
                    show='none',cvx=self.convex.get())
        else:
            print 'adaptive is off \n'
            self.imright, cont, self.area = fly.defaultFast(
                self.imleft,tr=self.tr1,magicnumber=self.magicnumber,\
                        show='none',cvx=self.convex.get())

    def recalculate(self):
        '''
        recalculate results with updated magic number
        '''
        self.magicnumber = float(self.magicEntry.get())
        self.getarea_default()
        self.results[-1][-1] = self.area
        self.showresults()
        self.saveImage()
        self.saved = False


    def saveQuitButton(self):
        #self.save()
        print 'saved status = ', self.saved
        if not self.saved:
            self.showSaveQuit()
        else:
            self.close_window()


    def save(self):
        #massage the data into the desired shape and format
        presaveable = self.reshapeList()
        #Function toCSV does the actual saving
        self.toCSV(presaveable)
        #save content, empty list
        self.saved = True

    
    def saveAs(self):

        
        #default options
        filename = self.results[0][0]+'-to-'+self.results[-1][0]+'.csv'
        savepath = self.saveEntry.get()

        #get destination file
        openfile = tkFileDialog.asksaveasfile(mode='w',defaultextension='.csv',\
                initialfile = filename, initialdir = savepath)

        #reshape the list
        presaveable = self.reshapeList()

        #save at destination
        self.toCSV(presaveable,openedfile = openfile)

        #set flag
        self.saved=True
        
        


    def savequit(self):
        self.save()
        self.close_window()

    def reshapeList(self):
        '''
        Takes the results from the GUI, which are shaped as:
        filename, folder, eyesize
        filenmae, folder, eyesize

        and returns a list of lists, each sublist consisting of a header containing
        the filename, and two columns containing the name of the picture and the
        eyesize

        [
         [[folder1, folder1],[name1, size1],[name2, size2]],
         [[folder2, folder2],[name1, size1],...]
         ....,
         
        ]
        '''

        #TODO tosave -> self.result

        folders = [self.results[i][1] for i in range(len(self.results))]
        folderset = set(folders)

        savelist = []
        flatlist = [val for row in self.results for val in row]

        #Part one: make a list of lists of lists
        for folder in folderset:
            
            curfolderresults = [[folder,folder]]
            #find all eyes in this folder
            for row in range(len(self.results)):
                if self.results[row][1] == folder:
                    curfolderresults.append([self.results[row][0],self.results[row][2]])

            savelist.append(curfolderresults)

        return savelist


    def toCSV(self,savefile, openedfile = ''):
        '''
        takes the list of lists of lists from reshapeList and writes is to csv
        '''


                #number of columns
        cols = len(savefile)

        #find the longest list of lists
        maxlen=0
        for lis in savefile:

            if len(lis) > maxlen:
                maxlen = len(lis)

        if openedfile == '':
            ##where to save
            filename = self.results[0][0]+'-to-'+self.results[-1][0]+'.csv'
            savepath = self.saveEntry.get()

            #open the file
            f = open(savepath+'/'+filename,'wb')
        else:
            f = openedfile
        
        writer = csv.writer(f)

        for i in range(maxlen):
            curRow = []
            for j in range(cols): 
                #if the current list is shorter
                if i >= len(savefile[j]):
                    curRow.extend([0,0])
                else:
                    curRow.extend(savefile[j][i])
            writer.writerow(curRow)
        f.close()

        #interact with user
        if openedfile == '':
            self.show_messg('saved the file at \n'+(savepath+filename)[-80:])


    def close_window(self):
        os.chdir(self.instpath)
        self.quit()
        self.master.destroy()

#=============================COPY TO FULL VERSION LATER

    def showleft(self):
        '''
        shows the left image
        '''
        self.tkim0 = self.convertim(self.imleft)
        self.im1label.configure(image=self.tkim0)


    def showright(self):
        '''
        shows the right image
        '''
        self.tkim1 = self.convertim(self.imright)
        self.im2label.configure(image=self.tkim1)


    def loadim(self):
        '''
        loads an image
        '''
        fullpath = self.impath+"/"+self.imlist[self.imi]
        if self.verbose:
            print 'in loadim: loading : ', fullpath
        im = cv2.imread(fullpath)
        return im


    def showresults(self):
        #show left, show right, show resultlabels
        if self.verbose:
            print 'in showresults'
        
        self.showleft()
        self.showright()
        #show result la'''bels
        self.name1label.configure(text=self.name)
        self.folder1label.configure(text=self.impath[-50:])
        self.res1label.configure(text=self.area)
       
        if len(self.results)>1:
            self.name2label.configure(text=self.results[-2][0])
            self.folder2label.configure(text=self.results[-2][1][-50:])
            self.res2label.configure(text=self.results[-2][2])
 
        if len(self.results)>2:
            self.name3label.configure(text=self.results[-3][0])
            self.folder3label.configure(text=self.results[-3][1][-50:])
            self.res3label.configure(text=self.results[-3][2])
        
        #actually show the stuff
        Frame.update(self)


    def show_messg(self, mstext):
        '''
        second popup message
        '''
        msswin2 = Toplevel()
        typotext = Button(msswin2, text=mstext,
                   bg="blue", fg="yellow",
                   activebackground="red", activeforeground="white",
                   padx=msswin2.winfo_screenwidth()/6,
                   pady=msswin2.winfo_screenheight()/6,
                   command=msswin2.destroy)
        typotext.grid(row=0,column=0)


    def showSaveQuit(self):
        '''
        shows explanation and 2 choices
        '''
        choicewin = Toplevel(self)
        explanation = Label(choicewin, text='Data is not saved',fg='red')
        explanation.grid(row=0,column=1,columnspan=3)
        choice1 = Button(choicewin,text='save',command=self.save)
        choice2 = Button(choicewin,text='save and then quit',command=self.savequit)
        choice3 = Button(choicewin,text='quit NOW!',command=self.close_window)
        quitbutton2 = Button(choicewin,text='close this window',command=choicewin.destroy)
        choice1.grid(row=1,column=0)
        choice2.grid(row=1,column=1)
        choice3.grid(row=1,column=2)
        quitbutton2.grid(row=2,column=2)
        print 'at the end of show_savequit'

    def showStats(self):

        results = reshapeList(self.results)
        pass
        #TODO finish this
        for sublist in results:
            pass
            
    

#================================WRAPPERS FOR FLYEYE==============================
#Functions that do nothing but pass class properties to the flyeye module
#Needed for all functions that get called directly by the UI
# Consider using inline Lambda functions

    def colorfastUI(self):
        '''
        UI wrapper for the colorfast function in the flyeye module
        '''
        self.cls =  fly.colorfast(self.im, self.vh, self.vs, self.vv, self.cspace)

def main():
    root = Tk()
    #root["bg"] = 'black'
    root.title('Fly eye surface measuring')
    interface = Interface(root)
    root.mainloop()

if __name__ == '__main__':
    main()
