'''
---------------------Scrapping UIUC Airfoil Database---------------------
This module presents different classes and functions in order 
to scrap, format, organize and save airfoil data locally.

Created: 10/05/2022
Updated: 24/05/2022
@Auteur: Ilyas Baktache
'''
# %%
# Librairies
# -------------------------------------------------------------

# Scrapping
from distutils.log import error
import requests
from bs4 import BeautifulSoup

# Log
import logging as lg
lg.basicConfig(level=lg.INFO)  # Debug and info
from datetime import datetime

# Folder and string
import os

# math and plot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Threading
import threading

#%%
class utils():

    def createMainFile(mainFileName,bigfolder = 'data'):
        if not os.path.isdir(bigfolder):
            if os.name == 'nt':
                os.makedirs(mainFileName) #Windows
            else:
                os.mkdir(bigfolder) # Mac ou linux
        
        mainFileName = os.path.join(bigfolder,mainFileName)
        # Main folder for airfoil data
        if not os.path.isdir(mainFileName):
            if os.name == 'nt':
                os.makedirs(mainFileName) #Windows
            else:
                os.mkdir(mainFileName) # Mac ou linux
        return mainFileName
    
    def link2file(text,nameMainFolder,nameFile,extension = 'txt'):
            '''
            Write Processed data as name.dat in processed_data folder
            '''
            # Get processed coordonnate
            data_file = os.path.join(nameMainFolder,nameFile + '.{}'.format(extension))
            if not os.path.isfile(data_file):
                with open(data_file, 'w') as f:
                    f.write(text)
                f.close()
    
    def coordFile2list(name,mainFileName = r'data/Airfoil_Coordinate'):
        data_file = os.path.join(mainFileName,name+'.dat')
        x = []
        y = []
        with open(data_file, 'r') as f:
            allLines = f.readlines()[1:]
            for line in allLines:
                coord = line.split()
                x.append(float(coord[0]))
                y.append(float(coord[1]))
        return x,y

    def polarFile2list(name,M,Re,mainFileName = r'data/Airfoil_Polar'):
        '''
        Function that convert data from polar file into List
        '''
        mainFileName = os.path.join(mainFileName,'M_'+str(M))
        mainFileName = os.path.join(mainFileName,'Re_'+str(Re))
        data_file = os.path.join(mainFileName,name+'.txt')
        
        # Result list definition
        alpha  = [] # Attack angle (degree)
        cL = [] # Lift coefficient
        cD = [] # Drag coefficient
        cDp = [] # Drag coefficient (pressure)
        cM = [] # Moment coefficient
        with open(data_file, 'r') as f:
            allLines = f.readlines()[12:]
            for line in allLines:
                data = line.split()
                alpha.append(float(data[0]))
                cL.append(float(data[1]))
                cD.append(float(data[2]))
                cDp.append(float(data[3]))
                cM.append(float(data[4]))

        return alpha,cL,cD,cDp,cM

class scrap():

    def airfoils_name():
        '''
        Function that scrap all the airfoil names in the airfoil Tools database
        '''

        # Link with all airfoils from airfoil Tools website
        link_airfoil_name = "http://airfoiltools.com/search/airfoils"
        try :
            # Request and Parse
            main_data_page = requests.get(link_airfoil_name) 
            soup = BeautifulSoup(main_data_page.content, 'html.parser')
            # Define the list of airfoil name
            name_data = []
            for html_td in soup.findAll('td',class_ = "link"):
                link = html_td.findAll('a')[0].get('href')
                name_data.append(link.replace("/airfoil/details?airfoil=",''))
            
            lg.debug("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Airfoil name were scrapped from Airfoil Tools website")
            return name_data

        except Exception as err: 
            lg.error("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Airfoil name couldn't be scrapped from Airfoil Tools website" + err)

    def airfoil_coord(airfoil,mainFileName,airfoil_coord_not,base_link =  "http://airfoiltools.com/airfoil/seligdatfile?airfoil="):
    
        link_airfoil_coor_dat = base_link + airfoil
        try: 
            # Request and Parse
            coordinate_page = requests.get(link_airfoil_coor_dat) 
            # Write airfoil coordinate in a dat file
            utils.link2file(coordinate_page.text,mainFileName,airfoil,extension='dat')
        except:
            airfoil_coord_not.append(airfoil)
            lg.error("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} coordinate couldn't be scrapped from Airfoil Tools website".format(airfoil))
        
    def airfoils_coordinate(all_airfoils):
        '''
        Function that scrap all the airfoil coordinates in the airfoil Tools database
        and save them locally
        '''
        # Create Coordinale Folder
        mainFileName = "Airfoil_Coordinate"        
        mainFileName = utils.createMainFile(mainFileName)
        # List of coordinate airfoils that couldn't be scrapped
        airfoil_coord_not = []
        # List all the airfoils name
        n = len(all_airfoils)
        
        # Threading Data
        num_thread = 50 # number of thread
        i0 = 0 # index of the frist page of the multithreading
        i1 = num_thread  # index of the last page of the multithreading
        # Start of threadings
        while i0<n:
            threadears = []
            for i in range(i0, i1):
                try:
                    name_thread = all_airfoils[i]
                    t = threading.Thread(target= scrap.airfoil_coord, args=(name_thread,mainFileName,airfoil_coord_not))
                    t.start()
                    threadears.append(t)
                except Exception as err:
                    lg.debug("error:".format(err))
            for threadear in threadears:
                threadear.join()
            i0 += num_thread
            i1 += num_thread
            if i1 > n:
                i1 = n
        if len(airfoil_coord_not)>0:
            lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} airfoil data couldn't be processed:\n {}".format(len(airfoil_coord_not),airfoil_coord_not))
        nb_coordinate_file = len(os.listdir(mainFileName))
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {}({}%) airfoils coordinate were downloaded and saved locally".format(nb_coordinate_file,int(nb_coordinate_file/n*100)))

    def airfoil_pol(airfoil,Re,mainFileName,airfoil_polar_not,base_link =  "http://airfoiltools.com/polar/text?polar=xf-"):
        link_polar = base_link + airfoil + '-{}'.format(Re)
        try: 
            # Request and Parse
            coordinate_page = requests.get(link_polar) 
            # Write airfoil polar in a txt file
            utils.link2file(coordinate_page.text,mainFileName,airfoil)
        except:
            airfoil_polar_not.append(airfoil)
            lg.error("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} coordinate couldn't be scrapped from Airfoil Tools website".format(airfoil))        

    def airfoils_polar(all_airfoils,Re,mainFileName):
        '''
        Function that scrap all the airfoil coordinates in the airfoil Tools database
        and save them locally
        '''
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Starting scrapping airfoils polar for M = 0 and Re = {}".format(Re))

        # List of coordinate airfoils that couldn't be scrapped
        airfoil_polar_not = []
        # List all the airfoils name
        n = len(all_airfoils)
        
        # Threading Data
        num_thread = 50 # number of thread
        i0 = 0 # index of the frist page of the multithreading
        i1 = num_thread  # index of the last page of the multithreading
        # Start of threadings
        while i0<n:
            threadears = []
            for i in range(i0, i1):
                try:
                    name_thread = all_airfoils[i]
                    t = threading.Thread(target= scrap.airfoil_pol, args=(name_thread,Re,mainFileName,airfoil_polar_not))
                    t.start()
                    threadears.append(t)
                except Exception as err:
                    lg.debug("error:".format(err))
            for threadear in threadears:
                threadear.join()
            i0 += num_thread
            i1 += num_thread
            if i1 > n:
                i1 = n
        if len(airfoil_polar_not)>0:
            lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} airfoil data couldn't be processed:\n {}".format(len(airfoil_polar_not),airfoil_polar_not))
        nb_polar_file = len(os.listdir(mainFileName))
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {}({}%) airfoils polar for M = 0 and Re = {} were downloaded and saved locally".format(nb_polar_file,int(nb_polar_file/n*100),Re))

class plot():

    def airfoil(name):
        x,y = utils.coordFile2list(name)
        plt.figure(figsize = (12,8))
        plt.plot(x,y)
        plt.title('Airfoil {}'.format(name))
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.show()

    def airfoils(mainFileName = os.path.join('data','Airfoil_Coordinate')):
        
        airfoils = os.listdir(mainFileName)

        coord_x = []
        coord_y = []
        for airfoil in airfoils:
            name = airfoil.replace('.dat','')
            x,y = utils.coordFile2list(name)
            coord_x.append(x)
            coord_y.append(y)
        
        fig,ax = plt.subplots()
        dt = 0.01
        graph, = plt.plot(coord_x[0], coord_y[0] , color="k",lw=1,markersize=3,label='{}'.format(airfoils[0].replace('.dat','')))
        L=plt.legend(loc=1) #Define legend objects

        def init():
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            return graph,

        def animate(i):
            lab = '{}'.format(airfoils[i].replace('.dat',''))
            graph.set_data(coord_x[i], coord_y[i])
            L.get_texts()[0].set_text(lab) #Update label each at frame

            return graph,

        ani = animation.FuncAnimation(fig,animate,frames=np.arange(len(airfoils)),init_func=init,interval=50)
        plt.show()


class analyse():

    def area_shape(name):
        '''
        Function that calculate the area of an airfoil using Gauss’s area formula
        '''
        x,y = utils.coordFile2list(name)
        n = len(x)
        A = 0
        for i in range(1,n-1):
            A += x[i]*(y[i+1]-y[i-1])
        return (1/2)*A

    def finesse(name,M,Re):
        alpha,cL,cD,cDp,cM = utils.polarFile2list(name,M,Re)
        cL = np.array(cL)
        cD = np.array(cD)

        finesse = cL/cD

        return alpha,finesse
        
    def characteristic_point(name,M,Re,class1 = True,val = 'val'):
        '''
        Function that calculate the 4 characteristics points:
        
        Input:
            * airfoil name
            * Re 
            * Type of airplane 
                class1(True) = Turbomachine, planeur / 
                class1(False): à piston, Turbopropulseur
        Output:
            * pD : Point de décrochage
            * pC : Point de chute minimum (if class1)
            * pA : Point d'autonomie maximale
            * pF : Point de distance franchissable maximale sans vent
        Each point is defined as point = [alpha,cL,cD]
        '''
        # Points definition 
        pD = []
        pC = []
        pA = []
        pF = []

        # Get aerodynamics coefficient of the airfoil at Re
        alpha,cL,cD,cDp,cM = utils.polarFile2list(name,M,Re)
        cL = np.array(cL)
        cD = np.array(cD)

        # Point de décrochage
        i_pD = np.argmax(cL)
        pD = [cL[i_pD],alpha[i_pD],cL[i_pD],cD[i_pD]]

        # Preliminary calculation
        cL_cD = cL/cD
        i_fmax = np.argmax(cL_cD)
        cD_cL_3_2 = cD/(cL**(3/2))
        i_cD_cL_3_2= np.nanargmax(cD_cL_3_2)
        cD_sqrtcL = cD/np.sqrt(cL)
        i_cD_sqrtcL = np.nanargmax(cD_sqrtcL)

        if class1:
            pC = [cD_cL_3_2[i_cD_cL_3_2],alpha[i_cD_cL_3_2],cL[i_cD_cL_3_2],cD[i_cD_cL_3_2]]
            pA =  [cL_cD[i_fmax],alpha[i_fmax],cL[i_fmax],cD[i_fmax]]
            pF = [cD_sqrtcL[i_cD_sqrtcL],cL[i_cD_sqrtcL],cD[i_cD_sqrtcL]]
        else : 
            pA =  [cD_cL_3_2[i_cD_cL_3_2],alpha[i_cD_cL_3_2],cL[i_cD_cL_3_2],cD[i_cD_cL_3_2]]
            pF = [cL_cD[i_fmax],alpha[i_fmax],cL[i_fmax],cD[i_fmax]]

        if val == 'val':
            i = 0
        elif val == 'alpha':
            i = 1
        elif val == 'cL':
            i = 2
        elif val == 'cD':
            i = 3
        return pD[i],pC[i],pA[i],pF[i]
# %%

# %%
