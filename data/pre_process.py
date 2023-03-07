'''
---------------------Pre-Processing and Label extraction---------------------
This module presents different classes and functions in order 
to asses the Pre-processing:
* Scrapping of the Data
* Data set sampling and cleaning
And the Label extraction:
* Lift and Drag coefficient calculation with Xfoil
* Shape Area calculation
* Caracteritic point calculation

Created: 02/02/2023
Updated: 02/02/2023
@Auteur: Ilyas Baktache
'''

'''
Librairies
'''

# Log
import logging as lg
lg.basicConfig(level=lg.INFO)  # Debug and info
from datetime import datetime

# Folder and string
import os

# Data manipulation
from scipy import interpolate
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as snc
import jenkspy as jk


#Import scrapping module
from data.scrapping import *


'''
Classes
'''

class deco():
    def createMainFile_CNN(mainFileName,bigfolder = 'data'):
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
    
class le():
    '''
    Class that contains all the function to extract the label
    '''

    def polar_Re_M(Re,M):

        # Create Airfoil Polar main File if it doesn't exist
        mainFileName = "Airfoil_Polar"  
        mainFileName = utils.createMainFile(mainFileName)
        # Create Airfoil Polar Mach main File if it doesn't exist 
        mainFileName = utils.createMainFile('M_{}'.format(M),bigfolder = mainFileName)
        # Create Airfoil Polar Mach-Reynolds main File if it doesn't exist 
        mainFileName = utils.createMainFile('Re_{}'.format(Re),bigfolder = mainFileName)

        # Scrap all airfoil name in AirfoilTools.com Database
        all_airfoils = scrap.airfoils_name()

        if M == 0:
            # We directly use the data of Airfoil Tools from M = 0 
            scrap.airfoils_polar(all_airfoils,Re,mainFileName)
        else :
            if not os.path.isdir(r'data/Airfoil_Coordinate'):
                # Scrap and save locally coordinate of airfoils
                scrap.airfoils_coordinate(all_airfoils)
                
            # In the case that all the airfoil name from the website 
            # couldn't be scrapped we take only the name of 
            # data that we have
            all_airfoils = os.listdir(r'data/Airfoil_Coordinate') 

            # List of coordinate airfoils that couldn't be scrapped
            airfoil_polar_not = []
            # List all the airfoils name
            n = len(all_airfoils)
            
            # Threading Data
            num_thread = 250 # number of thread
            i0 = 0 # index of the frist page of the multithreading
            i1 = num_thread  # index of the last page of the multithreading
            # Start of threadings
            while i0<n:
                threadears = []
                for i in range(i0, i1):
                    try:
                        name_thread = all_airfoils[i].replace('.dat','')
                        t = threading.Thread(target= le.polar, args=(name_thread,Re,M,airfoil_polar_not))
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
                lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} airfoil data couldn't be processed in Xfoil:\n {}".format(len(airfoil_polar_not),airfoil_polar_not))
            nb_polar_file = len(os.listdir(mainFileName))
            lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {}({}%) airfoils polar for Re = {} were downloaded and saved locally".format(nb_polar_file,int(nb_polar_file/n*100),Re))

    def allPolar(Re_list=[50000,100000,200000,500000,100000],M_list=[0,0.25,0.5,0.75,0.9]):
        '''
        Function that return a polar of all the airfoil in the folder 
        Airfoil_Coordinate for the Re and M wanted
        '''
        if type(M_list)==list:
            for M in M_list:
                if type(Re_list)==list:
                    for Re in Re_list:
                        le.polar_Re_M(Re,M)
                else:
                    le.polar_Re_M(Re_list,M)
        else : 
            if type(Re_list)==list:
                for Re in Re_list:
                    le.polar_Re_M(Re,M_list)
            else:
                le.polar_Re_M(Re_list,M_list)
      
class format():
    '''
    Create an unique format for data and label
    '''
    def raw_format(dir=r"data/Airfoil_Coordinate"):
        nb_coord_x = []
        nb_coord_y = []
        airfoils = os.listdir(dir)
        for airfoil in airfoils:
            name = airfoil.replace('.dat','')
            x,y = utils.coordFile2list(name)
            if x[0] != 1 :
                nb_coord_x.append(name)
            if x[-1] != 1 :
                nb_coord_y.append(name)
        print(len(nb_coord_x),nb_coord_y)
    
    def diff_extra_intra(name,dir=r"data/Airfoil_Coordinate"):

        '''
        Permet de distinguer les coordonnées de l'extrados
        et de l'intrados. 
        '''
        i_extra = 0
        x,y = utils.coordFile2list(name,mainFileName = dir)
        for i in range(len(x)-1):
            if np.abs(x[i])>np.abs(x[i+1]):
                i_extra = i+1
        x_extra = x[:i_extra+1]
        y_extra = y[:i_extra+1]
        x_intra = x[i_extra+1:]
        y_intra = y[i_extra+1:]
        return x_extra,y_extra,x_intra,y_intra
    
    def indice_leading_trailing_edges(x_extra,x_intra,x_rang_LE,x_rang_TE):

        i_extra_TE = []
        i_extra_LE = []
        for i in range(len(x_extra)):
            if x_extra[i]<=x_rang_LE:
                i_extra_LE.append(i)
            if x_extra[i]>=x_rang_TE:
                i_extra_TE.append(i)

        i_intra_TE = []
        i_intra_LE = []
        for i in range(len(x_intra)):
            if x_intra[i]<=x_rang_LE:
                i_intra_LE.append(i)
            if x_intra[i]>=x_rang_TE:
                i_intra_TE.append(i)

        return min(i_extra_LE),max(i_extra_TE),max(i_intra_LE),min(i_intra_TE)

    def rawtointer(name, dir=r"data/Airfoil_Coordinate",nb_point = 30, nb_LE = 20, nb_TE = 10,x_rang_LE = 0.15, x_rang_TE = 0.75):
        '''
        Function that take the raw airfoil data coordinate and return
        '''

        x_extra,y_extra,x_intra,y_intra = format.diff_extra_intra(name,dir=dir)
        i_LE_extra,i_TE_extra,i_LE_intra,i_TE_intra = format.indice_leading_trailing_edges(x_extra,x_intra,x_rang_LE,x_rang_TE)

        # Interpolation extrados LE
        x_extrados_LE = np.linspace(0,x_rang_LE,nb_LE,endpoint = True)[::-1][1:]
        f_extrados_LE= interpolate.interp1d(x_extra[i_LE_extra:], y_extra[i_LE_extra:], fill_value="extrapolate", kind="quadratic")
        y_extrados_LE = f_extrados_LE(x_extrados_LE)

        # Interpolation extrados body
        x_extrados_body = np.linspace(x_rang_LE,x_rang_TE,nb_point,endpoint = True)[::-1][:]
        f_extrados_body= interpolate.interp1d(x_extra[i_TE_extra:i_LE_extra+1], y_extra[i_TE_extra:i_LE_extra+1], fill_value="extrapolate", kind="quadratic")
        y_extrados_body = f_extrados_body(x_extrados_body)

        # Interpolation extrados TE
        x_extrados_TE = np.linspace(x_rang_TE,1,nb_TE,endpoint = True)[::-1]
        f_extrados_TE= interpolate.interp1d(x_extra[:i_TE_extra+1], y_extra[:i_TE_extra+1], fill_value="extrapolate", kind="quadratic")
        y_extrados_TE = f_extrados_TE(x_extrados_TE)

        # Interpolation Intrados LE
        x_intrados_LE = np.linspace(0,x_rang_LE,nb_LE,endpoint = True)[1:]
        f_intrados_LE= interpolate.interp1d(x_intra[:i_LE_intra+1], y_intra[:i_LE_intra+1], fill_value="extrapolate", kind="quadratic")
        y_intrados_LE = f_intrados_LE(x_intrados_LE)

        # Interpolation Intrados Body
        x_intrados_body = np.linspace(x_rang_LE,x_rang_TE,nb_point,endpoint = True)
        f_intrados_body= interpolate.interp1d(x_intra[i_LE_intra:i_TE_intra+1], y_intra[i_LE_intra:i_TE_intra+1], fill_value="extrapolate", kind="quadratic")
        y_intrados_body = f_intrados_body(x_intrados_body)

        # Interpolation Intrados TE
        x_intrados_TE = np.linspace(x_rang_TE,1,nb_TE,endpoint = True)
        f_intrados_TE= interpolate.interp1d(x_intra[i_TE_intra:], y_intra[i_TE_intra:], fill_value="extrapolate", kind="quadratic")
        y_intrados_TE = f_intrados_TE(x_intrados_TE)

        x_inter = list(x_extrados_TE) + list(x_extrados_body) + list(x_extrados_LE) + list(x_intrados_LE) + list(x_intrados_body) + list(x_intrados_TE)
        y_inter = list(y_extrados_TE) + list(y_extrados_body) + list(y_extrados_LE) + list(y_intrados_LE) + list(y_intrados_body) + list(y_intrados_TE)

        #x_camb = np.divide(np.addition(x_extrados_LE,x_intrados_LE),2) + np.divide(np.addition(x_extrados_body,x_intrados_body),2) + np.divide(np.addition(x_extrados_body,x_intrados_body),2)
        #y_camb = np.divide(np.addition(y_extrados_LE,y_intrados_LE),2) + np.divide(np.addition(y_extrados_body,y_intrados_body),2) + np.divide(np.addition(y_extrados_body,y_intrados_body),2)

        return x_inter,y_inter

    def coordinate(dir=r"data/Airfoil_Coordinate",nb_point = 30, nb_LE = 20, nb_TE = 10):
        if not os.path.exists(dir):
            all_airfoils = scrap.airfoils_name()
            scrap.airfoils_coordinate(all_airfoils)
            le.allPolar(Re_list=[50000,100000,200000,500000,1000000],M_list=0)
        airfoils = os.listdir(dir)
        marchepas = []
        all_y = []
        nom_profil = []
        for airfoil in airfoils:
            name = airfoil.replace('.dat','')
            try:
                x_inter,y_inter = format.rawtointer(name,dir=dir, nb_point = nb_point, nb_LE =nb_LE , nb_TE = nb_TE)
                all_y.append(np.array(y_inter).T)
                nom_profil.append(name)
            except: 
                marchepas.append(name)
        return np.array(x_inter).T, np.matrix(all_y).T,nom_profil,marchepas

class pre_processing():
    def save_data():
        # On récupère les données de polaire
        x,ally,nom_profil,marchepas = format.coordinate()
        # On cherche les données de polaire pour un nombre de Mach nul et 
        # des nombres de Reynolds allant de 50000 à 1000000
        M = 0
        Re_list=[50000,100000,200000,500000,1000000]
        # On s'assure que les données de polaire sont disponibles pour tous
        # les profils
        if not os.path.exists('data/Airfoil_Polar'):
                le.allPolar(Re,0)
        # On note dans cette liste les finesses maximales
        finesse_max = np.zeros((len(nom_profil),len(Re_list)))
        no_data_all = [] 
        for j in range(len(Re_list)):
            Re = Re_list[j]
            # Certaines données de polaire ne sont pas disponible pour tous
            # les profils
            no_data = [] 
            for i in range(len(nom_profil)):
                name = nom_profil[i]
                # Ici on choisit alpha = 0
                try :
                    alpha,cL,cD,cDp,cM = utils.polarFile2list(name,M,Re)
                    cL = np.array(cL)
                    cD = np.array(cD)
                    finesse = cL/cD
                    finesse_max[i,j] = np.max(finesse)
                except:
                    no_data.append(name)
            no_data_all.append(no_data)
        finesse_max = finesse_max.round(1).T

        # M = 0, Re = 50000
        ally_0_50000 = ally.copy()
        nom_profil_0_50000 = nom_profil.copy()
        finesse_max_0_50000 = list(finesse_max[0])
        z = [False for _ in range(len(nom_profil_0_50000))]
        for nom in no_data_all[0]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_50000.pop(index)
            nom_profil_0_50000.pop(index)
        ally_0_50000 = ally_0_50000.compress(np.logical_not(z), axis = 1)

        # M = 0, Re = 100000
        ally_0_100000 = ally.copy()
        nom_profil_0_100000 = nom_profil.copy()
        finesse_max_0_100000 = list(finesse_max[1])
        z = [False for _ in range(len(nom_profil_0_100000))]
        for nom in no_data_all[1]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_100000.pop(index)
            nom_profil_0_100000.pop(index)
        ally_0_100000 = ally_0_100000.compress(np.logical_not(z), axis = 1)


        # M = 0, Re = 200000
        ally_0_200000 = ally.copy()
        nom_profil_0_200000 = nom_profil.copy()
        finesse_max_0_200000 = list(finesse_max[2])
        z = [False for _ in range(len(nom_profil_0_200000))]
        for nom in no_data_all[2]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_200000.pop(index)
            nom_profil_0_200000.pop(index)
        ally_0_200000 = ally_0_200000.compress(np.logical_not(z), axis = 1)


        # M = 0, Re = 500000
        ally_0_500000 = ally.copy()
        nom_profil_0_500000 = nom_profil.copy()
        finesse_max_0_500000 = list(finesse_max[3])
        z = [False for _ in range(len(nom_profil_0_500000))]
        for nom in no_data_all[3]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_500000.pop(index)
            nom_profil_0_500000.pop(index)
        ally_0_500000 = ally_0_500000.compress(np.logical_not(z), axis = 1)


        # M = 0, Re = 1000000
        ally_0_1000000 = ally.copy()
        nom_profil_0_1000000 = nom_profil.copy()
        finesse_max_0_1000000 = list(finesse_max[4])
        z = [False for _ in range(len(nom_profil_0_1000000))]
        for nom in no_data_all[4]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_1000000.pop(index)
            nom_profil_0_1000000.pop(index)
        ally_0_1000000 = ally_0_1000000.compress(np.logical_not(z), axis = 1)

        finesse_max = [finesse_max_0_50000,finesse_max_0_100000,finesse_max_0_200000,finesse_max_0_500000,finesse_max_0_1000000]
        nom_profil_tt_Re = [nom_profil_0_50000,nom_profil_0_100000,nom_profil_0_200000,nom_profil_0_500000,nom_profil_0_1000000]

        def discretisation_label(nom_profil_Re,finesse_max_Re,nb_class):
            Re_fin = {'nom' : nom_profil_Re, 
                            'finesse_max' : finesse_max_Re}

            df_fin = pd.DataFrame(Re_fin)

            intervalle_finesse_max = jk.jenks_breaks(df_fin['finesse_max'], n_classes=nb_class)
            df_fin['classe'] = pd.cut(df_fin['finesse_max'],
                                bins=intervalle_finesse_max,
                                labels=[i for i in range(1,nb_class+1)],
                                include_lowest=True)
            return df_fin, intervalle_finesse_max

        df_fin, intervalle_finesse_max = discretisation_label(nom_profil_0_50000,finesse_max[0],100)

        def classe2finesse_max(classe,intervalle_finesse_max):
            finesse_max_loc = (intervalle_finesse_max[classe-1] + intervalle_finesse_max[classe])/2
            return np.round(finesse_max_loc,2)

        def finesse_classe(df_fin,intervalle_finesse_max):
            # Cette fonction permet de rajouter dans le dataframe pandas
            # les données de finesse_max associée au classes
            finesse_max_fct = []
            for i in range(len(df_fin['finesse_max'])):
                classe = df_fin['classe'][i]
                finesse_max_fct.append(classe2finesse_max(classe,intervalle_finesse_max))
            
            df_fin['finesse_max_class'] = finesse_max_fct
            return df_fin

        df_fin = finesse_classe(df_fin,intervalle_finesse_max)

        def comparaison_fin_fct_Re(nom_profil_Re,finesse_max_Re,nb_class):
        
            df_fin, intervalle_finesse_max = discretisation_label(nom_profil_Re,finesse_max_Re,nb_class)
            df_fin = finesse_classe(df_fin,intervalle_finesse_max)

            list_err = []
            for i in range(len(df_fin['finesse_max'])):
                finesse_max_reelle = df_fin['finesse_max'][i]
                finesse_max_fct = df_fin['finesse_max_class'][i]
                
                if finesse_max_reelle != 0:
                    err = np.abs((finesse_max_reelle - finesse_max_fct)) / np.abs((finesse_max_reelle))
                else :
                    pass
                list_err.append(err)
            
            return list_err

        def choix_nb_classe(nom_profil_Re,finesse_max_Re,Re):
            index_class = []

            for nb_class in range(10,100):
                try:
                    list_err = comparaison_fin_fct_Re(nom_profil_Re,finesse_max_Re,nb_class)
                    err_max = (np.max(list_err)*100)
                    err_moy = (np.mean(list_err)*100)

                    if err_max <= 50 and err_moy <= 1:
                        index_class.append(nb_class)
                except:
                    pass

            #print('Pour Re = {}, il faut prendre {} classes pour respecter les critères.'.format(Re,index_class[0]))
            return index_class[0]

        # On note alors le nombres de classes nécessaire pour 
        # chaque Re dans une liste
        nb_class_list = []
        for i in range(5):
            nb_class_list.append(choix_nb_classe(nom_profil_tt_Re[i],finesse_max[i],Re_list[i]))
        
        finesse_max_classe = []

        def list_label(nom_profil_Re,finesse_max_Re,nb_class):
            df_fin, intervalle_finesse_max = discretisation_label(nom_profil_Re,finesse_max_Re,nb_class)
            df_fin = finesse_classe(df_fin,intervalle_finesse_max)
            classe_list = list(df_fin['classe'])

            return classe_list

        for i in range(5):
            finesse_max_classe.append(list_label(nom_profil_tt_Re[i],finesse_max[i],nb_class_list[i]))
        
        def split_data(dataset,finesse_max):

            dataset = np.matrix.tolist(dataset.T)
            # Number of data 
            num = len(dataset)

            # Number of each dataset : Train and Test.
            n_train  = int(num*0.7)
            n_test  = int(num*0.3)

            while n_train+n_test !=num:
                n_train+=1
            
            # All the index of the big dataset
            allindex = [i for i in range(num)]

            # Random list of index of the train dataset
            list_train = random.sample(allindex, n_train)
            # List of allindex without the train index
            index_notrain = list(set(allindex)-set(list_train))

            # List of random train index
            list_test = random.sample(index_notrain, n_test)

            x_train = []
            x_test = []

            y_test = []
            y_train = []
            for i in allindex:
                if i in list_train:
                    x_train.append(dataset[i])
                    y_train.append(finesse_max[i])
                elif i in list_test:
                    x_test.append(dataset[i])
                    y_test.append(finesse_max[i])
            return x_train,y_train,x_test,y_test

        def save_Re_data_CNN(dict):
            mainFileName = deco.createMainFile_CNN('post_processed_data_CNN')
            Re = dict['reynoldsNumber']
            name = os.path.join(mainFileName,"Re_0_{}.pickle".format(Re))
            with open(name, "wb") as tf:
                pickle.dump(dict,tf)

        x_train,y_train,x_test,y_test = split_data(ally_0_50000,finesse_max_classe[0])
        dict_0_50000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_50000,
                        'nb_classe' : nb_class_list[0],
                        'reynoldsNumber' : 50000,
                        }
        save_Re_data_CNN(dict_0_50000)

        x_train,y_train,x_test,y_test = split_data(ally_0_100000,finesse_max_classe[1])
        dict_0_100000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_100000,
                        'nb_classe' : nb_class_list[2],
                        'reynoldsNumber' : 100000,
                        }
        save_Re_data_CNN(dict_0_100000)

        x_train,y_train,x_test,y_test = split_data(ally_0_200000,finesse_max_classe[2])
        dict_0_200000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_200000,
                        'nb_classe' : nb_class_list[2],
                        'reynoldsNumber' : 200000,
                        }
        save_Re_data_CNN(dict_0_200000)

        x_train,y_train,x_test,y_test = split_data(ally_0_500000,finesse_max_classe[3])
        dict_0_500000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_500000,
                        'nb_classe' : nb_class_list[3],
                        'reynoldsNumber' : 500000,
                        }
        save_Re_data_CNN(dict_0_500000)
        
        x_train,y_train,x_test,y_test = split_data(ally_0_1000000,finesse_max_classe[4])
        dict_0_1000000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_1000000,
                        'nb_classe' : nb_class_list[4],
                        'reynoldsNumber' : 1000000,
                        }
        
        save_Re_data_CNN(dict_0_1000000)
    
    def get_data_pre_process_CNN(M,Re):
        if not os.path.exists(r'data/post_processed_data_CNN/Re_{}_{}'.format(M,Re)):
            pre_processing.save_data()

        with open(r"data/post_processed_data_CNN/Re_{}_{}.pickle".format(M,Re), "rb") as file:
                dict_ok = pickle.load(file)
        
        return dict_ok

    def data_CNN(M,Re):
        '''
        Fonction qui permet d'avoir accés au données d'entrainenment 
        et de test pour entrainer un réseau de neurone
        '''
        dict_ok = pre_processing.get_data_pre_process_CNN(M,Re)

        # Importe les données d'entrainement
        x_train = np.array(dict_ok['x_train']).astype('float32')
        y_train = np.array(dict_ok['y_train']).astype('float32')

        # Importe les données de test
        x_test = np.array(dict_ok['x_test']).astype('float32')
        y_test = np.array(dict_ok['y_test']).astype('float32')

        # Nombre de classes à définir
        nb_class = dict_ok['nb_classe']

        # one-hot-encoding of our labels
        y_train = y_train - 1
        y_test = y_test - 1

        nb_class = dict_ok['nb_classe']
        return x_train,y_train,x_test,y_test,nb_class
