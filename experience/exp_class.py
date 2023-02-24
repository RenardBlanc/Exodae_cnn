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


# File managment 
import os

# Import from other class
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from data.pre_process import *

# Tensorflow
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Reshape, GlobalAveragePooling1D, LeakyReLU, Input,concatenate, PReLU
from tensorflow.keras.utils import to_categorical,plot_model
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.models import load_model

# Sklearn
from sklearn.metrics import classification_report, confusion_matrix

# Time managment
import time

# Data manipulation
import numpy as np
from itertools import combinations_with_replacement

'''
Librairies
'''

class models():
    '''
    Defintion of the neuronal network Modele
    '''
    def mod_1(nb_coord,nb_class,nb_neurones = 128,fct_activation='LeakyReLU'):
        model_1 = Sequential()
        # hidden layer
        model_1.add(Dense(nb_neurones, input_shape=(nb_coord,), activation=fct_activation))
        # output layer
        model_1.add(Dense(nb_class, activation='softmax'))
        return model_1
    
    def mod_2(nb_coord,nb_class,nb_filter_1 = 64, kernel_size_1 = 3, pool_size_1 = 3,nb_filter_2 = 100, kernel_size_2 = 3,fct_activation = 'relu',nb_neurone = 128,drop1 = 0.5):
        model_2 = Sequential()
        model_2.add(Reshape((nb_coord, 1), input_shape=(nb_coord,)))
        model_2.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(MaxPooling1D(pool_size=pool_size_1))
        model_2.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(GlobalAveragePooling1D())
        model_2.add(Dropout(drop1))
        model_2.add(Dense(nb_neurone, activation=fct_activation))
        model_2.add(Dense(nb_class, activation='softmax'))
        return model_2
    
    def mod_3(nb_coord,nb_class,nb_filter_1 = 128, kernel_size_1 = 3, pool_size_1 = 3,drop1  =0.1, nb_filter_2 = 256, kernel_size_2 = 3, pool_size_2 = 3,drop2 = 0.25, nb_filter_3 = 512, kernel_size_3 = 3, drop3 = 0.5,drop4 = 0.5,fct_activation = 'relu',nb_neurone = 1024):
        model_3 = Sequential()
        model_3.add(Reshape((nb_coord, 1), input_shape=(nb_coord,)))
        model_3.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(MaxPooling1D(pool_size=pool_size_1))
        model_3.add(Dropout(drop1))
        model_3.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(MaxPooling1D(pool_size=pool_size_2))
        model_3.add(Dropout(drop2))
        model_3.add(Conv1D(filters=nb_filter_3, kernel_size=kernel_size_3, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(Conv1D(filters=nb_filter_3, kernel_size=kernel_size_3, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(GlobalAveragePooling1D())
        model_3.add(Dropout(drop3))
        model_3.add(Dense(nb_neurone, activation=fct_activation))
        model_3.add(Dropout(drop4))
        model_3.add(Dense(nb_class, activation='softmax'))
        return model_3

    def mod_4(nb_coord,nb_class,nb_filter_1 = 64, kernel_size_1 = 3, pool_size_1 = 3, nb_drop1 =0.5,nb_filter_2 = 64, kernel_size_2 = 3, pool_size_2 = 3, nb_drop2 =0.5,nb_filter_3 = 64, kernel_size_3 = 3, pool_size_3 = 3, nb_drop3 =0.5,fct_activation = 'relu',nb_neurone = 126):
        model_4 = Sequential()
        model_4.add(Reshape((nb_coord, 1), input_shape=(nb_coord,)))
        # Head 1
        inputs1 = Input(shape = (nb_coord,1))
        conv1 = Conv1D(filters = nb_filter_1, kernel_size = kernel_size_1, activation=fct_activation)(inputs1)
        drop1 = Dropout(nb_drop1)(conv1)
        pool1 = MaxPooling1D(pool_size=pool_size_1)(drop1)
        flat1 = Flatten()(pool1)
        # Head 2
        inputs2 = Input(shape = (nb_coord,1))
        conv2 = Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation)(inputs2)
        drop2 = Dropout(nb_drop2)(conv2)
        pool2 = MaxPooling1D(pool_size=pool_size_2)(drop2)
        flat2 = Flatten()(pool2)
        # Head 3
        inputs3 = Input(shape = (nb_coord,1))
        conv3 = Conv1D(filters=nb_filter_3, kernel_size=kernel_size_3, activation=fct_activation)(inputs3)
        drop3 = Dropout(nb_drop3)(conv3)
        pool3 = MaxPooling1D(pool_size=pool_size_3)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(nb_neurone, activation=fct_activation)(merged)
        outputs = Dense(nb_class, activation='softmax')(dense1)
        model_4 = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        return model_4

class training():

    def launch(nb_model,M,Re,BATCH_SIZE = 50,EPOCHS = 1000,plot = False):
        x_train,y_train,x_test,y_test,nb_class = pre_processing.data_CNN(M,Re)
        
        # one-hot-encoding of our labels
        y_train_hot = to_categorical(y_train, nb_class) 
        y_test_hot = to_categorical(y_test, nb_class)

        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]
        
        def link_model(nb_model):
            if nb_model == 1:
                model = models.mod_1
            elif nb_model == 2:
                model = models.mod_2
            elif nb_model == 3:
                model = models.mod_3
            elif nb_model == 4:
                model = models.mod_4
            return model 

        model = link_model(nb_model)
        modele= model(nb_coord,nb_class)
        modele.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

        if nb_model == 4 :
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                verbose=1)
            # Print confusion matrix for training data
            y_pred_train = modele.predict([x_train,x_train,x_train])
        else : 
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                verbose=1)
            # Print confusion matrix for training data
            y_pred_train = modele.predict(x_train)
        # Take the class with the highest probability from the train predictions
        max_y_pred_train = np.argmax(y_pred_train, axis=1)
        print(classification_report(y_train, max_y_pred_train))

        if plot == 1 or plot == True:
            plt.figure(figsize=(12, 8))
            plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
            plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
            plt.plot(history.history['loss'], 'r--', label='Loss of training data')
            plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
            plt.title('Model Accuracy and Loss')
            plt.ylabel('Accuracy and Loss')
            plt.xlabel('Training Epoch')
            plt.ylim(0)
            plt.legend()
            plt.show()
        
    def mod_1(M,Re,number_of_epochs_test = 1000,toTest = ['neurone','epoque','paquet','fct_activation']):
        # On définis le numero du modèle pour pouvoir plus
        # facilement reconnaitre les experiences
        nb_mod = '1'
        # On crée le fichier de résultat si besoin
        mainFileName = deco.createMainFile_CNN('results',bigfolder = 'experience')
        # On importe les données de tests et d'entrainement
        x_train,y_train,x_test,y_test,nb_class = pre_processing.data_CNN(M,Re)
        # one-hot-encoding of our labels
        y_train_hot = to_categorical(y_train, nb_class) 
        y_test_hot = to_categorical(y_test, nb_class)
        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]

        # Liste des meilleurs hyper-paramètre
        best_param = []
        # On lance les tests suivants les paramètres qu'on cherche
        # à tester
        if 'neurone' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'neurone'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")

            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)
            
            # Test de différents nombre de neurones sur 
            # la couche entierement connectée
            nb_neurone_list = [4,16,64,128,256,512,1024,2048,4096]
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            fct_activation = 'LeakyReLU'
            nb_test = len(nb_neurone_list)
            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []
            
            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(nb_neurone_list,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for nb_neurone in nb_neurone_list:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} Nombre d'époque(s) \n* Nombre de paquet : {}\n Fonction d'activation: {}\n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation=fct_activation)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1
            
            report_file.write("\n------------------------------------------------------------------------------\n")
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(nb_neurone_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            best_param.append(nb_neurone_list[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'epoque' in toTest:
            # Test de différents nombre d'époque
            paramToTest = 'epoque'
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")

            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)
            
            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            nb_neurone =128
            if number_of_epochs_test == 1000:
                number_of_epochs_list = [1,10,100,200,500,1000,1500,2000,3000,5000,10000,15000,20000,50000]
                nb_test = len(nb_neurone_list)
            else : 
                number_of_epochs_list = [number_of_epochs_test]
                nb_test = len(nb_neurone_list)
            batch_size = 50
            
            nb_test = len(number_of_epochs_list)
            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(number_of_epochs_list,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for number_of_epochs in number_of_epochs_list:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} Nombre d'époque(s) \n* Nombre de paquet : {}\n Fonction d'activation: {}\n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation=LeakyReLU)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])

                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])

                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1
            report_file.write("\n------------------------------------------------------------------------------\n")
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(number_of_epochs_list)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            best_param.append(number_of_epochs_list[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'paquet' in toTest:
            # Test de différents nombre d'époque
            paramToTest = 'paquet'
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")

            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            fct_activation = 'LeakyReLU'
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size_list = [1,5,10,20,50,100,200,400,500,1000]
            nb_test = len(batch_size_list)

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(batch_size_list,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for batch_size in batch_size_list:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} Nombre d'époque(s) \n* Nombre de paquet : {}\n Fonction d'activation: {}\n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation= fct_activation)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])

                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1       

            report_file.write("\n------------------------------------------------------------------------------\n")
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(batch_size_list)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            best_param.append(batch_size_list[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        fct_activation = [ 'relu' , 'sigmoid' , 'softmax' , 'softplus' , 'softsign' , 'tanh' , 'selu' , 'elu' , 'exponential' ]

        if 'fct_activation' in toTest:
            # Test de différents nombre d'époque
            paramToTest = 'fct_activ'
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")

            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur les fonctions d'activations pour le modèle {} \n------------------------------------------------------------------------------\n".format(nb_mod)
            report_file.write(text_start)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            fct_activation_list = ['relu' , 'sigmoid' , 'softmax' , 'softplus' , 'softsign' , 'tanh' , 'selu' , 'elu' , 'exponential', PReLU, LeakyReLU ]

            nb_test = len(fct_activation_list)

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(batch_size_list,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for fct_activation in fct_activation_list:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} Nombre d'époque(s) \n* Nombre de paquet : {}\n Fonction d'activation: {}\n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation= fct_activation)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])

                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1       

            report_file.write("\n------------------------------------------------------------------------------\n")
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(fct_activation_list)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            best_param.append(fct_activation_list[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        # On lance l'enregistrement du meilleur modèle 
        report_file_path = os.path.join(mainFileName, 'best'+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")
        text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur les fonctions d'activations pour le modèle {} \n------------------------------------------------------------------------------\n".format(nb_mod)
        report_file.write(text_start)
        nb_neurone = best_param[0]
        number_of_epochs = best_param[1]
        batch_size = best_param[2]
        fct_activation = best_param[3]
        text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'entrainement pour le meilleur modèle \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {}\n* fonction d'activation : {}\n ".format(nb_mod,nb_class,nb_neurone,number_of_epochs,batch_size,fct_activation)
        report_file.write(text_start)
        text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
        report_file.write(text_start)
         # Enregistre le modèle
        dossierparent = deco.createMainFile_CNN('model',bigfolder = 'experience')
        nom_fichier = os.path.join(dossierparent,'mod{}_M{}_Re{}'.format(nb_mod,M,Re))
        
        # Définir les callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(filepath=nom_fichier + '.h5', monitor='val_loss', save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * (0.9 ** epoch))
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        csv_logger = CSVLogger(nom_fichier + '.csv')
        # Définition du modèle
        model = models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation=fct_activation)
        model.compile(loss='categorical_crossentropy',
                                    optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train,
                            y_train_hot,
                            batch_size=batch_size,
                            epochs=number_of_epochs,
                            validation_split=0.2,
                            verbose=0,
                            callbacks = [early_stopping, model_checkpoint, learning_rate_scheduler, reduce_lr_on_plateau, csv_logger])
        report_file.write("\nEntrainement terminé\n")
        results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
        report_file.write(results)
        history_test = model.evaluate(x_test,y_test_hot)
        report_file.write("La précision du modèle avec les données de test est {}\n".format(history_test[1]))
        return model


    def mod_2(M,Re,number_of_epochs_test = 1000,toTest = ['neurone','epoque','paquet','filtre','noyau','pool','dropout','fct_activation']):
        '''
        Cette fonction permet de lancer les tests sur les hyper-paramètre 
        du modèle 2 et crée des rapport d'experience pour chaque test
        - *Nombre de neurone*
        - *Nombre d’époque*
        - *Nombre de paquets*
        - *Nombre de filtre (2)*
        - *Nombre de noyau (2)*
        - *Pool size*
        - *Dropout*
        - *Fonction d’activation*

        Et elle enregistre le meilleur modèle sous le nom : best_mod2.h5 
        '''
        # On définis le numéro du mode pour faciliter 
        # la lecture des rapport par la suite
        nb_mod = '2'
        mainFileName = deco.createMainFile_CNN('results',bigfolder = 'experience')
        x_train,y_train,x_test,y_test,nb_class = pre_processing.data_CNN(M,Re)
        # one-hot-encoding of our labels
        y_train_hot = to_categorical(y_train, nb_class) 
        y_test_hot = to_categorical(y_test, nb_class)

        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]
        best_param = []
        if 'neurone' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'neurone'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de neurones sur 
            # la couche entierement connectée
            list_to_Test = [4,16,64,128,256,512,1024,2048,4096] 
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            nb_filter_2 = 100 
            kernel_size_2 = 3
            fct_activation = 'relu'
            drop1 = 0.5
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0 
            for nb_neurone in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1            
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            
            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'epoque' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'epoque'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre d'époque 
            nb_neurone =128
            if number_of_epochs_test == 1000:
                list_to_Test = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
            else : 
                list_to_Test = [number_of_epochs_test]
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            nb_filter_2 = 100 
            kernel_size_2 = 3
            drop1 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for number_of_epochs in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin

                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        if 'paquet' in toTest :
            # On définis le paramétre à tester pour les logs
            paramToTest = 'paquet'
            
            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de batch 
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            list_to_Test = [5,10,20,50,100,200,400,500,1000]
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            nb_filter_2 = 100 
            kernel_size_2 = 3
            drop1 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for batch_size in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1             
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)
            report_file.close()

        if 'filtre' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'filtre'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de filtre
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            filtr_list = [2,4,8,16,32,64,128,256,512]
            list_to_Test = list(combinations_with_replacement(filtr_list,2))
            kernel_size_1 = 3
            pool_size_1 = 3
            kernel_size_2 = 3
            fct_activation = 'relu'
            nb_test = len(list_to_Test)
            drop1 = 0.5
            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for filtrs in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = filtrs[0], kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = filtrs[2], kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1             
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(filtr_list)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()
            
        if 'noyau' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'noyau'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de noyau
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 64
            kernel_list = [1,2,3,4,5,6]
            list_to_Test = list(combinations_with_replacement(kernel_list,2))
            pool_size_1 = 3
            nb_filter_2 = 100 
            kernel_size_2 = 3
            drop1 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for kernels in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernels[0], pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernels[1],fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1                  
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(kernel_list)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        if 'pool' in toTest : 
            # On définis le paramétre à tester pour les logs
            paramToTest = 'pool'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de pool size
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            list_to_Test = [1,2,3,4,5,6]
            nb_filter_2 = 100 
            kernel_size_2 = 3
            drop1 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for pool_size_1 in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])

                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1   
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        if 'dropout' in toTest :
            # On définis le paramétre à tester pour les logs
            paramToTest = 'dropout'

            #On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de pool size
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            nb_filter_2 = 100 
            kernel_size_2 = 3
            list_to_Test = [0.05,0.1,0.2,0.25,0.3,0.5]
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for drop1 in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])

                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1   
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)
            report_file.close()

        if 'fct_activation' in toTest :
             # On définis le paramétre à tester pour les logs
            paramToTest = 'fct_activation'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur les fonctions d'activation pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de pool size
            nb_neurone =128
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            nb_filter_2 = 100 
            kernel_size_2 = 3
            drop1 = 0.5
            list_to_Test = ['relu' ,'sigmoid' ,'softmax' ,'softplus' ,'softsign' ,'tanh' ,'selu' ,'elu' ,'exponential','PReLU' , 'LeakyReLU' ]
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for fct_activation in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes : \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* fonction activation : {} \n ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = fct_activation,nb_neurone = nb_neurone,drop1 = drop1)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])

                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1   
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n' )
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        # On entre les meilleurs paramètres déterminés par tests
        nb_neurone = best_param[0]
        number_of_epochs = best_param[1]
        batch_size = best_param[2]
        nb_filter_1 = best_param[3]
        kernel_size_1 = best_param[5]
        pool_size_1 = best_param[7]
        nb_filter_2 = best_param[4] 
        kernel_size_2 = best_param[6]
        drop1 = best_param[8]
        fct_activation = best_param[9]

        # On lance l'enregistrement du meilleur modèle 
        report_file_path = os.path.join(mainFileName, 'best'+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")
        text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le meilleurs pour le modèle {} \n------------------------------------------------------------------------------\n".format(nb_mod)
        report_file.write(text_start)

        text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'entrainement pour le modèle {} \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_mod,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2)
        report_file.write(text_start)
        text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
        report_file.write(text_start)
        # Enregistre le modèle
        dossierparent = deco.createMainFile_CNN('model',bigfolder = 'experience')
        nom_fichier = os.path.join(dossierparent,'mod{}_M{}_Re{}'.format(nb_mod,M,Re))
        
        # Définir les callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(filepath=nom_fichier + '.h5', monitor='val_loss', save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * (0.9 ** epoch))
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        csv_logger = CSVLogger(nom_fichier + '.csv')

        model= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = 'relu',nb_neurone = nb_neurone,drop1=drop1)
        model.compile(loss='categorical_crossentropy',
                                    optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train,
                            y_train_hot,
                            batch_size=batch_size,
                            epochs=number_of_epochs,
                            validation_split=0.2,
                            verbose=0,
                            callbacks=[early_stopping, model_checkpoint, learning_rate_scheduler, reduce_lr_on_plateau, csv_logger])
        report_file.write("\nEntrainement terminé\n")
        results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
        report_file.write(results)
        history_test = model.evaluate(x_test,y_test_hot)
        report_file.write("La précision du modèle avec les données de test est {}\n".format(history_test[1]))
        return model
    
   
    def mod_3(M,Re,number_of_epochs_test = 500,toTest = ['neurone','epoque','paquet','filtre','noyau','pool','dropout','fct_activation']):
        '''
        Cette fonction permet de lancer les tests sur les hyper-paramètre 
        du modèle 3 et crée des rapport d'experience pour chaque test
        - *Nombre de neurone*
        - *Nombre d’époque*
        - *Nombre de paquets*
        - *Nombre de filtre (3)*
        - *Nombre de noyau (3)*
        - *Pool size (2)*
        - *Dropout (4)*
        - *Fonction d’activation*

        Et elle enregistre le meilleur modèle sous le nom : best_mod2.h5 
        '''

       # On définis le numéro du mode pour faciliter 
        # la lecture des rapport par la suite
        nb_mod = '3'
        mainFileName = deco.createMainFile_CNN('results',bigfolder = 'experience')
        x_train,y_train,x_test,y_test,nb_class = pre_processing.data_CNN(M,Re)
        # one-hot-encoding of our labels
        y_train_hot = to_categorical(y_train, nb_class) 
        y_test_hot = to_categorical(y_test, nb_class)

        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]
        best_param = []
        # On crée un fichier de rapport pour chaque test

        if 'neurone' in toTest:
        
            # On définis le paramétre à tester pour les logs
            paramToTest = 'neurone'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de neurones sur 
            # la couche entierement connectée
            list_to_Test = [4,16,64,128,256,512,1024,2048,4096] 
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 128
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.1
            nb_filter_2 = 256
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.25
            nb_filter_3 = 512
            kernel_size_3 = 3
            drop3 = 0.5
            drop4 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for nb_neurone in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1 
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'epoque' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'epoque'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de epoques 
            nb_neurone = 1024
            if number_of_epochs_test==500:
                list_to_Test = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
            else : 
                list_to_Test = [number_of_epochs_test]
            batch_size = 50
            nb_filter_1 = 128
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.1
            nb_filter_2 = 256
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.25
            nb_filter_3 = 512
            kernel_size_3 = 3
            drop3 = 0.5
            drop4 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for number_of_epochs in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1 
            
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'paquet' in toTest : 

            # On définis le paramétre à tester pour les logs
            paramToTest = 'paquet'
            
            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)


            # Test de différents nombre de paquet 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            list_to_Test = [5,10,20,50,100,200,400,500,1000]
            nb_filter_1 = 128
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.1
            nb_filter_2 = 256
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.25
            nb_filter_3 = 512
            kernel_size_3 = 3
            drop3 = 0.5
            drop4 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for batch_size in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1             
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)
            report_file.close()

        if 'filtre' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'filtre'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de filtres 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            filtr_list = [64,128,256,512,1024]
            list_to_Test = combinations_with_replacement(filtr_list,3)
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.1
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.25
            kernel_size_3 = 3
            drop3 = 0.5
            drop4 = 0.5
            fct_activation = 'relu'
            nb_test = len(list(list_to_Test))
            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for filtrs in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = filtrs[0], kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = filtrs[2], kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = filtrs[3], kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1 
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(filtr_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Les meilleurs paramètres est {} \n".format(best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()
            
        if 'noyau' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'noyau'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de noyaux 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 128
            kernel_size_list = [1,2,3,4]
            list_to_Test = combinations_with_replacement(kernel_size_list,3)
            pool_size_1 = 3
            drop1 = 0.1
            nb_filter_2 = 256
            pool_size_2 = 3
            drop2 = 0.25
            nb_filter_3 = 512
            drop3 = 0.5
            drop4 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for kernels in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernels[0], pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernels[1], pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernels[2], drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(kernel_size_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'pool' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'pool'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de noyaux 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 128
            kernel_size_1 = 3
            drop1 = 0.1
            pool_list = [1,2,3,4,5,6]
            list_to_Test = combinations_with_replacement(pool_list,2)
            nb_filter_2 = 256
            kernel_size_2 = 3
            drop2 = 0.25
            nb_filter_3 = 512
            kernel_size_3 = 3
            drop3 = 0.5
            drop4 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for pool in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 =kernel_size_1, pool_size_1 = pool[0],drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool[1],drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(kernel_size_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'dropout' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'dropout'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de drop 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 128
            kernel_size_1 = 3
            pool_size_1 =3
            drop_list = [0.05,0.1,0.2,0.25,0.3,0.5]
            list_to_Test = combinations_with_replacement(drop_list,4)
            nb_filter_2 = 256
            kernel_size_2 = 3
            pool_size_2 = 3
            nb_filter_3 = 512
            kernel_size_3 = 3
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0

            for drops in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drops[0], nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drops[1], nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drops[2],drop4 = drops[3],fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(drop_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][3])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-4],best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)
            report_file.close()


        if 'fct_activation' in toTest : 

            # On définis le paramétre à tester pour les logs
            paramToTest = 'fct_activation'
            
            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)


            # Test de différents nombre de paquet 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 128
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.1
            nb_filter_2 = 256
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.25
            nb_filter_3 = 512
            kernel_size_3 = 3
            drop3 = 0.5
            drop4 = 0.5
            list_to_Test = ['relu' ,'sigmoid' ,'softmax' ,'softplus' ,'softsign' ,'tanh' ,'selu' ,'elu' ,'exponential','PReLU' , 'LeakyReLU' ]
            nb_test = len(list_to_Test)

            

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for fct_activation in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit(x_train,
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate(x_test,y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1             
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)
            report_file.close()

        # On entre les meilleurs paramètres déterminés par tests
        # Test de différents nombre de paquet 
        nb_neurone = best_param[0]
        number_of_epochs = best_param[1]
        batch_size = best_param[2]
        nb_filter_1 = best_param[3]
        kernel_size_1 =best_param[6]
        pool_size_1 = best_param[9]
        drop1 = best_param[11]
        nb_filter_2 = best_param[4]
        kernel_size_2 =best_param[7]
        pool_size_2 = best_param[10]
        drop2 = best_param[12]
        nb_filter_3 = best_param[5]
        kernel_size_3 = best_param[8]
        drop3 = best_param[12]
        drop4 = best_param[14]
        list_to_Test = best_param[15]

        # On lance l'enregistrement du meilleur modèle 
        report_file_path = os.path.join(mainFileName, 'best'+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")
        text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le meilleurs pour le modèle {} \n------------------------------------------------------------------------------\n".format(nb_mod)
        report_file.write(text_start)
        
        text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes :\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n* fonction activation : {} \n".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4,fct_activation)
        report_file.write(text_start)
        text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
        report_file.write(text_start)
         # Enregistre le modèle
        dossierparent = deco.createMainFile_CNN('model',bigfolder = 'experience')
        nom_fichier = os.path.join(dossierparent,'mod{}_M{}_Re{}'.format(nb_mod,M,Re))
        
        # Définir les callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(filepath=nom_fichier + '.h5', monitor='val_loss', save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * (0.9 ** epoch))
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        csv_logger = CSVLogger(nom_fichier + '.csv')
        
        model= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = fct_activation,nb_neurone = nb_neurone)
        model.compile(loss='categorical_crossentropy',
                                    optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train,
                            y_train_hot,
                            batch_size=batch_size,
                            epochs=number_of_epochs,
                            validation_split=0.2,
                            verbose=0,
                            callbacks = [early_stopping, model_checkpoint, learning_rate_scheduler, reduce_lr_on_plateau, csv_logger])
        report_file.write("\nEntrainement terminé\n")
        results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
        report_file.write(results)
        history_test = model.evaluate(x_test,y_test_hot)
        report_file.write("La précision du modèle avec les données de test est {}\n".format(history_test[1]))
        return model

    def mod_4(M,Re,number_of_epochs_test = 500,toTest = ['neurone','epoque','paquet','filtre','noyau','pool','dropout','fct_activation']):
        '''
        Cette fonction permet de lancer les tests sur les hyper-paramètre 
        du modèle 3 et crée des rapport d'experience pour chaque test
        - *Nombre de neurone*
        - *Nombre d’époque*
        - *Nombre de paquets*
        - *Nombre de filtre (3)*
        - *Nombre de noyau (3)*
        - *Pool size (3)*
        - *Dropout (3)*
        - *Fonction d’activation*

        Et elle enregistre le meilleur modèle sous le nom : best_mod2.h5 
        '''

        # On définis le numéro du mode pour faciliter 
        # la lecture des rapport par la suite
        nb_mod = '4'
        mainFileName = deco.createMainFile_CNN('results',bigfolder = 'experience')
        x_train,y_train,x_test,y_test,nb_class = pre_processing.data_CNN(M,Re)
        # one-hot-encoding of our labels
        y_train_hot = to_categorical(y_train, nb_class) 
        y_test_hot = to_categorical(y_test, nb_class)

        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]
        best_param = []
        # On crée un fichier de rapport pour chaque test
        
        if 'neurone' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'neurone'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de neurones sur 
            # la couche entierement connectée
            list_to_Test = [4,16,64,128,256,512,1024,2048,4096] 
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.5
            nb_filter_2 = 64
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.5
            nb_filter_3 = 64
            kernel_size_3 = 3
            pool_size_3 = 3
            drop3 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for nb_neurone in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1            
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()


        if 'epoque' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'epoque'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de epoques 
            nb_neurone = 1024
            if number_of_epochs_test ==500:
                list_to_Test = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
            else: 
                list_to_Test = [number_of_epochs_test]
            batch_size = 50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.5
            nb_filter_2 = 64
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.5
            nb_filter_3 = 64
            kernel_size_3 = 3
            pool_size_3 = 3
            drop3 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for number_of_epochs in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1                           
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'paquet' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'paquet'
            
            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de paquet 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            list_to_Test = [5,10,20,50,100,200,400,500,1000]
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.5
            nb_filter_2 = 64
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.5
            nb_filter_3 = 64
            kernel_size_3 = 3
            pool_size_3 = 3
            drop3 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for batch_size in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)
            report_file.close()

        if 'filtre' in toTest: 
            paramToTest = 'filtre'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de filtres 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            filtr_list = [64,128,256,512,1024]
            list_to_Test = combinations_with_replacement(filtr_list,3)
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.5
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.5
            kernel_size_3 = 3
            pool_size_3 = 3
            drop3 = 0.5
            fct_activation = 'relu'
            nb_test = len(list(list_to_Test))

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0

            for filters in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = filters[0], kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = filters[1], kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = filters[2], kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1 
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(filtr_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Les meilleurs paramètres est {} \n".format(best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        # Test de différents nombre de noyaux
        if 'noyau' in toTest: 
            # On définis le paramétre à tester pour les logs
            paramToTest = 'noyau'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)
            # -------------
            # Définition des paramètres
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            kernel_size_list = [1,2,3,4,5,6]
            list_to_Test = combinations_with_replacement(kernel_size_list,3)
            nb_filter_1 = 64
            pool_size_1 = 3
            drop1 = 0.5
            nb_filter_2 = 64
            pool_size_2 = 3
            drop2 = 0.5
            nb_filter_3 = 64
            pool_size_3 = 3
            drop3 = 0.5
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for kernels in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernels[0], pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernels[1], pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernels[2], pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(kernel_size_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        if 'pool' in toTest: 
            # On définis le paramétre à tester pour les logs
            paramToTest = 'pool'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)
            # -------------
            # Test de différents nombre de paquet 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            list_to_Test = [5,10,20,50,100,200,400,500,1000]
            nb_filter_1 = 64
            kernel_size_1 = 3
            drop1 = 0.5
            nb_filter_2 = 64
            kernel_size_2 = 3
            drop2 = 0.5
            nb_filter_3 = 64
            kernel_size_3 = 3
            drop3 = 0.5
            fct_activation = 'relu'
            pool_size_list = [1,2,3,4,5,6]
            list_to_Test = combinations_with_replacement(pool_size_list,3)
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for pool in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool[0], nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool[1], nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 =kernel_size_3, pool_size_3 = pool[2], nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(pool_size_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()

        if 'dropout' in toTest: 
            # On définis le paramétre à tester pour les logs
            paramToTest = 'dropout'

            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)
            # -------------
            # Test de différents nombre de paquet 
            # Test de différents nombre de drop 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size = 50
            drop_list = [0.05,0.1,0.2,0.25,0.3,0.5]
            list_to_Test = combinations_with_replacement(drop_list,3)
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            nb_filter_2 = 64
            kernel_size_2 = 3
            pool_size_2 = 3
            nb_filter_3 = 64
            kernel_size_3 = 3
            pool_size_3 = 3
            fct_activation = 'relu'
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i=0
            for drop in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop[0],nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop[1],nb_filter_3 = nb_filter_3, kernel_size_3 =kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop[2],fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience\n"
            report_file.write(text_start)
            report_file.write(str(drop_list)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')

            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)][0])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][1])
            best_param.append(list_to_Test[np.argmax(accurancy_test)][2])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-3],best_param[-2],best_param[-1])
            report_file.write(text_start)

            report_file.close()
        
        if 'fct_activation' in toTest:
            # On définis le paramétre à tester pour les logs
            paramToTest = 'fct_activation'
            
            # On crée un fichier txt qui correspond au rapport des
            # expériences sur cette hyper-paramètre
            report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
            report_file = open(report_file_path, "a")
            
            text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le nombres de {} pour le modèle {} \n------------------------------------------------------------------------------\n".format(paramToTest,nb_mod)
            report_file.write(text_start)

            # Test de différents nombre de paquet 
            nb_neurone = 1024
            number_of_epochs = number_of_epochs_test
            batch_size =  50
            nb_filter_1 = 64
            kernel_size_1 = 3
            pool_size_1 = 3
            drop1 = 0.5
            nb_filter_2 = 64
            kernel_size_2 = 3
            pool_size_2 = 3
            drop2 = 0.5
            nb_filter_3 = 64
            kernel_size_3 = 3
            pool_size_3 = 3
            drop3 = 0.5
            list_to_Test = ['relu' ,'sigmoid' ,'softmax' ,'softplus' ,'softsign' ,'tanh' ,'selu' ,'elu' ,'exponential','PReLU' , 'LeakyReLU' ]
            nb_test = len(list_to_Test)

            # Définition des listes de resultats
            accurancy_train = []
            loss_train = []
            accurancy_val_train = []
            loss_val_train = []
            accurancy_test = []
            loss_test = []

            text_start = "Les tests à effectuer sont : {} \nLe nombre total de test est de :{}\n".format(list_to_Test,nb_test)
            report_file.write(text_start)

            text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
            report_file.write(text_start)
            i = 0
            for fct_activation in list_to_Test:
                text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience {}/{} avec les données suivantes:\n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n* fct_activation : {} ".format(i,nb_test,nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3,fct_activation)
                report_file.write(text_start)
                start = time.perf_counter() # temps de début de l'entrainement 
                modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
                modele.compile(loss='categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
                history = modele.fit([x_train,x_train,x_train],
                                    y_train_hot,
                                    batch_size=batch_size,
                                    epochs=number_of_epochs,
                                    validation_split=0.2,
                                    verbose=0)
                report_file.write("\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] L'entrainement est terminé, les resultats sont les suivants:\n")
                results = "* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_train.append(history.history['accuracy'][-1])
                loss_train.append(history.history['loss'][-1])
                accurancy_val_train.append(history.history['val_accuracy'][-1])
                loss_val_train.append(history.history['val_loss'][-1])
                # -------------
                history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
                # Temps ecoulé
                end = time.perf_counter()  # temps de fin
                report_file.write("\nTest terminé\n")
                results = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La validation avec les données de tests est terminées :\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
                report_file.write(results)
                # -------------
                # Ajout au listes
                # -------------
                accurancy_test.append(history_test[1])
                loss_test.append(history_test[0])
                # -------------
                report_file.write("----------------------------------------------------------------------\n")
                minute = round(end-start) // 60
                secondes = round(end-start) % 60
                report_file.write(f"Temps total écoulé durant cette experience:  {minute} m et {secondes} s.\n")
                report_file.write("\n------------------------------------------------------------------------------\n")
                i +=1               
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience \n"
            report_file.write(text_start)
            report_file.write(str(list_to_Test)+ '\n')
            report_file.write(str(accurancy_train) + '\n')
            report_file.write(str(loss_train)+ '\n')
            report_file.write(str(accurancy_val_train)+ '\n')
            report_file.write(str(loss_val_train)+ '\n')
            report_file.write(str(accurancy_test)+ '\n')
            report_file.write(str(loss_test)+ '\n')
            
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(list_to_Test[np.argmax(accurancy_test)])
            report_file.write(text_start)

            best_param.append(list_to_Test[np.argmax(accurancy_test)])
            text_start = "\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Le meilleur paramètre est {} \n".format(best_param[-1])
            report_file.write(text_start)
            report_file.close()
        
        # On entre les meilleurs paramètres déterminés par tests
        # Test de différents nombre de paquet 
        nb_neurone = best_param[0]
        number_of_epochs = best_param[1]
        batch_size = best_param[2]
        nb_filter_1 =best_param[3]
        kernel_size_1 =best_param[6]
        pool_size_1 = best_param[9]
        drop1 =best_param[12]
        nb_filter_2 =best_param[4]
        kernel_size_2 =best_param[7]
        pool_size_2 = best_param[10]
        drop2 = best_param[13]
        nb_filter_3 = best_param[5]
        kernel_size_3 = best_param[8]
        pool_size_3 = best_param[11]
        drop3 = best_param[14]
        list_to_Test = best_param[15]

        start = time.perf_counter() # temps de début de l'entrainement 
        # Enregistre le modèle
        dossierparent = deco.createMainFile_CNN('model',bigfolder = 'experience')
        nom_fichier = os.path.join(dossierparent,'mod{}_M{}_Re{}'.format(nb_mod,M,Re))
        
        # On lance l'enregistrement du meilleur modèle 
        report_file_path = os.path.join(mainFileName, 'best'+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")
        text_start = "------------------------------------------------------------------------------\n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Expériences sur le meilleurs pour le modèle {} \n------------------------------------------------------------------------------\n".format(nb_mod)
        report_file.write(text_start)
        text_start = "\nDonnées d'entrainements :\n------------------------------------------------------------------------------\n{}\n{}\n{}\n{}\n".format(x_train,y_train,x_test,y_test)
        report_file.write(text_start)
        # Définir les callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(filepath=nom_fichier + '.h5', monitor='val_loss', save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * (0.9 ** epoch))
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        csv_logger = CSVLogger(nom_fichier + '.csv')
        
        model= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = fct_activation,nb_neurone = nb_neurone)
        model.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
        history = model.fit([x_train,x_train,x_train],
                            y_train_hot,
                            batch_size=batch_size,
                            epochs=number_of_epochs,
                            validation_split=0.2,
                            verbose=0,
                            callbacks = [early_stopping, model_checkpoint, learning_rate_scheduler, reduce_lr_on_plateau, csv_logger])
        report_file.write("\nEntrainement terminé\n")
        results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
        report_file.write(results)
        history_test = model.evaluate(x_test,y_test_hot)
        report_file.write("La précision du modèle avec les données de test est {}\n".format(history_test[1]))

    def launch_test(nb_mod,M,Re,number_of_epochs_test = 1000):

        if nb_mod == 1:
            training.mod_1(M,Re,number_of_epochs_test = number_of_epochs_test)
        elif nb_mod == 2 :
            training.mod_2(M,Re,number_of_epochs_test = number_of_epochs_test)
        elif nb_mod == 3 :
            training.mod_3(M,Re,number_of_epochs_test = number_of_epochs_test)
        elif nb_mod == 4 : 
            training.mod_4(M,Re,number_of_epochs_test = number_of_epochs_test)
        else :
            lg.error('Le modèle demandé n\'existe pas')

class read_test():
    # Fonction qui permet de lire les résultats des tests
    def txt2list(txt):
            list = txt.split(', ')
            list[0] =list[0].replace('[','')
            list[-1] =list[-1].replace(']','')
            list = [float(elem) for elem in list]  
            return list

    def data(nb_mod,paramToTest,M,Re):
        mainFileName = os.path.join('experience','results')
        report_file_path = os.path.join(mainFileName, paramToTest+'mod'+nb_mod+'_M{}_Re{}.txt'.format(M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test

    def initial_data(nb_mod,M,Re):
        mainFileName = os.path.join('experience','results')
        report_file_path = os.path.join(mainFileName, r'best'+'mod'+nb_mod+'_M{}_Re{}.txt'.format(Mach,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Données d'entrainements" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]
            x_train = read_test.txt2list(datafile[i+2])
            y_train = read_test.txt2list(datafile[i+3])
            x_test = read_test.txt2list(datafile[i+4])
            y_test = read_test.txt2list(datafile[i+5])
        except:
            lg.error(error)
        
        return x_train,y_train,x_test,y_test

    def get_best_param(nb_mod,M,Re):
        '''
        Fonction qui permet de lire les résultats des tests
        Elle renvoie les meilleurs paramètres pour chaque expérience
        Dans cet ordre : neurone, epoque, paquet, filtre, noyau, pool, dropout, fct_activation
        '''
        hyper_param = ['neurone','epoque','paquet','filtre','noyau','pool','dropout','fct_activation']

        if nb_mod == 1:
            hyper_param = ['neurone','epoque','paquet','fct_activation']
        elif nb_mod == 2:
            nb_filtr = 2
            nb_noyau = 2
            nb_pool = 1
        elif nb_mod == 3:
            nb_filtr = 3
            nb_noyau = 3
            nb_drop = 4
            nb_pool = 2
        elif nb_mod == 4:
            nb_filtr = 3
            nb_noyau = 3
            nb_drop = 3
            nb_pool = 3
        else:
            error
        
        # Listes des meilleurs paramètres
        best_param = []
        
        for type_param in hyper_param:
            text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
            lg.info(text)

            if type_param in ['neurone','epoque','paquet','fct_activation']:
                nb_param = 1
            elif type_param == 'filtre':
                nb_param = nb_filtr
            elif type_param == 'noyau':
                nb_param = nb_noyau
            elif type_param == 'dropout':
                nb_param = nb_drop
            elif type_param == 'pool':
                nb_param =  nb_pool

            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_filtre(nb_mod,M,Re)
            # On crée la liste des combinaison
            filtr_comb = list(combinations_with_replacement(data,nb_param))
            index_max = accurancy_test.index(max(accurancy_test))
            for i in range(nb_param):
                best_param.append(int(filtr_comb[index_max][i]))

        return best_param

    def show_results(nb_mod,Mach,Re,plot_loss=True):
        # Chargement des données de test
        x_train,y_train,x_test,y_test = read_test.initial_data(nb_mod,Mach,Re)

        # Chargement du modèle
        # Chemin vers le fichier contenant le modèle enregistré
        model_path = r'experience/model/mod1_M{}_Re{}.h5'.format(Mach,Re)
        model = load_model(model_path)

        if plot_loss : 
            # Chemin vers le fichier CSV créé par CSVLogger
            csv_path = 'mod{}_M{}_Re{}.csv'.format(nb_mod,Mach,Re)
            # Lecture du fichier CSV avec pandas
            data = pd.read_csv(csv_path)

            # Affichage des courbes de perte et de précision sur les données d'entraînement et de validation
            plt.plot(data['loss'], label='train_loss')
            plt.plot(data['val_loss'], label='val_loss')
            plt.plot(data['accuracy'], label='train_acc')
            plt.plot(data['val_accuracy'], label='val_acc')
            plt.legend()
            plt.show()

        # Prédictions sur les données de test
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        print(conf_matrix)

        # Rapport de classification
        class_report = classification_report(y_test, y_pred_classes)
        print(class_report)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        Mach = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        for i in range(1,5):
            training.launch_test(i,Mach,Re,number_of_epochs_test = 1000)
    elif len(sys.argv) == 4:
        Mach = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        nb_mod = int(sys.argv[3])
        training.launch_test(nb_mod,Mach,Re,number_of_epochs_test = 1000)
    else:
        raise Exception(
            'Entrer <Nb_Mach> <Nb_Re> <Nb_Model>')