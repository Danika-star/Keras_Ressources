
# This code presents an example of how different parts of Keras-TF work together.

import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
from keras.utils.vis_utils import plot_model
import matplotlib
import numpy as np
import random
import time

import networks.vgg16_dual as vgg16_dual
import networks.densenet_dual as densenet_dual

import class_OHEM
import class_ProductDistances
import class_DataAugmentation
import class_Loss
import class_Callback
import utils


def main():
    # =================================================================================================================================================
    # ============================================================== Step 1 : Parameters ==============================================================
    # =================================================================================================================================================

    trainDual(
        loadFirstXtrainYtrainFromSave = False, # Enables fast run of one epoch with pre-selected triplets, to debug
        dataAugmentationRate = 0, # 0 means all images get DA'ed, 1 means no image get DA'ed.
        maxRandomPositives = 3, # In OHEM, per subset, additionally to the hardest triplet, how many other triplets with positive loss are also selected for training.
        numberOfClasses = 303, 
        batchSize = 256, # MobileNet supports 128. ResNet supports 64.
        numberOfEpochs = 3, 
        fvSize = 1024, # If GAP is used, this needs to be fixed according to the output of the network. DensePose121 : 1024
        margin = 0.1, 
        freezeBase = False,  # If True the underlying model is frozen during Regression training
        maxNumberOfProducts_Train = 100, # -1 keeps all products
    )


def trainDual(loadFirstXtrainYtrainFromSave = False, dataAugmentationRate = 0, maxRandomPositives = 3, numberOfClasses = 303, batchSize = 256, numberOfEpochs = 3, fvSize = 1024, margin = 0.1, freezeBase=False, maxNumberOfProducts_Train=-1):

    # ============================================================================================================================================
    # ============================================================== Step 2 : Model ==============================================================
    # ============================================================================================================================================

    # model_singular, model_triplet, model_classification, model_dual = vgg16_dual.loadModel_vgg16((100, 100, 3), fvSize=fvSize, numberOfClasses=numberOfClasses)
    model_singular, model_triplet, model_classification, model_dual = densenet_dual.loadModel_densenet((100, 100, 3), fvSize=fvSize, numberOfClasses=numberOfClasses, freezeBase=freezeBase)

    # print(model_singular.summary())
    # print(model_triplet.summary())
    # print(model_classification.summary()
    print(model_dual.summary())

    plot_model(model_singular, to_file = "./temp/model_singular.png", show_shapes = True, show_layer_names = True)
    plot_model(model_triplet, to_file = "./temp/model_triplet.png", show_shapes = True, show_layer_names = True)
    plot_model(model_classification, to_file = "./temp/model_classification.png", show_shapes = True, show_layer_names = True)
    plot_model(model_dual, to_file = "./temp/model_dual.png", show_shapes = True, show_layer_names = True)

    # ==============================================================================================================================================
    # ============================================================== Step 3 : Dataset ==============================================================
    # ==============================================================================================================================================

    ohemTrain = class_OHEM.OHEM(numberOfProductsPerTriplet=2)
    ohemTrain.loadDatabase("/home/nicolas/common/datasets/dvsDB/DF1_S2S_v0_cropped/Train/", maxNumberOfProducts=maxNumberOfProducts_Train)

    ohemValidation = class_OHEM.OHEM(labelNames=ohemTrain.labelNames, idToLabel=ohemTrain.idToLabel)
    ohemValidation.loadDatabase("/home/nicolas/common/datasets/dvsDB/DF1_S2S_v0_cropped/Validation/", maxNumberOfProducts = 200)

    ohemTest = class_OHEM.OHEM(labelNames=ohemTrain.labelNames, idToLabel=ohemTrain.idToLabel)
    ohemTest.loadDatabase("/home/nicolas/common/datasets/dvsDB/DF1_S2S_v0_cropped/Test/", maxNumberOfProducts = 200)

    # ================================================================================================================================================
    # ============================================================== Step 4 : Optimizer & Loss & Metrics =============================================
    # ================================================================================================================================================

    # opt = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    # opt = keras.optimizers.Adadelta(lr=1e-4, rho=0.95, decay=0.0)
    # opt = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    # opt = keras.optimizers.SGD(lr=0.001, momentum=0, decay=0, nesterov=False)
    # opt = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.99, nesterov=True)
    opt = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)

    loss = class_Loss.dual_loss_wrapper(margin=margin, fvSize=fvSize, tripletToRegressionRatio=0.1)

    metrics = [
            class_Loss.tripletLoss_Wrapper(fvSize=fvSize, margin=margin),
            class_Loss.regressionCE_Wrapper(dual=True, fvSize=fvSize)
    ]

    # ================================================================================================================================================
    # ============================================================== Step 5 : Callbacks ==============================================================
    # ================================================================================================================================================

    callbacks = []
    # callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto', min_delta=0.001, cooldown=3, min_lr=0))
    # callbacks.append(class_Callback.History_per_batch())
    callbacks.append(keras.callbacks.LearningRateScheduler(class_Callback.learningRateScheduler_Wrapper(lrWarmingLength=2, ratio=5), verbose=1))
    query, label = ohemTrain.getOneDualData(fvSize)
    print("len(query) : ", len(query))
    callbacks.append(class_Callback.DisplayOneInference(query, label))


    # ===============================================================================================================================================
    # ============================================================== Step 6 : Training - Functions ==================================================
    # ===============================================================================================================================================

    def generateXtrainYtrain(model_singular, ohemTrain, saveXtrainYtrain=False):
        ohemTrain.computeFVs(model_singular.predict)

        ohemTrain.mineHardExamples(margin=margin, maxRandomPositives=maxRandomPositives)

        x_train, y_train = ohemTrain.generateDualData(fvSize=fvSize, dataAugmentationFunction=class_DataAugmentation.dataAugmentation)
        # x_train, y_train = ohemTrain.generateDualData(fvSize=fvSize)
        if (saveXtrainYtrain): # Save every once in a while
            np.save(f"temp/x_train_{random.random()}.npy", x_train)
            np.save(f"temp/y_train_{random.random()}.npy", y_train)
        return x_train, y_train

    def loadXtrainYtrain():
        print("Loading x_train / y_train. 60s expected.")
        x_train = np.load("temp/x_train_70kTriplets_DA.npy")
        x_train = [x_train[0], x_train[1], x_train[2]]
        y_train = np.load("temp/y_train_70kTriplets_DA.npy")
        y_train = [y_train[0]]
        print("\tLoading done.")
        return x_train, y_train

    def evaluate(model_singular, ohemValidation):
        # This function evaluate the triplet-loss model on the validation db.
        ohemValidation.computeFVs(model_singular.predict)
        # print("Inter MeanMean :", class_ProductDistances.interNClassMeanMeanDistances(utils.get_productFV(ohemValidation.products)))
        class_ProductDistances.allInterclassDistances(utils.get_productsFV(ohemValidation.products))
        class_ProductDistances.allIntraclassDistances(utils.get_productFV(ohemValidation.products[0]))


    # ===============================================================================================================================================
    # ============================================================== Step 7 : Training ==============================================================
    # ===============================================================================================================================================

    # ================ Train Regression ================
    loss = class_Loss.dual_loss_wrapper(margin=margin, fvSize=fvSize, tripletToRegressionRatio=0) # TripletLoss is ignored
    model_dual.compile(loss=loss, optimizer=opt, metrics=metrics)
    x_train, y_train = generateXtrainYtrain(model_singular, ohemTrain, saveXtrainYtrain=False)
    print("Fitting number 1...")
    H = model_dual.fit(x=x_train, y=y_train, batch_size=batchSize, epochs=numberOfEpochs, callbacks=callbacks, validation_split=0.1)
    print("Fitting number 1 done")
    evaluate(model_singular, ohemValidation)

    # ================ Train Dual ================

    """
    # Unfreeze Model
    def unfreezeModel(model):
        for layer in model.layers:
            if isinstance(layer, keras.engine.training.Model):
                for nestedLayer in layer.layers:
                    if (nestedLayer.trainable == False):
                        nestedLayer.trainable = True
            if (layer.trainable == False):
                layer.trainable = True
    unfreezeModel(model_dual)
    """

    loss = class_Loss.dual_loss_wrapper(margin=margin, fvSize=fvSize, tripletToRegressionRatio=0.5)
    model_dual.compile(loss=loss, optimizer=opt, metrics=metrics)
    print(model_dual.summary())
    x_train, y_train = generateXtrainYtrain(model_singular, ohemTrain, saveXtrainYtrain=False)
    print("fitting number 2...")
    H = model_dual.fit(x=x_train, y=y_train, batch_size=batchSize, epochs=numberOfEpochs, callbacks=callbacks, validation_split=0.1)
    print("fitting number 2 done")
    evaluate(model_singular, ohemValidation)

    exit()

    for fitIndex in range(1000) :
        if (fitIndex == 0): evaluate(model_singular, ohemValidation) # Evaluate before the first training

        # Compile Model - This is done several times in case of some layers being unfrozen in-between .fit() calls
        model_dual.compile(loss=loss, optimizer=opt, metrics=metrics)

        # Compute (x_train, y_train)
        if (loadFirstXtrainYtrainFromSave and (fitIndex == 0)): x_train, y_train = loadXtrainYtrain()
        else: x_train, y_train = generateXtrainYtrain(model_singular, ohemTrain, saveXtrainYtrain=random.random() > 0.9)

        # Train Model
        H = model_dual.fit(x=x_train, y=y_train, batch_size=batchSize, epochs=numberOfEpochs, callbacks=callbacks, validation_split=0.1)

        # Evaluate Model
        evaluate(model_singular, ohemValidation)

        # Save Model
        saveName = f"./temp/model_application_trained_{fitIndex}.h5"
        model_singular.save(saveName)
        print(f"{saveName} saved", end="\n\n\n\n")

        # Unfreeze Model
        for layer in model_dual.layers[::-1] :
            if (layer.trainable == False):
                """
                if ("bn" in layer.name):
                    continue
                """
                layer.trainable = True
                print(f"{layer.name} now trainable")
                break
        # print(model_dual.summary())


if (__name__ == "__main__"):
    main()

