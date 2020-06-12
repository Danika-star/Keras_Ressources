
import keras
import keras.backend as K


def tripletLoss_Wrapper(fvSize=256, margin=0.1, verbose=False):
    # This wrapper returns a function that creates the graph to compute the Triplet Loss.
    # This wrapper is necessary to give parameters to the graph-building function, such as <fvSize>, <margin> and <verbose>.
    # This wrapper can be used to implement the loss and to implement the metric.

    def tripletLoss(y_true, y_pred):
        # This function creates a graph that computes the Triplet Loss from <y_pred>.
        # To be compatible with keras, the only parameters this function can have are <y_true> and <y_pred>.

        anchorFV = y_pred[:,0:fvSize]
        if verbose : anchorFV = K.print_tensor(anchorFV, message="anchorFV is: ")

        #positiveFV = y_pred[:,256:512]
        positiveFV = y_pred[:,fvSize:2*fvSize]
        if verbose : positiveFV = K.print_tensor(positiveFV, message="positiveFV is: ")

        # negativeFV = y_pred[:,512:]
        negativeFV = y_pred[:,2*fvSize:3*fvSize]
        if verbose : negativeFV = K.print_tensor(negativeFV, message="negativeFV is: ")

        positiveDistances = K.sqrt(K.sum(K.square(anchorFV - positiveFV), axis=-1))
        # positiveDistances = K.sum(K.square(anchorFV - positiveFV))
        # tmp = K.print_tensor(positiveDistances.shape, message="positiveDistances.shape is: ")
        if verbose : positiveDistances = K.print_tensor(positiveDistances, message="positiveDistances is: ")

        negativeDistances = K.sqrt(K.sum(K.square(anchorFV - negativeFV), axis=-1))
        # negativeDistances = K.sum(K.square(anchorFV - negativeFV))
        if verbose : negativeDistances = K.print_tensor(negativeDistances, message="  is: ")

        # Triplet Loss
        tripletLoss = K.maximum(positiveDistances - negativeDistances + margin, 0)
        # tripletLoss = K.sum(loss, tmp)
        tripletLoss = K.mean(tripletLoss)
        if verbose : tripletLoss = K.print_tensor(loss, message="loss is: ")

        return tripletLoss # The tensor named "tripletLoss"

    return tripletLoss # The nested function named "tripletLoss"


def dual_loss_wrapper(margin=1, fvSize=256, tripletToRegressionRatio=0.5, verbose=False) :
    # This wrapper returns a function that creates the graph to compute the Dual Loss (Triplet Loss & Regression Loss).
    # This wrapper is necessary to give parameters to the graph-building function, such as <fvSize>, <margin> and <verbose>.
    # This wrapper can be used to implement the loss and to implement the metric.

    def dual_loss(y_true, y_pred) :
        # This function creates a graph that computes the Triplet Loss, the Regression MSE and fuses both.
        # To be compatible with keras, the only parameters this function can have are <y_true> and <y_pred>.

        if verbose : y_true = K.print_tensor(y_true, message="y_true : ")
        if verbose : y_pred = K.print_tensor(y_pred, message="y_pred : ")

        tripletLoss = tripletLoss_Wrapper(fvSize=fvSize, margin=margin, verbose=verbose)(y_true, y_pred) # Creating the subgraph that computes the Triplet Loss
        if verbose : tripletLoss = K.print_tensor(tripletLoss, message="tripletLoss : ")

        y_pred_cropped = y_pred[:, 3*fvSize:]
        y_true_cropped = y_true[:, 3*fvSize:]
        # regressionLoss = regressionMSE_Wrapper(fvSize=fvSize, verbose=verbose)(y_true_cropped, y_pred_cropped) # Creating the subgraph that computes the regression Loss
        regressionLoss = regressionCE_Wrapper(fvSize=fvSize)(y_true_cropped, y_pred_cropped) # Creating the subgraph that computes the regression Loss
        if verbose : regressionLoss = K.print_tensor(regressionLoss, message="regressionLoss : ")

        # dualLoss = tripletLoss * regressionLoss # Fusing both losses
        dualLoss = tripletLoss * tripletToRegressionRatio + regressionLoss * (1 - tripletToRegressionRatio) # Fusing both losses
        if verbose : dualLoss = K.print_tensor(dualLoss, message="dualLoss : ")

        return dualLoss # The tensor named "dualLoss"

    return dual_loss # The nested function called "dualLoss"


def regressionMSE_Wrapper():
    # This wrapper returns a function that creates the graph to compute the Regression Mean Squared Error (MSE).
    # This wrapper is necessary to give parameters to the graph-building function, such as <fvSize> and <verbose>.
    # This wrapper can be used to implement the loss and to implement the metric.

    def regressionMSE(y_true, y_pred):
        # This function creates a graph that computes the Regression MSE from <y_true> and <y_pred>.
        # To be compatible with keras, the only parameters this function can have are <y_true> and <y_pred>.

        # regressionLoss = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
        regressionLoss = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
        # regressionLoss = K.print_tensor(regressionLoss, message="regressionLoss : ")
        # regressionLoss = K.mean(regressionLoss)

        return regressionLoss

    return regressionMSE


def regressionCE_Wrapper(dual=False, fvSize=0):
    # This wrapper returns a function that creates the graph to compute the Regression Mean Squared Error (MSE).
    # This wrapper is necessary to give parameters to the graph-building function, such as <fvSize> and <verbose>.
    # This wrapper can be used to implement the loss and to implement the metric.

    def regressionCE(y_true, y_pred):
        # This function creates a graph that computes the Regression MSE from <y_true> and <y_pred>.
        # To be compatible with keras, the only parameters this function can have are <y_true> and <y_pred>.

        if (dual):
            y_pred = y_pred[:, 3*fvSize:]
            y_true = y_true[:, 3*fvSize:]
        regressionLoss = -K.sum(K.log(1 - K.abs(y_true - y_pred)), axis=-1)
        # regressionLoss = K.print_tensor(regressionLoss, message="regressionLoss : ")
        # regressionLoss = K.mean(regressionLoss)


        return regressionLoss

    return regressionCE

