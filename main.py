import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def orbFeatureExtractor(image, orbExtractor):
    # extraction of image features using orb feature extraction method
    keypoints, descriptors = orbExtractor.detectAndCompute(image, None)
    return keypoints, descriptors

def createHistogram(data, clusteringModel):
    # creating histogram for given data
    histogram = np.zeros(len(clusteringModel.cluster_centers_))
    results = clusteringModel.predict(data)
    for index in results:
        histogram[index] += 1
    return histogram

def bagOfVisualWordsMethod(clusterNumber, interval, directory):
    # creates histograms for each image according to clustered data by KMeans and appends it to main list which is bag of visual words list
    # main list is train data for k-nearest neighbor algorithm
    startTime = time.time()
    # orb object
    orb = cv.ORB_create()
    # k-means object
    km = KMeans(n_clusters=clusterNumber)
    tempArray = interval.split("-")
    start = int(tempArray[0])
    end = int(tempArray[1])
    bagOfVisualWords = []
    fileNamesList = os.listdir("./SceneDataset/{}/".format(directory))
    fileNamesList = sorted(fileNamesList, key=lambda item: int(item.split(".")[0]))
    for number in range(start - 1, end):
        # read every image and create histogram from clustered image features and collect them together in bagOfVisualWords list
        image = cv.imread("./SceneDataset/{}/{}".format(directory, fileNamesList[number]), cv.IMREAD_GRAYSCALE)
        kp, descriptor = orbFeatureExtractor(image, orb)
        km_model = km.fit(descriptor)
        if descriptor is not None:
            histogram = createHistogram(descriptor, km_model)
            bagOfVisualWords.append(histogram)
    print("bovw for {} took {} seconds!".format(directory, time.time() - startTime))
    return bagOfVisualWords

def bovwKNNTrain(trainPercent, clusterNumber, train, modelDirectory):
    # creates related numpy arrays(training data) or loads them from saved according to "train" boolean parameter.
    if train:
        directories = ["Bedroom", "Highway", "Kitchen", "LivingRoom", "Mountain", "Office"]
        trainHistograms = []
        trainHistogramsMapping = []
        counter = 0
        for directory in directories:
            # creating  file names list
            numberOfData = len(os.listdir("./SceneDataset/{}/".format(directory)))

            # print(directory)
            """calculating intervals"""
            trainNumber = int(numberOfData * trainPercent / 100)  # 151
            testNumber = numberOfData - trainNumber  # 65
            trainStart = 1  # 1
            trainEnd = trainStart + trainNumber - 1  # 151
            bovw = bagOfVisualWordsMethod(clusterNumber, "{}-{}".format(trainStart, trainEnd), directory)
            indexingArray = np.ones(len(bovw)) * counter
            trainHistograms.extend(bovw)
            trainHistogramsMapping.extend(indexingArray)

            # counting environments,0-> bedroom,1-> highway
            counter += 1


        # saving numpy arrays to use later which is very efficient method
        np.save("./{}/train_{}percent_{}cluster.npy".format(modelDirectory, trainPercent, clusterNumber),
                trainHistograms)
        np.save("./{}/train_{}percent_{}cluster_mapping.npy".format(modelDirectory, trainPercent, clusterNumber),
                trainHistogramsMapping)
        return trainHistograms, trainHistogramsMapping
    else:
        # if we already created numpy arrays which are train data, we only need to load them
        trainHistograms = np.load(
            "./{}/train_{}percent_{}cluster.npy".format(modelDirectory, trainPercent, clusterNumber))
        trainHistogramsMapping = np.load(
            "./{}/train_{}percent_{}cluster_mapping.npy".format(modelDirectory, trainPercent,
                                                                clusterNumber))
        return trainHistograms, trainHistogramsMapping

def bovwKNNTest(trainPercent, trainHistograms, trainHistogramsMapping, neighborNumber, clusterNumber, prepare, modelDirectory):
    # creates test data for bag of visual words with k nearest neighbor according to given "prepare" boolean parameter which controls
    # function in terms of create and save arrays or loading created arrays from .npy file
    if prepare:
        directories = ["Bedroom", "Highway", "Kitchen", "LivingRoom", "Mountain", "Office"]

        testHistograms = []
        testHistogramsMapping = []

        counter = 0
        # creating histograms for testing
        for directory in directories:
            fileNamesList = os.listdir("./SceneDataset/{}".format(directory))
            numberOfData = len(os.listdir("./SceneDataset/{}/".format(directory)))
            trainNumber = int(numberOfData * trainPercent / 100)  # 151
            testNumber = numberOfData - trainNumber  # 65
            # trainStart = 1  # 1
            # trainEnd = trainStart + trainNumber - 1  # 151
            testStart = numberOfData - testNumber - 1  # 152
            testEnd = numberOfData  # 216

            # updating arrays
            bovw = bagOfVisualWordsMethod(clusterNumber, "{}-{}".format(testStart, testEnd), directory)
            testHistograms.extend(bovw)  # was extend
            indexingArray = np.ones(len(bovw)) * counter
            testHistogramsMapping.extend(indexingArray)  # was extend

            counter += 1

        np.save("./{}/test_{}percent_{}cluster.npy".format(modelDirectory, trainPercent, clusterNumber),
                trainHistograms)
        np.save("./{}/test_{}percent_{}cluster_mapping.npy".format(modelDirectory, trainPercent, clusterNumber),
                trainHistogramsMapping)
    else:
        # testing part
        testHistograms = np.load(
            "./{}/test_{}percent_{}cluster.npy".format(modelDirectory, trainPercent, clusterNumber))
        testHistogramsMapping = np.load(
            "./{}/test_{}percent_{}cluster_mapping.npy".format(modelDirectory, trainPercent, clusterNumber))

        success = 0
        expectedClasses = []
        predictedClasses = []
        knn = NearestNeighbors(n_neighbors=neighborNumber)
        knn.fit(trainHistograms)
        for i in range(len(testHistograms)):
            histogram = testHistograms[i]
            # print(histogram)
            distances, results = knn.kneighbors([histogram])

            distances = np.array(distances)
            results = np.array(results)

            tempDict = dict()
            mappingDict = dict()
            selectingScene = np.zeros(6)
            for j in range(len(distances[0])):
                tempDict[distances[0][j]] = results[0][j]

            # sorting dictionary basically
            for key in sorted(tempDict.keys()):
                mappingDict[key] = tempDict[key]

            # selecting nearest neighbor
            for distance in mappingDict.keys():
                # finding predicted scene
                valueClass = int(trainHistogramsMapping[mappingDict[distance]])
                # incrementing relevant class
                selectingScene[valueClass] += 1

            # predicting scene
            matchedClassOccurrence = max(selectingScene)
            matchedClass = None
            for j in range(len(selectingScene)):
                if selectingScene[j] == matchedClassOccurrence:
                    matchedClass = j
            # real scene
            realClass = testHistogramsMapping[i]
            if matchedClass == realClass:
                success += 1
            predictedClasses.append(matchedClass)
            expectedClasses.append(realClass)
        accuracy = round(success / len(predictedClasses) * 100, 2)

        return accuracy

def bovwSVM(clusterNumber, percent, train, modelDirectory):
    # application for bag of visual words approach with linear support vector machines algorithm
    # creates and saves related arrays for dataset if "train" is True and then it applies
    # if "train" is False, then loads related arrays from .npy file and applies
    if train:
        directories = ["Bedroom", "Highway", "Kitchen", "LivingRoom", "Mountain", "Office"]

        # features array(x values)
        trainHistograms = []

        # y values
        trainHistogramsMapping = []
        counter = 0

        for directory in directories:
            numberOfData = len(os.listdir("./SceneDataset/{}/".format(directory)))

            bovw = bagOfVisualWordsMethod(clusterNumber, "{}-{}".format(1, numberOfData), directory)
            indexingArray = np.ones(len(bovw)) * counter
            trainHistograms.extend(bovw)
            trainHistogramsMapping.extend(indexingArray)

            counter += 1

        # saving arrays to use them later if we want to apply only test
        np.save("./{}/train_{}percent_{}cluster.npy".format(modelDirectory, percent, clusterNumber), trainHistograms)
        np.save("./{}/train_{}percent_{}cluster_mapping.npy".format(modelDirectory, percent, clusterNumber),
                trainHistogramsMapping)

    else:
        # load data from saved files to apply only test
        trainHistograms = np.load("./{}/train_{}percent_{}cluster.npy".format(modelDirectory, percent, clusterNumber))
        trainHistogramsMapping = np.load(
            "./{}/train_{}percent_{}cluster_mapping.npy".format(modelDirectory, percent, clusterNumber))

    # random state is 42 to get same shuffled data in each time
    train_x, test_x, train_y, test_y = train_test_split(trainHistograms, trainHistogramsMapping,
                                                        test_size=(100 - percent) / 100, random_state=42)

    # standarization of data
    std = StandardScaler().fit(train_x)
    train_x = std.transform(train_x)
    test_x = std.transform(test_x)

    # linear svm algorithm application
    svm_model = LinearSVC(max_iter=60000)
    svm_model.fit(train_x, train_y)
    score = svm_model.score(test_x, test_y)

    # confusion matrix, uncomment to see confusion matrix
    # matrix = confusion_matrix(test_y, svm_model.predict(test_x))
    # print("confusion matrix for bovw svm :\n", matrix)
    return round(score, 2)

def tinyImageFeaturesKNN(percent, neighbor, modelDirectory, train):
    # tiny images approach with knn algorithm
    # "train" variable works same as before
    #
    if train:
        dataset = []
        datasetMapping = []
        directories = ["Bedroom", "Highway", "Kitchen", "LivingRoom", "Mountain", "Office"]

        counter = 0
        for directory in directories:
            fileNamesList = os.listdir("./SceneDataset/{}/".format(directory))
            fileNamesList = sorted(fileNamesList, key=lambda item: int(item.split(".")[0]))

            numberOfData = len(os.listdir("./SceneDataset/{}/".format(directory)))
            for number in range(0, numberOfData):
                image = cv.imread("./SceneDataset/{}/{}".format(directory, fileNamesList[number]))
                image = cv.resize(image, (16, 16), interpolation=cv.INTER_AREA)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                dataset.append(image.flatten())
                datasetMapping.append(counter)

            counter += 1

        # saving arrays
        np.save("./{}/{}percent_{}neighbor.npy".format(modelDirectory, percent, neighbor), dataset)
        np.save("./{}/{}percent_{}neighbor_mapping.npy".format(modelDirectory, percent, neighbor), datasetMapping)

    else:
        # loading arrays from saved .npy files
        dataset = np.load("./{}/{}percent_{}neighbor.npy".format(modelDirectory, percent, neighbor))
        datasetMapping = np.load("./{}/{}percent_{}neighbor_mapping.npy".format(modelDirectory, percent, neighbor))

    # random state is 42 for same shuffled data
    train_x, test_x, train_y, test_y = train_test_split(dataset, datasetMapping, test_size=(100 - percent) / 100,
                                                        random_state=42)

    # apply standarization to data
    std = StandardScaler().fit(train_x)
    train_x = std.transform(train_x)
    test_x = std.transform(test_x)

    # applying knn algorithm on separated data
    classifier = KNeighborsClassifier(n_neighbors=neighbor)
    classifier.fit(train_x, train_y)

    # prediction, not used in this case
    # predict_y = classifier.predict(test_x)

    # test score
    score = classifier.score(test_x, test_y)

    # confusion matrix, uncomment to see confusion matrix
    # matrix = confusion_matrix(test_y, classifier.predict(test_x))
    # print("confusion matrix for tiny knn :\n", matrix)

    return round(score, 2)

def tinyImageFeaturesSVM(percent, modelDirectory, train):
    # application of tiny image features approach with svm algorithm
    # "train" variable works same as on previous functions
    if train:
        directories = ["Bedroom", "Highway", "Kitchen", "LivingRoom", "Mountain", "Office"]
        dataset = []
        datasetMapping = []
        counter = 0

        for directory in directories:
            numberOfData = len(os.listdir("./SceneDataset/{}/".format(directory)))
            fileNamesList = os.listdir("./SceneDataset/{}/".format(directory))
            fileNamesList = sorted(fileNamesList, key=lambda item: int(item.split(".")[0]))

            for number in range(1, numberOfData):
                image = cv.imread("./SceneDataset/{}/{}".format(directory, fileNamesList[number]))
                image = cv.resize(image, (16, 16), interpolation=cv.INTER_AREA)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                dataset.append(image.flatten())
                datasetMapping.append(counter)

            counter += 1

        np.save("./{}/train_{}percent_.npy".format(modelDirectory, percent), dataset)
        np.save("./{}/train_{}percent_mapping.npy".format(modelDirectory, percent), datasetMapping)

    else:
        dataset = np.load("./{}/train_{}percent_.npy".format(modelDirectory, percent))
        datasetMapping = np.load("./{}/train_{}percent_mapping.npy".format(modelDirectory, percent))

    # random state is 42 to get same shuffled data every time
    train_x, test_x, train_y, test_y = train_test_split(dataset, datasetMapping, test_size=(100 - percent) / 100,
                                                        random_state=42)

    # applying standarization to reduce time complexity
    std = StandardScaler().fit(train_x)
    train_x = std.transform(train_x)
    test_x = std.transform(test_x)

    # applying linear svm algorithm with 40000 iterations
    svm_model = LinearSVC(max_iter=40000)
    svm_model.fit(train_x, train_y)

    # calculating accuracy score of model
    score = svm_model.score(test_x, test_y)

    # confusion matrix, uncomment to see confusion matrix
    # matrix = confusion_matrix(test_y, svm_model.predict(test_x))
    # print("confusion matrix for tiny svm :\n", matrix)

    return round(score, 2)

def trainForDifferentParametersBOVWKNN(clusterValues, percentValues, directory):
    # training data on bag of visual words with knn for different parameters to increase accuracy rate
    for cluster in clusterValues:
        for percent in percentValues:
            startTime = time.time()
            trainHistograms, trainHistogramsMapping = bovwKNNTrain(percent, cluster, True, directory)
            bovwKNNTest(percent, trainHistograms, trainHistogramsMapping, 5, cluster, True, directory)
            print("total took {} seconds".format(time.time() - startTime))

def testForDifferentParametersBOVWKNN(clusterValues, percentValues, neighborNumbers, directory):
    # testing data on bag of visual words with knn for different parameters to find best parameters
    maxVal = 0
    data = []
    for cluster in clusterValues:
        for percent in percentValues:
            for neighbor in neighborNumbers:
                startTime = time.time()
                trainHistograms, trainHistogramsMapping = bovwKNNTrain(percent, cluster, False, directory)
                accuracy = bovwKNNTest(percent, trainHistograms, trainHistogramsMapping, neighbor, cluster, False,
                                       directory)
                if accuracy > maxVal:
                    maxVal = accuracy
                    data = [percent, cluster, neighbor]
                print("Accuracy for cluster: {}, percent: {}, neighbor: {} ->".format(cluster, percent, neighbor),
                      accuracy)
                print("Took {} seconds".format(round(time.time() - startTime, 2)))

    print("Maximum success rate bovw knn: {}".format(maxVal))
    print("Cluster: {}, Percent: {}, Neighbor: {} ".format(data[1], data[0], data[2]))

def trainForDifferentParametersBOVWSVM(clusterValues, percentValues, train, modelDirectory):
    # training data on bag of visual words with svm for different parameters to increase accuracy rate
    for cluster in clusterValues:
        for percent in percentValues:
            startTime = time.time()
            bovwSVM(cluster, percent, train, modelDirectory)
            print("BOVW SVM Total took {} seconds".format(round(time.time() - startTime, 2)))

def testForDifferentParametersBOVWSVM(clusterValues, percentValues, train, modelDirectory):
    # testing data on bag of visual words with svm for different parameters to find best parameters
    maxVal = 0
    data = []
    for cluster in clusterValues:
        for percent in percentValues:
            score = bovwSVM(cluster, percent, train, modelDirectory)
            if score > maxVal:
                maxVal = score
                data = [cluster, percent]

    print("Maximum success rate bovw SVM: %{}".format(maxVal * 100))
    print("Cluster: {}, Percent: {}".format(data[0], data[1]))

def trainForDifferentParametersTinyKNN(percentValues, neighborValues, modelDirectory, train):
    # training data on tiny image features with knn for different parameters to increase accuracy rate
    for percent in percentValues:
        for neighbor in neighborValues:
            startTime = time.time()
            tinyImageFeaturesKNN(percent, neighbor, modelDirectory, train)
            print("Total took {} seconds".format(round(time.time() - startTime, 2)))

def testForDifferentParametersTinyKNN(percentValues, neighborValues, modelDirectory, train):
    # testing data on tiny image features with knn for different parameters to find best parameters
    maxVal = 0
    data = []
    for percent in percentValues:
        for neighbor in neighborValues:
            score = tinyImageFeaturesKNN(percent, neighbor, modelDirectory, train)
            if score > maxVal:
                maxVal = score
                data = [percent, neighbor]
    print("Maximum success rate tiny knn: %{}".format(maxVal * 100))
    print("Percent: {}, Neighbor: {}".format(data[0], data[1]))

def trainForDifferentParametersTinySVM(percentValues, modelDirectory, train):
    # training data on tiny image features with svm for different split percents to find best percent ratio
    for percent in percentValues:
        tinyImageFeaturesSVM(percent, modelDirectory, train)

def testForDifferentParametersTinySVM(percentValues, modelDirectory, train):
    # testing data on tiny image features with svm for different percents to find best percent value
    maxVal = 0
    data = []
    for percent in percentValues:
        score = tinyImageFeaturesSVM(percent, modelDirectory, train)
        if maxVal < score:
            maxVal = score
            data = [percent]
    print("Maximum success rate tiny SVM: %{}".format(maxVal * 100))
    print("Percent: {}".format(data[0]))

def main():
    while(True):
        userInput = input(
            "1) Bag of Visual Words With KNN\n2) Bag of Visual Words With SVM\n3) Tiny Image Features With KNN\n4) Tiny Image Features With SVM\n'exit' to terminate...\nChoose one: ")
        if userInput == "1":
            trainHistograms, trainHistogramsMapping = bovwKNNTrain(80, 40, False, "bovw_knn_models")
            score = bovwKNNTest(80, trainHistograms, trainHistogramsMapping, 3, 40, False, "bovw_knn_models")
            print("Accuracy for Bag of Visual Words With KNN: %{}".format(score))
        elif userInput == "2":
            score = bovwSVM(30, 80, False, "bovw_svm_models")
            print("Accuracy for Bag of Visual Words With SVM: %{}".format(score*100))
        elif userInput == "3":
            score = tinyImageFeaturesKNN(80, 7, "tiny_knn_models", False)
            print("Accuracy for Tiny Image Features With KNN: %{}".format(score * 100))
        elif userInput == "4":
            score = tinyImageFeaturesSVM(80, "tiny_svm_models", False)
            print("Accuracy for Tiny Image Features With SVM: %{}".format(score * 100))
        elif userInput == "exit":
            break
        else:
            print("INVALID CHOOSE!")
            continue
main()

"""Bag of Visual Words With K-Nearest Neighbors"""
"""uncomment below lines to train data on different percent and n_cluster values to find best pair"""
# trainForDifferentParametersBOVWKNN([10, 40, 70,], [20, 40, 50, 60, 80], "bovw_knn_models")
# testForDifferentParametersBOVWKNN([10, 40, 70], [20, 40, 50, 60, 80], [3, 5, 7], "bovw_knn_models")

"""Bag of Visual Words With Linear Support Vector Machines"""
"""uncomment below lines to train data on different percent and cluster values to find best parameter pair"""
# trainForDifferentParametersBOVWSVM([10, 30, 50, 70], [20, 40, 50, 60, 80], True, "bovw_svm_models")
# testForDifferentParametersBOVWSVM([10, 30, 50, 70], [20, 40, 50, 60, 80], False, "bovw_svm_models")

"""Tiny Image Features With K-Nearest Neighbors"""
"""uncomment below lines to train data on different percent and neighbor values to find best parameter pair"""
# trainForDifferentParametersTinyKNN([20, 40, 50, 60, 80], [3, 5, 7, 9], "tiny_knn_models", True)
# testForDifferentParametersTinyKNN([20, 40, 50, 60, 80], [3, 5, 7, 9], "tiny_knn_models", False)

"""Tiny Image Features With Linear Support Vector Machines"""
"""uncomment below lines to train data on different percent values to find best percent value"""
# trainForDifferentParametersTinySVM([20, 40, 50, 60, 80], "tiny_svm_models", True)
# testForDifferentParametersTinySVM([20, 40, 50, 60, 80], "tiny_svm_models", False)



