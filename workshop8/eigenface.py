import matplotlib.pyplot as plt
import cv2
import numpy as np

def calculation(width, height, dataset_path, train_image_names):
    training_tensor   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)
    for i in range(len(train_image_names)):
        img = plt.imread(dataset_path + train_image_names[i])
        training_tensor[i,:] = np.array(img, dtype='float64').flatten()

    # Mean face
    mean_face = np.zeros((1,height*width))
    for i in training_tensor:
        mean_face = np.add(mean_face,i)
    mean_face = np.divide(mean_face,float(len(train_image_names))).flatten()

    # Normalized faces
    normalised_training_tensor = np.ndarray(shape=(len(train_image_names), height*width))
    for i in range(len(train_image_names)):
        normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)

    # Calculate covariance matrix, eigenvalues, eigenvectors
    cov_matrix = np.cov(normalised_training_tensor)
    cov_matrix = np.divide(cov_matrix,8.0)
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

    # Choose the necessary no.of principle components
    reduced_data = np.array(eigvectors_sort[:7]).transpose()

    # Find projected data --> eigen space
    proj_data = np.dot(training_tensor.transpose(),reduced_data)
    proj_data = proj_data.transpose()

    # Weight for each training images
    w = np.array([np.dot(proj_data,i) 
                for i in normalised_training_tensor])
    return mean_face, proj_data, w

def upsize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    upsized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return upsized

def recogniser(img, train_image_names, mean_face, proj_data, w):
    unknown_face = plt.imread('Pos/'+img)
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
    normalised_uface_vector = np.subtract(unknown_face_vector, mean_face)

    # Components of text added to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (40, 30)
    fontScale = 0.8
    thickness = 3

    # Show input image
    input_img = cv2.imread('Pos/' + img)
    input_img = upsize(input_img, 200)
    input_img = cv2.putText(input_img, 'Input:'+'.'.join(img.split('.')[:2]),
                      org, font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.imshow('Input', input_img)
    
    # Calculate unknown weight
    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff  = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)

    # Choose threshold
    t1 = 100111536
    t0 = 88831687
    
    if norms[index] < t1:
        if norms[index] < t0: # It's a face
            if img.split('.')[0] == train_image_names[index].split('.')[0]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            image = cv2.imread('Pos/' + train_image_names[index])
            image = upsize(image, 200)
            image = cv2.putText(image, 'Matched:'+'.'.join(train_image_names[index].split('.')[:2]),
                                org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Result',image)
        else:
            if img.split('.')[0] not in [i.split('.')[0] for i in train_image_names] and img.split('.')[0] != 'apple':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            image = cv2.imread('unknown.jpg')
            image = upsize(image, 200)
            image = cv2.putText(image, 'Unknown face!', org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Result',image)
    else:   
        if len(img.split('.')) == 3:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        image = cv2.imread('noface.jpg')
        image = upsize(image, 200)
        image = cv2.putText(image, 'Not a face!', org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Result',image)