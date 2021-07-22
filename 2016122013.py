import cv2
import numpy as np
import os
import sys

## Step 1 Code ##
def Problem_solver(ratio) :


    train_folder_path = './faces_training/'

    train_images = []

    for i in os.listdir(train_folder_path):
        train_img_path = train_folder_path + i
        img = cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE)
        train_images.append(img)

    train_images = np.array(train_images)
    n = train_images.shape[1] * train_images.shape[2]
    p = len(train_images)
    X = np.reshape(train_images, (n, p))  # vectorized train data"""

    avg_face = np.mean(X, axis=1)


    # mean sub-tracted training data
    X = X - np.tile(avg_face, (X.shape[1], 1)).T

    U, Sigma, VT = np.linalg.svd(X, full_matrices=0)


    criteria = np.sum(Sigma)*ratio

    Selected = []
    for i in Sigma :
        Selected.append(i)
        if sum(Selected) >= criteria:
            break


    selected_dimension = len(Selected)


    print('######### STEP1 ##########')
    print(f'Input Percentage: {ratio}')
    print(f'Selected Dimension : {selected_dimension}')

    print('######### STEP2 ##########')
    # Reconstruct all images in the training set
    Y = np.matmul(U[:,:selected_dimension].T , X)
    Y = np.matmul(U[:,:selected_dimension] , Y)
    train_reconstruct = Y.reshape((-1,train_images.shape[1],train_images.shape[2]))

    reconstruction_errors = []

    os.makedirs('./2016122013')


    for i in range(train_reconstruct.shape[0]) :
        filename = './2016122013/'+ os.listdir(train_folder_path)[i]

        original = train_images[i, :, :]
        reconstructed = train_reconstruct[i, :, :]
        cv2.imwrite(filename, reconstructed)

        # Mean Squared Error
        reconstruction_errors.append(np.mean(np.square(original - reconstructed)))

    reconstruction_errors = np.array(reconstruction_errors)
    print('Reconstruction error')
    print(f'average : {np.mean(reconstruction_errors)}')

    # 메모장에 에러 결과 출력
    file = open('./2016122013/output.txt', 'a+')
    file.write('Reconstruction error')
    file.write('average : %f' % np.mean(reconstruction_errors) )


    for i in range(len(train_images)) :
        print(f'{i+1}: {reconstruction_errors[i]}')
        file.write('%d : %f \n' % (i+1, reconstruction_errors[i]))

    file.close()

    print('######### STEP3 ##########')
    test_folder_path = './faces_test/'

    test_images = []

    for i in os.listdir(test_folder_path):
        test_img_path = test_folder_path + i
        img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        test_images.append(img)


    # vectorization of test images
    test_images = np.array(test_images)
    print(test_images.shape)
    n = test_images.shape[1] * test_images.shape[2]
    p = len(test_images)
    X_test = np.reshape(test_images, (n, p))  # vectorized test data


    # reconstruction of test images
    Y_test = np.matmul(U[:, :selected_dimension].T, X_test)
    Y_test = np.matmul(U[:, :selected_dimension], Y_test)
    test_reconstruct_comparison = np.reshape(Y_test, (n, p))  # vectorized test reconstruct image

    for i in range(len(test_images)) :
        a = test_reconstruct_comparison[:,i]

        dist = []
        for j in range(len(train_images)) :
            b = Y[:,j]    # Y is reconstructed train image
            dist.append(np.linalg.norm(a - b))


        min_index = dist.index(min(dist))  # dist 리스트 중 최소값의 인덱스
        similar_img_name = os.listdir(train_folder_path)[min_index]
        test_img_name = os.listdir(test_folder_path)[i]

        print(f'{test_img_name} => {similar_img_name}')


if __name__ == '__main__' :
    Problem_solver(float(sys.argv[1]))



