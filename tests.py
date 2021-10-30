from numpy import array
import numpy as np

if __name__ == "__main__":

    data = array([[11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]])



    print("Shape: " + str(data.shape))

    # for i in range(0, 3):
    #     print(f"{i+1}: {data[i, :]}")

    for co, i in enumerate(data[0, :], start=1):
        print(f"co: {co}, i: {i}")

    data = np.uint8(np.around(data))

    print("\nShape: " + str(data.shape))

    # for i in range(0, 3):
    #     print(f"{i+1}: {data[i, :]}")

    for co, i in enumerate(data[0, :], start=1):
        print(f"co: {co}, i: {i}")


    # data2 = array([])
    #
    # print("\nShape: " + str(data2.shape))