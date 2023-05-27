from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import norm


class Node:
    def __init__(self, index, parent):
        self.parent = parent
        self.index = index


# def convertMatrixToTree():

#     Nodearr[0] = Node(0, 0)

#     Nodearr[4] = Node(4, 0)
#     Nodearr[13] = Node(13, 0)
#     Nodearr[1] = Node(1, 0)

#     Nodearr[5] = Node(5, 4)
#     Nodearr[6] = Node(6, 5)

#     Nodearr[10] = Node(10, 13)
#     Nodearr[14] = Node(14, 13)
#     Nodearr[7] = Node(7, 13)

#     Nodearr[11] = Node(11, 10)
#     Nodearr[12] = Node(2, 11)

#     Nodearr[8] = Node(8, 1)
#     Nodearr[9] = Node(9, 8)

#     Nodearr[2] = Node(2, 1)
#     Nodearr[3] = Node(3, 2)


def ConvertMatrixToMoveNetSkeleton(Nodearr):
    Nodearr[0] = Node(0, 0)  # origin

    Nodearr[1] = Node(1, 0)  # 5 left shoulder
    Nodearr[7] = Node(7, 0)  # 11 right shoulder
    Nodearr[8] = Node(8, 0)  # 12 left hip
    Nodearr[2] = Node(2, 0)  # 6 right hip

    Nodearr[3] = Node(3, 1)  # 7 left elbow
    Nodearr[5] = Node(5, 3)  # 9 left wrist

    Nodearr[4] = Node(4, 2)  # 8 rigth elbow
    Nodearr[6] = Node(6, 4)  # 10 right wrist

    Nodearr[9] = Node(9, 7)  # 13 left knee
    Nodearr[11] = Node(11, 9)  # 15 left ankle

    Nodearr[10] = Node(10, 8)  # 14 right knee
    Nodearr[12] = Node(12, 10)  # 16 right ankle

    return Nodearr


def RotationAngles(matrix):

    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    theta1 = np.arctan(-r23 / r33)
    theta2 = np.arctan(-r13 * np.cos(theta1))
    theta3 = np.arctan(-r12 / r11)

    t1 = np.array([theta1, 0, 0])
    t2 = np.array([0, theta2, 0])
    t3 = np.array([0, 0, theta3])

    return t1, t2, t3


def getTransformationMatrix(matrix_prev, matrix_curr, matrix_next, is_global=1):

    centroid1 = (matrix_prev + matrix_curr) / 2
    centroid2 = (matrix_curr + matrix_next) / 2

    matrix1 = (matrix_prev - centroid1).reshape(-1, 1)
    matrix2 = (matrix_curr - centroid2).reshape(1, -1)
    matrix3 = (matrix_curr - centroid1).reshape(-1, 1)
    matrix4 = (matrix_next - centroid2).reshape(1, -1)

    H = matrix1.dot(matrix2) + matrix3.dot(matrix4)

    U, S, V = np.linalg.svd(H)
    R = np.dot(V, U.T)  # R = (3, 3)
    t = -R.dot(centroid1) + centroid2  # t = (3, 1)

    arr = np.array([0, 0, 0, 1])

    tm = np.concatenate((R, t.reshape(-1, 1)), axis=1)
    tm = np.concatenate((tm, arr.reshape(1, -1)), axis=0)

    if (is_global):
        return tm, R, t
    else:
        return tm


def Quantification(tensor, total_frames, num_joints, Nodearr):

    g_motion = np.empty((total_frames - 1, num_joints), dtype=object)
    l_motion = np.empty((total_frames - 1, num_joints), dtype=object)

    for i in range(1, total_frames - 1):

        for j in range(num_joints):

            gm, R, t = getTransformationMatrix(tensor[i-1][j],
                                               tensor[i][j], tensor[i+1][j], is_global=1)
            gm_theta1, gm_theta2, gm_theta3 = RotationAngles(R)

            if j == 0 or Nodearr[j].parent == 0:
                lm = gm

            else:
                k = Nodearr[j].parent
                sub_lm = 1

                while(k != 0):
                    k = Nodearr[k].parent
                    sub_lm *= getTransformationMatrix(
                        tensor[i-1][k], tensor[i][k], tensor[i+1][k], is_global=0)

                lm = gm * np.linalg.inv(sub_lm)

            lm_R = lm[0:3, 0:3].copy()
            lm_t = lm[0:3, 3].copy()

            lm_theta1, lm_theta2, lm_theta3 = RotationAngles(lm_R)

            g_result = np.vstack((gm_theta1, gm_theta2, gm_theta3, t))
            l_result = np.vstack((lm_theta1, lm_theta2, lm_theta3, lm_t))

            g_motion[i][j] = np.copy(g_result)
            l_motion[i][j] = np.copy(l_result)

    return g_motion, l_motion


def Distance(matrix1, matrix2):
    # print(matrix1)
    # print(matrix2)
    return np.linalg.norm(matrix1 - matrix2)


def MatrixNormalize(matrix, origin):  # matrix = (N_joints, 3), origin = (3, )
    length = len(matrix)
    result = []

    for i in range(length):
        result.append(matrix[i] - origin)

    return result


# Nodearr is length 13 Node-type array
def Comparsion(tensor1, tensor2, total_frames, num_joints, Nodearr):

    if total_frames < 3:
        print("lack of frame")
        return

    for i in range(total_frames):
        tensor1[i] = MatrixNormalize(
            tensor1[i], origin=((tensor1[i][1] + tensor1[i][2] + tensor1[i][7] + tensor1[i][8]) / 4))
        tensor2[i] = MatrixNormalize(
            tensor2[i], origin=((tensor2[i][1] + tensor2[i][2] + tensor2[i][7] + tensor2[i][8]) / 4))

    g_motion1, l_motion1 = Quantification(
        tensor1, total_frames, num_joints, Nodearr)
    g_motion2, l_motion2 = Quantification(
        tensor2, total_frames, num_joints, Nodearr)

    #print("g_motion1: ", len(g_motion1))

    g_total = 0
    l_total = 0
    total = 0

    for i in range(1, total_frames - 1):
        g_total = 0
        l_total = 0

        for j in range(num_joints):

            g_dt = Distance(g_motion1[i][j], g_motion2[i][j])
            l_dt = Distance(l_motion1[i][j], l_motion2[i][j])

            g_total += g_dt
            l_total += l_dt

        total += g_total + l_total

    return total / (total_frames - 2)


def CreateRandomMatrix(total_frames, num_of_joints):
    matrix1 = np.random.rand(total_frames, num_of_joints, 2)
    matrix2 = np.random.rand(total_frames, num_of_joints, 2)
    zeros = np.zeros((total_frames, num_of_joints, 1))

    matrix1 = np.concatenate((matrix1, zeros), axis=2)
    matrix2 = np.concatenate((matrix2, zeros), axis=2)

    return matrix1, matrix2


def ZtoPercentile(z_score):
    percentile = norm.cdf(z_score) * 100
    return percentile


def GetScore(total_frames, mat1, mat2, num_joints = 13):

    Nodearr = np.empty(num_joints, dtype=Node)
    Nodearr = ConvertMatrixToMoveNetSkeleton(Nodearr)
    # mat1, mat2 = CreateRandomMatrix(total_frames, num_joints)

    # Comparsion
    score = Comparsion(mat1, mat2, total_frames, num_joints, Nodearr)

    print(score)
    return score



app = Flask(__name__)

total_frames = None
tensor1 = None
tensor2 = None

@app.route('/api', methods = ['POST'])
def query_tensor(): 
    data = request.get_json()
    total_frames = data.get('total_frames')

    tensor1 = np.array(data.get('tensor1'))
    tensor2 = np.array(data.get('tensor2'))
    zeros = np.zeros((total_frames, 13, 1))

    tensor1 = np.concatenate((tensor1, zeros), axis=2)
    tensor2 = np.concatenate((tensor2, zeros), axis=2)

    print("Tensor1: ", tensor1[0].size)

    score = GetScore(total_frames, tensor1, tensor2)
    
    return str(score)

if __name__ == "__main__": 
    app.run()
    


# result = []

# for i in range(3):
#     print(i)
#     mat1, mat2 = CreateRandomMatrix(total_frames, num_joints)
#     score = Comparsion(mat1, mat2, total_frames, num_joints)
#     print(score)

#     result.append(score)

# z_score = (score - 23.6) / 1.35
# percentile = ZtoPercentile(z_score)
# print(score, percentile)

# print(np.mean(result))  # 평균
# print(np.var(result))  # 분산
# print(np.std(result))  # 표준 편차
# print(np.max(result))  # 최댓값
# print(np.min(result))  # 최솟값-M