###PnP 함수 설명
# retval, rvec, tvec, inliers = cv2.solvePnPRansac(
#    objectPoints(필수), 
#   imagePoints(필수), 
#   cameraMatrix(필수), 
#   distCoeffs(필수)[, 
#   rvec[, 
#   tvec[, 
#   useExtrinsicGuess[, 
#   iterationsCount[, 
#   reprojectionError[, 
#   confidence[, 
#   inliers[, 
#   flags]]]]]]]]
## 반환값
#   retval: 함수의 성공 여부를 나타내는 불리언 값
#   rvec: 회전 벡터(로드리게스 형식)
#   tvec: 번역 벡터
#   inliers: 내리어의 인덱스를 포함하는 리스트
###

#### code flow
# image_finder.py로 부터 2D 특징점(keypoints)와 몇 번째 이미지 인지 입력받음
# GetPosePnp.first_img_mat 또는 .second_omg_mat을 호출하면 됨
####

import depthai as dai
import numpy as np
import numpy as np
import cv2

class GetPosePnp():
    def __init__(self):
        self.device = None

    def img_matrices(self, keypoints_3Dcoords, keypoints_2Dcoords):
        object_points = keypoints_3Dcoords[:6]
        imagePoints = keypoints_2Dcoords[:6]
        object_points = np.hstack((object_points, np.zeros((object_points.shape[0], 1), dtype=np.float32)))
        self.get_tf(object_points, imagePoints)

    # def second_img_mat(self, object_points, imagePoints):
    #     object_points = np.array([
    #         [0.0, 0.0, 0.0],
    #         [0.18, 0.0, 0.0],
    #         [0.18, 0.18, 0.0],
    #         [0.0, 0.18, 0.0]
    #     ], dtype=np.float32)
    #     self.get_tf(object_points, imagePoints)

    def get_OAK_matrices(self):
        # 위 방법으로 추출한 내부 파라미터와 왜곡 계수를 저장하여 사용
        # 예시
        camera_intrinsics_matrix = np.array([
            [200.99806213378906,   0.0, 125.13585662841797],
            [  0.0, 200.99806213378906, 130.5442657470703],
            [  0.0,   0.0,   1.0]
        ],dtype=np.float32)
        
        distortion_coeffs = np.array([
            13.380138397216797,
            -150.46047973632812,
            0.0022977159824222326,
            0.001133363926783204,
            541.2130126953125,
            13.132322311401367,
            -148.4298858642578,
            533.8888549804688,
        ], dtype=np.float32)
        return camera_intrinsics_matrix, distortion_coeffs

    def get_tf(self, objectPoints, imagePoints):
        camera_intrinsics_matrix, distortion_coeffs = self.get_OAK_matrices()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, camera_intrinsics_matrix, distortion_coeffs)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            # R 은 월드좌표계 기준 카메라 좌표계의 회전 행렬
            # tvec 카메라 좌표계기준 물체의 중심점의 위치
            return R, tvec
            
            # rotated_tvec은 월드좌표계 기준 물체의 중심점의 위치
            rotated_tvec = np.dot(R, tvec)
            return rotated_tvec
        else:
            print("PnP 실패")
            return None