import cv2
import numpy as np

class Yolov2349865:
    def __init__(self):
        pass

    def image_resize(self, frame):
        # 샘플 이미지 저장
        self.sample_image1 = cv2.imread('/home/pgt/doosan/5/ext_orig.png', cv2.IMREAD_GRAYSCALE)
        self.sample_image2 = cv2.imread('/home/pgt/doosan/5/man_orig.png', cv2.IMREAD_GRAYSCALE)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        A1 = self.sample_image1.copy()
        A2 = self.sample_image2.copy()

        image1 = [A1]
        image2 = [A2]

        for i in range(4):
            A1 = cv2.pyrDown(A1)
            A2 = cv2.pyrDown(A2)
            image1.append(A1)
            image2.append(A2)

        # SIFT 생성
        self.sift = cv2.SIFT.create()
        
        self.brute_force_matcher(frame, image1, image2)

    def brute_force_matcher(self, frame, image1, image2):
        # 매칭 객체 생성 (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # 특징점과 기술자 계산
        keypoints3, descriptors3 = self.sift.detectAndCompute(frame, None)
        for img1, img2 in zip(image1, image2):
            ratio1 = False
            ratio2 = False
            keypoints_coords3 = np.array([keypoint.pt for keypoint in keypoints3])
            keypoint1, descriptor1 = self.sift.detectAndCompute(img1, None)
            keypoint2, descriptor2 = self.sift.detectAndCompute(img2, None)
            keypoints_coords2 = np.array([keypoint.pt for keypoint in keypoint2])
            keypoints_coords1 = np.array([keypoint.pt for keypoint in keypoint1])

            if descriptors3 is not None and len(keypoints3) > 80:
                descriptor1 = np.float32(descriptor1)
                descriptor2 = np.float32(descriptor2)
                descriptors3 = np.float32(descriptors3)

                # 매칭 수행
                matches1 = bf.match(descriptor1, descriptors3)
                matches2 = bf.match(descriptor2, descriptors3)

                # 매칭된 특징점들 정렬
                matches1 = sorted(matches1, key = lambda x: x.distance)
                matches2 = sorted(matches2, key = lambda x: x.distance)

                # 매칭된 특징점의 비율을 계산
                good_matches1 = len(matches1)
                total_matches1 = min(len(keypoint1), len(keypoints3))
                good_matches2 = len(matches2)
                total_matches2 = min(len(keypoint2), len(keypoints3))

                img3 = cv2.drawMatches(img1,keypoint1,frame,keypoints3,matches1[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                img4 = cv2.drawMatches(img2,keypoint2,frame,keypoints3,matches2[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # 유사도 판단 기준 (threshold 조정 가능)
                threshold = 0.5  # 예: 매칭된 특징점이 전체 특징점의 50% 이상일 때 같은 이미지로 판단
                similarity_ratio1 = good_matches1 / (total_matches1 + 10)
                print(similarity_ratio1)
                similarity_ratio2 = good_matches2 / (total_matches2 + 35)
                print(similarity_ratio2)
                if similarity_ratio1 >= threshold:
                    ratio1 = True
                if similarity_ratio2 >= threshold:
                    ratio2 = True
                
                if ratio1 and ratio2 == True:
                    if max(similarity_ratio1, similarity_ratio2) == similarity_ratio1:
                        cv2.imshow('', img3)
                        cv2.waitKey(1)
                        print('first image')
                        return 1, keypoints_coords1, keypoints_coords3
                    else:
                        cv2.imshow('', img4)
                        cv2.waitKey(1)
                        print('second image')
                        return 2, keypoints_coords2, keypoints_coords3
                elif ratio1 == True and ratio2 == False:
                    cv2.imshow('', img3)
                    cv2.waitKey(1)
                    print('first image')
                    return 1, keypoints_coords1, keypoints_coords3
                elif ratio2 == True and ratio1 == False:
                    cv2.imshow('', img4)
                    cv2.waitKey(1)
                    print('second image')
                    return 2, keypoints_coords2, keypoints_coords3
                else:
                    cv2.imshow('', frame)
                    cv2.waitKey(1)
                    print('not found image')
                    return 0, 0, 0
            else:
                return 0, 0, 0
                
            

def main():
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            yas = Yolov2349865()
            yas.image_resize(frame)

if __name__ == '__main__':
    main()