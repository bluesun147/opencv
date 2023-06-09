import cv2 
import numpy as np 

# https://metar.tistory.com/entry/openCV얼굴인식-기술을-이용한-화면-잠금해제
# https://blog.naver.com/chandong83/221436424539

# 사진 100장 찍기

# 얼굴 인식용 xml 파일
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# 전체 사진에서 얼굴만 잘라서 리턴
def face_extractor(img): 

    # 흑백 처리
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 얼굴 찾기
    faces = face_classifier.detectMultiScale(gray,1.3,5) 

    # 얼굴 못찾으면
    if faces is(): 
        return None 

    # 얼굴 있으면
    for(x,y,w,h) in faces: 
        # 해당 얼굴 크기만큼 잘라 넣기
        cropped_face = img[y:y+h, x:x+w] 

    return cropped_face 


# 카메라 실행
cap = cv2.VideoCapture(0)
# 저장할 이미지 카운트
count = 0 

while True:
    # 카메라로 부터 사진 1장 얻기
    ret, frame = cap.read()
    # 얼굴 감지래 얼굴만 가져오기
    if face_extractor(frame) is not None: 
        count+=1
        # 얼굴 이미지 크기 조정
        face = cv2.resize(face_extractor(frame),(200,200))
        # 조정 후 흑백 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) 
        # 이미지 저장
        file_name_path = 'C:/Users/blues/Desktop/4-1/face/'+str(count)+'.jpg' 
        cv2.imwrite(file_name_path,face) 

        # 화면에 얼굴과 저장 개수 표시
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
        cv2.imshow('Face Cropper',face) 
    else: 
        print("Face not Found") 
        pass 

    if cv2.waitKey(1)==13 or count==100: 
        break 

cap.release() 
cv2.destroyAllWindows() 
print('Colleting Samples Complete!!!')