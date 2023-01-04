from scipy import stats
import time
import numpy as np
import cv2
# from ipynb.fs.full.Detection import mp_detection, mp_draw, extract_kp
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
model=load_model('detector.h5')
colors = [(245,117,16), (117,245,16), (16,117,245)]
actions=np.array(['attention', 'pistol', 'sniper'])

def prob_visualization(res, actions, input_frame, colors):
    output_frame=input_frame.copy()
    for id, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+id*40), (int(prob*100), 90+id*40), colors[id], -1)
        cv2.putText(output_frame, actions[id], (0,85+id*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def mp_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def mp_draw(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

def extract_kp(results):
    pose_kp=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face_kp=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh_kp=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh_kp=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose_kp, face_kp, lh_kp, rh_kp])

def main():
    vid = []
    sentence = []
    predictions = []
    threshold = 0.8
    cap=cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame=cap.read()
            image, results=mp_detection(frame, holistic)
            mp_draw(image,results)

            keypoints=extract_kp(results)
            vid.append(keypoints)
            vid=vid[-30:]

            if len(vid)==30:
                res= model.predict(np.expand_dims(vid, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-5:])[0]==np.argmax(res):
                    if res[np.argmax(res)]>threshold:
                        if len(sentence)>0:
                            if actions[np.argmax(res)]!=sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence)>5:
                    sentence=sentence[-5:]
                image= prob_visualization(res, actions, image, colors) 

            cv2.imshow('Military Sign Recognition', image)
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()