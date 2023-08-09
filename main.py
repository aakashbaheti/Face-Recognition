#Aakash Baheti, 08/08/2023, Face Recognition Program

import face_recognition
import os, sys
import cv2
import numpy as np
import math

def confidence_of_face(distance_of_face, matching_face_threshold=0.6):
    num = (1.0 - matching_face_threshold)
    linear_value = (1.0 - distance_of_face) / (num / 2.0)

    if distance_of_face > matching_face_threshold:
        return str(round(linear_value * 100, 2)) + '%'
    else: 
        val = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + '%'
    

class FaceRecognition: 
    face_position = []
    face_encodings = []
    face_people = []
    known_face_encodings = []
    known_face_peoples = []

    comprehend_the_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}') 
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_peoples.append(image)

        print (self.known_face_peoples) # tests program functionality by printing png's names to terminal

    
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source is not found...')

        while True:
            ret, frame = video_capture.read()

            if self.comprehend_the_current_frame:
                # Resize and change frame to RGB
                tiny_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_tiny_frame = tiny_frame[:, :, ::-1]

                # Find all faces in the current frame
                self.face_position = face_recognition.face_locations(rgb_tiny_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_tiny_frame, self.face_position)

                self.face_people = []
                for face_encoding in self.face_encodings:
                    match_identified = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    people_name = 'Unkown'
                    people_confidnece = 'Unkown'

                    distance_of_faces = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(distance_of_faces)

                    if match_identified[best_match_index]:
                        people_name = self.known_face_peoples[best_match_index]
                        people_confidnece = confidence_of_face(distance_of_faces[best_match_index])

                    self.face_people.append(f'{people_name} ({people_confidnece})')

            self.comprehend_the_current_frame = not self.comprehend_the_current_frame

            # Display annotations
            for (top, right, bottom, left), people_name in zip(self.face_position, self.face_people):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, people_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition