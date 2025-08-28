import cv2
import face_recognition
import os

KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

print("[INFO] Loading known faces...")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            
            known_face_names.append(os.path.splitext(filename)[0])
            print(f"[INFO] Loaded {filename} as {known_face_names[-1]}")
        else:
            print(f"[WARNING] No face found in {filename}, skipped.")


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces + encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
       
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # --- Draw facial landmarks ---
        for feature, points in landmarks.items():
            for point in points:
                x, y = point[0] * 4, point[1] * 4  # scale up coords
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # --- Label the face ---
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Face Recognition with Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
