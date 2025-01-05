from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import UserSerializer
import face_recognition
import base64
import numpy as np
from PIL import Image
from io import BytesIO


class UserSignup(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES.get("photo")
        if not image_file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = Image.open(image_file)
            image = image.convert("RGB") 
            image_array = np.array(image)

            face_encodings = face_recognition.face_encodings(image_array)
            if not face_encodings:
                return Response({"error": "No face detected in the image"}, status=status.HTTP_400_BAD_REQUEST)

            face_encoding = face_encodings[0]

            face_encoding_str = base64.b64encode(face_encoding).decode("utf-8")

            serializer.save(face_encoding=face_encoding_str)
            return Response({"message": "User signed up successfully"}, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": f"Error processing the image: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
    

class UserLogin(APIView):
    def post(self, request):
        username=request.data.get('username','').strip()
        password=request.data.get('password','').strip()
        
        if not username or not password:
            return Response ({"error": "username and password are required"}, status=status.HTTP_400_BAD_REQUEST)
                    
        try:
            user = User.objects.get(username__iexact=username.strip())
        except User.DoesNotExist as e:
            print(f"Error: {str(e)}")
            return Response({"error": "Invalid username or password"}, status=status.HTTP_400_BAD_REQUEST)
        
        if password == user.password:
            return Response({"message": "Login successful"}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Invalid username or password"}, status=status.HTTP_400_BAD_REQUEST)




from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import face_recognition
from .models import User
import numpy as np 

class FaceInputView(APIView):
    def post(self, request):
        face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        video_cap = cv2.VideoCapture(0)
        if not video_cap.isOpened():
            return Response({"error": "Could not access the camera"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        face_details = []
        try:
            while True:
                ret, video_data = video_cap.read()
                if not ret:
                    return Response({"error": "Failed to capture video frame"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                rgb_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2RGB)

                faces = face_cap.detectMultiScale(
                    cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY),
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                face_details = []
                for (x, y, w, h) in faces:
                    face_details.append({"x": x, "y": y, "width": w, "height": h})
                    cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("Face Detection", video_data)

                if cv2.waitKey(10) == ord("a"):
                    break

        finally:
            video_cap.release()
            cv2.destroyAllWindows()

        if face_details:
            face_locations = [(face['x'], face['y'], face['x'] + face['width'], face['y'] + face['height']) for face in face_details]
            camera_face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)

            if camera_face_encoding:  
                camera_face_encoding = camera_face_encoding[0] 
                          
                user_profiles = User.objects.all()
                for user_profile in user_profiles:
                    stored_face_encoding = user_profile.face_encoding
                    if stored_face_encoding is None:
                        return Response({"error": f"User {user_profile.id} does not have a stored face encoding","camera_face_encoding":camera_face_encoding}, status=status.HTTP_400_BAD_REQUEST)
                    
                    try:
                        stored_face_encoding = np.fromstring(stored_face_encoding, sep=',')  
                    except ValueError:
                        return Response({"error": f"Stored face encoding for user {user_profile.id} is invalid"}, status=status.HTTP_400_BAD_REQUEST)

                    if stored_face_encoding.size > 0: 
                        matches = face_recognition.compare_faces([stored_face_encoding], camera_face_encoding)
                        
                        if True in matches: 
                            return Response({"message": "Faces match", "result": True, "user_id": user_profile.id}, status=status.HTTP_200_OK)

                return Response({"message": "Faces do not match with any stored user", "result": False}, status=status.HTTP_400_BAD_REQUEST)
            
            else:
                return Response({"message": "No valid face encoding detected from camera"}, status=status.HTTP_404_NOT_FOUND)

        else:
            return Response({"message": "No face detected in the video frame"}, status=status.HTTP_404_NOT_FOUND)




import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.conf import settings
from .models import Attendance

class MarkAttendance(APIView):
    def post(self, request):
        camera_image = request.FILES.get('image')
        if not camera_image:
            return Response({"error": "Image not provided"}, status=status.HTTP_400_BAD_REQUEST)

        for user in User.objects.all():
            user_photo_path = default_storage.path(user.photo.name)
            user_photo = cv2.imread(user_photo_path)
            user_photo_gray = cv2.cvtColor(user_photo, cv2.COLOR_BGR2GRAY)
            
            camera_image_array = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            camera_image_decoded = cv2.imdecode(camera_image_array, cv2.IMREAD_COLOR)
            camera_image_gray = cv2.cvtColor(camera_image_decoded, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            user_faces = face_cascade.detectMultiScale(user_photo_gray, scaleFactor=1.1, minNeighbors=5)
            camera_faces = face_cascade.detectMultiScale(camera_image_gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in camera_faces:
                for (ux, uy, uw, uh) in user_faces:
                    user_face_crop = user_photo_gray[uy:uy+uh, ux:ux+uw]
                    camera_face_crop = camera_image_gray[y:y+h, x:x+w]
                    user_face_resized = cv2.resize(user_face_crop, (100, 100))
                    camera_face_resized = cv2.resize(camera_face_crop, (100, 100))

                    diff = cv2.absdiff(user_face_resized, camera_face_resized)
                    if np.mean(diff) < 50:  
                        Attendance.objects.create(user=user)
                        return Response({"message": "Attendance marked"}, status=status.HTTP_200_OK)

        return Response({"message": "Your attendance is not recorded"}, status=status.HTTP_400_BAD_REQUEST)





from django.utils.dateparse import parse_date

class AttendanceReport(APIView):
    def get(self, request):
        date = request.query_params.get('date')
        if not date:
            return Response({"error": "Date parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            parsed_date = parse_date(date)
            attendance_records = Attendance.objects.filter(date=parsed_date).select_related('user')
            report = [
                {
                    "username": record.user.username,
                    "email": record.user.email,
                    "date": record.date,
                    "time": record.time,
                }
                for record in attendance_records
            ]
            return Response(report, status=status.HTTP_200_OK)
        except ValueError:
            return Response({"error": "Invalid date format"}, status=status.HTTP_400_BAD_REQUEST)


