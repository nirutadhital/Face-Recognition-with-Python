from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import UserSerializer



class UserSignup(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "User signed up successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    



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
            
            # Read camera image
            camera_image_array = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            camera_image_decoded = cv2.imdecode(camera_image_array, cv2.IMREAD_COLOR)
            camera_image_gray = cv2.cvtColor(camera_image_decoded, cv2.COLOR_BGR2GRAY)

            # Initialize Face Recognizer
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            user_faces = face_cascade.detectMultiScale(user_photo_gray, scaleFactor=1.1, minNeighbors=5)
            camera_faces = face_cascade.detectMultiScale(camera_image_gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in camera_faces:
                for (ux, uy, uw, uh) in user_faces:
                    user_face_crop = user_photo_gray[uy:uy+uh, ux:ux+uw]
                    camera_face_crop = camera_image_gray[y:y+h, x:x+w]
                    user_face_resized = cv2.resize(user_face_crop, (100, 100))
                    camera_face_resized = cv2.resize(camera_face_crop, (100, 100))

                    # Check similarity
                    diff = cv2.absdiff(user_face_resized, camera_face_resized)
                    if np.mean(diff) < 50:  # Threshold for similarity
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


