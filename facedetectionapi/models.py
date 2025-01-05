from django.db import models

class User(models.Model):
    username=models.CharField(max_length=100, unique=True)
    password=models.CharField(max_length=128, null=False)
    email=models.EmailField(unique=True)
    photo=models.ImageField(upload_to='photos/')
    created_at=models.DateTimeField(auto_now_add=True)
    face_encoding = models.TextField(blank=True, null=True)  # Use TextField to store large data like encodings

    
    def __str__(self):
        return self.username
    


class Attendance(models.Model):
    user=models.ForeignKey(User, on_delete=models.CASCADE)
    date=models.DateField(auto_now_add=True)
    time=models.TimeField(auto_now_add=True)
    
    
    def __str__(self):
        return f"{self.user.username}- {self.date}-{self.time}"