from django.db import models

class BaseModel(models.Model):
    is_active = models.BooleanField(default=True)
    is_deleted = models.BooleanField(default=False)
    added_on = models.DateTimeField(auto_now_add=True)
    added_by = models.ForeignKey('User', on_delete=models.CASCADE, related_name='%(class)s_added_by')
    updated_on = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey('User', on_delete=models.CASCADE, related_name='%(class)s_updated_by')
    deleted_on = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        'User', on_delete=models.CASCADE, null=True, blank=True, related_name='%(class)s_deleted_by'
    )

    class Meta:
        abstract = True

class Company(BaseModel):
    company_name = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    contact = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    print_logo = models.CharField(max_length=100)
    application_logo = models.CharField(max_length=100)
    terms_and_conditions = models.CharField(max_length=100)
    company_vat_no = models.CharField(max_length=100)
    company_pan_no = models.CharField(max_length=100)

    def __str__(self):
        return self.company_name
    

class Department(BaseModel):
    department_name = models.CharField(max_length=100)

    def __str__(self):
        return self.department_name


class Faculty(BaseModel):
    faculty_name = models.CharField(max_length=100)

    def __str__(self):
        return self.faculty_name


class Classes(BaseModel):
    classes_name = models.CharField(max_length=100)

    def __str__(self):
        return self.classes_name



class User(models.Model):
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=128, null=False)
    email = models.EmailField(unique=True)
    photo = models.ImageField(upload_to='photos/')
    created_at = models.DateTimeField(auto_now_add=True)
    face_encoding = models.TextField(blank=True, null=True)
    company=models.ForeignKey(Company, on_delete=models.CASCADE)
    department=models.ForeignKey(Department,on_delete=models.CASCADE)
    faculty=models.ForeignKey(Faculty,on_delete=models.CASCADE)
    classes=models.ForeignKey(Classes,on_delete=models.CASCADE)
    
    
    def __str__(self):#used to define the human readable representation of an object
        return self.username


class Attendance(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.time}"


class Role(BaseModel):
    role_name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.role_name


class UserInRole(models.Model):
    user_in_role_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    role = models.ForeignKey(Role, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'role')

    def __str__(self):
        return f"{self.user.username} - {self.role.role_name}"



class Location(BaseModel):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    location_name = models.CharField(max_length=100)
    contact = models.CharField(max_length=100)

    def __str__(self):
        return self.location_name


class Holiday(BaseModel):
    holiday_name = models.CharField(max_length=100)

    def __str__(self):
        return self.holiday_name
    
class LeaveType(BaseModel):
    leave_name=models.CharField(max_length=100)
    
    def __str__(self):
        return self.leave_name
    
    
    
class UserInLeave(BaseModel):
    user_in_leave_id=models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    leave = models.ForeignKey(LeaveType, on_delete=models.CASCADE)
    description=models.CharField(max_length=100)
    
    def __str__(self):
        return f"{self.user.username} - {self.leave.leave_name}"
    
    
