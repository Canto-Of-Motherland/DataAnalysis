from django.db import models


class User(models.Model):
    user_id = models.CharField(primary_key=True, max_length=16)
    user_name = models.CharField(max_length=16)
    user_email = models.CharField(max_length=32)
    user_password = models.CharField(max_length=16)
