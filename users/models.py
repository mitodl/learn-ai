"""Custom user model and related stuff."""

import json

from django.conf import settings
from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.core.exceptions import ValidationError
from django.db import models, transaction

from main.models import TimestampedModel


class UserManager(BaseUserManager):
    """User manager for custom user model"""

    use_in_migrations = True

    @transaction.atomic
    def _create_user(self, username, email, password, **extra_fields):
        """Create and save a user with the given email and password"""
        email = self.normalize_email(email)
        fields = {**extra_fields, "email": email}
        if username is not None:
            fields["username"] = username
        user = self.model(**fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, username, email=None, password=None, **extra_fields):
        """Create a user"""
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(username, email, password, **extra_fields)

    def create_superuser(self, username, email, password, **extra_fields):
        """Create a superuser"""
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            msg = "Superuser must have is_staff=True."
            raise ValueError(msg)
        if extra_fields.get("is_superuser") is not True:
            msg = "Superuser must have is_superuser=True."
            raise ValueError(msg)

        return self._create_user(username, email, password, **extra_fields)

    @transaction.atomic
    def create_global_user(self, global_id, email, **extra_fields):
        """
        Create a global (SSO) user.

        SSO users don't get usable passwords and their usernames are set to the
        global ID by default, unless specified in extra_fields.
        """

        if not global_id:
            msg = "global_id is required"
            raise ValidationError(msg)

        extra_fields = {
            **extra_fields,
            "global_id": global_id,
            "is_active": True,
        }

        username = extra_fields.get("username", global_id)

        user = self.create_user(username, email, **extra_fields)
        user.set_unusable_password()
        user.save()
        return user


class User(AbstractBaseUser, TimestampedModel, PermissionsMixin):
    """Primary user class"""

    EMAIL_FIELD = "email"
    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = ["email", "name"]

    username = models.CharField(unique=True, max_length=150)
    email = models.EmailField(blank=False, unique=True)
    name = models.CharField(blank=True, default="", max_length=255)
    global_id = models.CharField(blank=True, default="", max_length=128)
    is_staff = models.BooleanField(
        default=False, help_text="The user can access the admin site"
    )
    is_active = models.BooleanField(
        default=False, help_text="The user account is active"
    )

    objects = UserManager()

    def get_full_name(self):
        """Return the user's fullname"""
        return self.name

    def __str__(self):
        """Str representation for the user"""
        return (
            f"User global_id={self.global_id} username={self.username} "
            f"email={self.email}"
        )

    def to_json(self):
        return json.dumps(
            {
                "id": self.id,
                "global_id": self.global_id,
                "username": self.username,
                "email": self.email,
                "name": self.name,
                "is_active": self.is_active,
                "is_staff": self.is_staff,
                "is_superuser": self.is_superuser,
            }
        )


class UserProfile(TimestampedModel):
    """Provides additional fields for userdata."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile"
    )

    country_code = models.CharField(blank=True, default="", max_length=2)
