"""
Factory for Users
"""

from uuid import uuid4

import ulid
from django.conf import settings
from django.contrib.auth.models import Group
from factory import LazyFunction, Sequence, SubFactory
from factory.django import DjangoModelFactory
from factory.fuzzy import FuzzyText
from social_django.models import UserSocialAuth


class UserFactory(DjangoModelFactory):
    """Factory for Users"""

    username = Sequence(lambda n: f"{ulid.new().str}{n:03d}")
    global_id = LazyFunction(lambda: str(uuid4()))
    email = Sequence(lambda n: f"{n:03d}_{ulid.new().str}@example.com")
    name = FuzzyText()
    is_active = True

    class Meta:
        model = settings.AUTH_USER_MODEL
        skip_postgeneration_save = True


class GroupFactory(DjangoModelFactory):
    """Factory for Groups"""

    name = FuzzyText()

    class Meta:
        model = Group


class UserSocialAuthFactory(DjangoModelFactory):
    """Factory for UserSocialAuth"""

    provider = FuzzyText()
    user = SubFactory(UserFactory)
    uid = FuzzyText()

    class Meta:
        model = UserSocialAuth
