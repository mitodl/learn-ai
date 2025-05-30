"""
Classes related to models for main
"""

from django.core.cache import cache
from django.db.models import CharField, DateTimeField, IntegerField, Model
from django.db.models.query import QuerySet

from main.constants import CONSUMER_THROTTLES_KEY
from main.utils import now_in_utc


class TimestampedModelQuerySet(QuerySet):
    """
    Subclassed QuerySet for TimestampedModelManager
    """

    def update(self, **kwargs):
        """
        Automatically update updated_on timestamp when .update(). This is because .update()
        does not go through .save(), thus will not auto_now, because it happens on the
        database level without loading objects into memory.
        """  # noqa: E501
        if "updated_on" not in kwargs:
            kwargs["updated_on"] = now_in_utc()
        return super().update(**kwargs)


class TimestampedModel(Model):
    """
    Base model for create/update timestamps
    """

    objects = TimestampedModelQuerySet.as_manager()
    created_on = DateTimeField(auto_now_add=True, db_index=True)  # UTC  # noqa: DJ012
    updated_on = DateTimeField(auto_now=True)  # UTC

    class Meta:
        abstract = True


class NoDefaultTimestampedModel(TimestampedModel):
    """
    This model is an alternative for TimestampedModel with one
    important difference: it doesn't specify `auto_now` and `auto_now_add`.
    This allows us to pass in our own values without django overriding them.
    You'd typically use this model when backpopulating data from a source that
    already has values for these fields and then switch to TimestampedModel
    after existing data has been backpopulated.
    """

    created_on = DateTimeField(default=now_in_utc)
    updated_on = DateTimeField(default=now_in_utc)

    class Meta:
        abstract = True


class ConsumerThrottleLimit(Model):
    """
    Model for throttling consumers by a certain rate
    """

    throttle_key = CharField(max_length=255, primary_key=True)
    auth_limit = IntegerField(default=0)
    anon_limit = IntegerField(default=0)
    interval = CharField(
        choices=[
            ("minute", "minute"),
            ("hour", "hour"),
            ("day", "day"),
            ("week", "week"),
        ],
        max_length=12,
    )

    def __str__(self):
        return f"{self.throttle_key} -\
            Auth {self.auth_limit}, \
            Anon {self.anon_limit} \
            per {self.interval}"

    def save(self, **kwargs):
        """Override save to reset the throttles cache"""
        cache.delete(CONSUMER_THROTTLES_KEY)
        cache.close()
        return super().save(**kwargs)
