from django.core.cache import cache
from django.db import migrations

from main.constants import CONSUMER_THROTTLES_KEY


def set_support_rate_limits(apps, schema_editor):
    """Populate initial rate limits for the support consumer"""
    ConsumerThrottleLimit = apps.get_model("main", "ConsumerThrottleLimit")
    ConsumerThrottleLimit.objects.get_or_create(
        throttle_key="support_bot",
        defaults={
            "auth_limit": 100,
            "anon_limit": 50,
            "interval": "day",
        },
    )
    # Historical models don't call the custom save(), so reset the cache here
    cache.delete(CONSUMER_THROTTLES_KEY)


def remove_support_rate_limits(apps, schema_editor):
    """Remove the rate limits for the support consumer"""
    ConsumerThrottleLimit = apps.get_model("main", "ConsumerThrottleLimit")
    ConsumerThrottleLimit.objects.filter(throttle_key="support_bot").delete()
    cache.delete(CONSUMER_THROTTLES_KEY)


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0003_search_summary_rate_limit"),
    ]

    operations = [
        migrations.RunPython(
            set_support_rate_limits,
            reverse_code=remove_support_rate_limits,
        )
    ]
