from datetime import datetime
from zoneinfo import ZoneInfo


def format_et(ts_iso: str) -> str:
    dt_utc = datetime.fromisoformat(ts_iso)
    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    return dt_et.strftime("Updated %b %d, %Y · %I:%M %p ET")
