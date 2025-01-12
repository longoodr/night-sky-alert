import base64
import json
import os
from typing import Iterable
from zoneinfo import ZoneInfo
import requests
from timezonefinder import TimezoneFinder
from datetime import date, time, datetime, timedelta, timezone
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # For local development only

CHECKED_BODIES = ['Moon', 'Mars', 'Jupiter', 'Saturn']

OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
PUSHOVER_KEY = os.getenv('PUSHOVER_KEY')
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN')
ASTRONOMYAPI_APP_ID = os.getenv('ASTRONOMYAPI_APP_ID')
ASTRONOMYAPI_SECRET = os.getenv('ASTRONOMYAPI_SECRET')

LOCATION_STR = os.getenv('LOCATION')

ASTRONOMYAPI_AUTH_STR = base64.b64encode(f"{ASTRONOMYAPI_APP_ID}:{ASTRONOMYAPI_SECRET}".encode()).decode()

def get_coords(location: str) -> tuple[float, float]:
    GEOCODE_URL = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
    response = requests.get(GEOCODE_URL).json()
    if not response:
        raise ValueError("Could not find coordinates for the given location.")
    return response[0]['lat'], response[0]['lon']

def get_elevation(lat: float, lon: float) -> float:
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    response = requests.get(url).json()
    return response['results'][0]['elevation']


LAT, LON = get_coords(LOCATION_STR)
TZ = ZoneInfo(TimezoneFinder().timezone_at(lat=LAT, lng=LON))
ELEVATION = get_elevation(LAT, LON)
print(f"Location: {LOCATION_STR} -> Lat: {LAT}, Lon: {LON}, Elevation: {ELEVATION}")


def get_forecast(lat: float, lon: float) -> dict:
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url).json()
    return response

def get_bodies(lat: float, lon: float, elevation: float, date: datetime) -> dict:
    date_param = date.strftime('%Y-%m-%d')
    time_param = date.strftime('%H:%M:%S')
    url = f"https://api.astronomyapi.com/api/v2/bodies/positions?from_date={date_param}&to_date={date_param}&time={time_param}&latitude={lat}&longitude={lon}&elevation={elevation}&output=rows"
    response = requests.get(url, headers={"Authorization": f"Basic {ASTRONOMYAPI_AUTH_STR}"}).json()
    return response

def post_notification(message: str):
    url = "https://api.pushover.net/1/messages.json"
    requests.post(url, data={
        "token": PUSHOVER_APP_TOKEN,
        "user": PUSHOVER_KEY,
        "message": message
    })

@dataclass
class DayEntry:
    dt: datetime
    cloud_cover: float
    has_max_visiblility: bool

class DayWeather:
    sunset: datetime
    after_sunset_entries: list[DayEntry]
    
    def __init__(self, forecast: dict):
        self.sunset = datetime.fromtimestamp(forecast['current']['sunset'], timezone.utc).astimezone(TZ)
        self.after_sunset_entries = []
        for entry in forecast['hourly']:
            dt = datetime.fromtimestamp(entry['dt'], timezone.utc).astimezone(TZ)
            if dt.date() != self.sunset.date():
                break
            if dt < self.sunset:
                continue
            cloud_cover = float(entry['clouds'])
            has_max_visiblility = entry['visibility'] >= 10000
            self.after_sunset_entries.append(DayEntry(dt, cloud_cover, has_max_visiblility))
    
    def get_good_viewing_intervals(self) -> list[tuple[datetime, datetime]]:
        intervals = []
        start = None
        for entry in self.after_sunset_entries:
            if entry.cloud_cover < 10 and entry.has_max_visiblility:
                if start is None:
                    start = entry.dt
            else:
                if start is not None:
                    intervals.append((start, entry.dt + timedelta(hours=1)))
                    start = None
        if start is not None:
            intervals.append((start, self.after_sunset_entries[-1].dt + timedelta(hours=1)))
        return intervals

def yield_hours_from_intervals(intervals: list[tuple[time, time]]) -> Iterable[time]:
    for start, end in intervals:
        current = start
        while current < end:
            yield current
            current += timedelta(hours=1)

weather = DayWeather(get_forecast(LAT, LON))
good_intervals = weather.get_good_viewing_intervals()
if not good_intervals:
    print("No good viewing intervals found.")
    exit()
print("Good viewing intervals:")
for start, end in good_intervals:
    formatted_start = start.astimezone(TZ).strftime("%Y-%m-%d %I:%M %p")
    formatted_end = end.astimezone(TZ).strftime("%Y-%m-%d %I:%M %p")
    print(f"{formatted_start} - {formatted_end}")
    
hour_to_good_viewing: dict[datetime, list[str]] = {}
for dt_to_query in yield_hours_from_intervals(good_intervals):
    print(f"Checking {dt_to_query.strftime('%I:%M %p')}")
    hour_to_good_viewing[dt_to_query] = []
    viewable_bodies = get_bodies(LAT, LON, ELEVATION, dt_to_query)
    for viewable_body in viewable_bodies['data']['rows']:
        name = viewable_body['body']['name']
        if name not in CHECKED_BODIES:
            continue
        body_info = viewable_body['positions'][0]
        degs_above_horizon = float(body_info['position']['horizonal']['altitude']['degrees'])
        if degs_above_horizon < 30:
            continue
        print(f"{name} is {degs_above_horizon} degrees above the horizon.")
        if name == 'Moon':
            phase_frac = float(body_info['extraInfo']['phase']['fraction'])
            print(f"Moon phase: {phase_frac}")
            if phase_frac < 0.05 or phase_frac > 0.90:
                continue
        print(f"{name} has good viewing.")
        hour_to_good_viewing[dt_to_query].append(name)

interval_to_views: dict[tuple[datetime, datetime], list[str]] = {}
body_to_start_dt: dict[str, datetime] = {}
finished_bodies = set()

def update_intervals(dt: datetime, viewable_bodies: list[str]):
    bodies_to_finish = body_to_start_dt.keys() - (finished_bodies.union(set(viewable_bodies)))
    for body_to_finish in bodies_to_finish:
        interval = (body_to_start_dt[body_to_finish], dt + timedelta(hours=1))
        if interval not in interval_to_views:
            interval_to_views[interval] = []
        interval_to_views[interval].append(body_to_finish)
        finished_bodies.add(body_to_finish)

for dt, viewable_bodies in hour_to_good_viewing.items():
    for viewable_body in viewable_bodies:
        if viewable_body not in body_to_start_dt:
            body_to_start_dt[viewable_body] = dt
    update_intervals(dt, viewable_bodies)
update_intervals(dt, [])

sorted_intervals = sorted(interval_to_views.items(), key=lambda x: (x[0], x[1]))
message = "Good viewing tonight:"
for interval, bodies in sorted_intervals:
    start, end = interval
    formatted_start = start.strftime("%I:%M %p")
    formatted_end = end.strftime("%I:%M %p")
    body_str = ", ".join(bodies)
    message += f"\n{formatted_start} - {formatted_end}: {body_str}"

post_notification(message)