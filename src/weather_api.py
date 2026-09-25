"""Weather API Module.

Fetches real-time weather from OpenWeatherMap API, with graceful fallback
and mock simulation when API keys are absent or network requests fail.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_weather_data(city: str = None) -> dict:
    """Fetch current weather data for the specified city.
    
    Falls back to a simulated agricultural weather report if OPENWEATHER_API_KEY
    is not provided, expired, or network request fails.
    
    Args:
        city: City name (defaults to DEFAULT_LOCATION or 'Bengaluru')
        
    Returns:
        dict: Weather parameters including temperature, humidity, wind, and description.
    """
    if not city:
        city = os.getenv("DEFAULT_LOCATION", "Bengaluru")
        
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    
    # If API key is present and not dummy placeholder, attempt live API call
    if api_key and api_key != "your_openweather_api_key_here":
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_info = {
                    "city": data.get("name", city),
                    "country": data.get("sys", {}).get("country", "IN"),
                    "temperature_c": round(data["main"]["temp"], 1),
                    "feels_like_c": round(data["main"]["feels_like"], 1),
                    "humidity_pct": data["main"]["humidity"],
                    "pressure_hpa": data["main"]["pressure"],
                    "wind_speed_ms": data["wind"]["speed"],
                    "condition": data["weather"][0]["main"] if data.get("weather") else "Clear",
                    "description": data["weather"][0]["description"].capitalize() if data.get("weather") else "Clear sky",
                    "source": "OpenWeatherMap Live API"
                }
                return weather_info
            else:
                # Log issue silently and fall back
                fallback_reason = f"OpenWeatherMap returned HTTP {response.status_code}"
        except Exception as e:
            fallback_reason = f"Network exception: {str(e)}"
    else:
        fallback_reason = "No API Key provided (using realistic agrometeorological simulation)"

    # Fallback simulated live weather
    return {
        "city": city,
        "country": "IN",
        "temperature_c": 28.4,
        "feels_like_c": 29.8,
        "humidity_pct": 65,
        "pressure_hpa": 1012,
        "wind_speed_ms": 3.8,
        "condition": "Scattered Clouds",
        "description": "Scattered clouds with mild breeze, moderate evapotranspiration",
        "source": f"Simulated Fallback ({fallback_reason})"
    }


def format_weather_summary(weather_data: dict) -> str:
    """Format weather data dictionary into an agronomic context string.
    
    Args:
        weather_data: Dictionary returned by get_weather_data
        
    Returns:
        str: Concise readable weather advisory for LLM and farmer display.
    """
    summary = (
        f"Live Weather at {weather_data['city']}, {weather_data['country']} ({weather_data['source']}):\n"
        f"- Condition: {weather_data['description']}\n"
        f"- Temperature: {weather_data['temperature_c']}°C (Feels like {weather_data['feels_like_c']}°C)\n"
        f"- Relative Humidity: {weather_data['humidity_pct']}%\n"
        f"- Wind Speed: {weather_data['wind_speed_ms']} m/s\n"
        f"- Atmospheric Pressure: {weather_data['pressure_hpa']} hPa\n"
        f"- Agronomic Note: "
    )
    
    # Agronomic guidance based on weather
    temp = weather_data['temperature_c']
    humidity = weather_data['humidity_pct']
    if humidity > 80:
        summary += "High humidity increases fungal blight risk. Avoid overhead irrigation."
    elif temp > 35:
        summary += "High heat stress potential. Ensure afternoon moisture availability."
    elif humidity < 40:
        summary += "Dry conditions. Consider mulch retention and drip schedule optimization."
    else:
        summary += "Favorable growing conditions with moderate transpiration rate."
        
    return summary
