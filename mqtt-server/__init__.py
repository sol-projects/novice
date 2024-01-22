import paho.mqtt.client as mqtt
import subprocess
import base64
import requests
import json

MQTT_BROKER = "localhost"
INPUT_TOPIC = "image/find-text-input"
OUTPUT_TOPIC = "image/find-text-response"

ADD_NEWS_TOPIC = "news/add"
NEWS_API_URL = "http://localhost:8000/news/"
LOGIN_URL = "http://localhost:8000/news/login"

def get_jwt_token():
    print("Getting JWT token.")
    response = requests.post(LOGIN_URL)

    if response.status_code == 200:
        return response.json().get("token")
    else:
        print(f"Failed to get JWT token. Status code: {response.status_code}")
        return None

def post_news_article(article_data, jwt_token):
    headers = {"Authorization": f"Bearer {jwt_token}"}
    response = requests.post(NEWS_API_URL, json=article_data, headers=headers)

    if response.status_code == 201:
        print("News article posted successfully.")
    else:
        print(f"Failed to post news article. Status code: {response.status_code}")

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(INPUT_TOPIC)
    client.subscribe(ADD_NEWS_TOPIC)

def on_message(client, userdata, msg):
    if msg.topic == INPUT_TOPIC:
        print("Image received, processing...")
        transformed_image = transform_image(msg.payload)
        client.publish(OUTPUT_TOPIC, transformed_image)
        print("Transformed image published.")
    if msg.topic == ADD_NEWS_TOPIC:
        print("Posting news article.")
        payload_json = msg.payload.decode('utf-8')
        json_object = json.loads(payload_json)
        print(json_object)
        post_news_article(json_object, get_jwt_token())
        print("Posted news article.")

def transform_image(image_data):
    image_data = base64.b64decode(image_data)
    with open('../newspaper_to_digital/input.png', 'wb') as f:
        f.write(image_data)

    try:
        subprocess.check_output(
            'source env/bin/activate && python __init__.py --eval',
            cwd='../newspaper_to_digital',
            shell=True
        )
    except subprocess.CalledProcessError as e:
        print(e)
        return None

    try:
        with open('../newspaper_to_digital/output.png', 'rb') as f:
            output_image_data = f.read()
    except FileNotFoundError as e:
        print(e)
        return None

    return base64.b64encode(output_image_data)



if __name__ == "__main__":
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_forever()
    except Exception as ex:
        print(f"Exception: {ex}")
    raise

