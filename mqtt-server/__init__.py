import paho.mqtt.client as mqtt
import subprocess
import base64

MQTT_BROKER = "localhost"
INPUT_TOPIC = "image/find-text-input"
OUTPUT_TOPIC = "image/find-text-response"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(INPUT_TOPIC)

def on_message(client, userdata, msg):
    if msg.topic == INPUT_TOPIC:
        print("Image received, processing...")
        transformed_image = transform_image(msg.payload)
        client.publish(OUTPUT_TOPIC, transformed_image)
        print("Transformed image published.")

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
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_forever()
