import paho.mqtt.client as mqtt
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
from siamese_network import SiameseNetwork, load_siamese_model

MQTT_BROKER = "localhost"
INPUT_TOPIC = "image/query"
OUTPUT_TOPIC = "image/top_matches"

device = "cpu"
model = load_siamese_model("siamese_model.pth", device)
model.to(device

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(INPUT_TOPIC)


def on_message(client, userdata, msg):
    if msg.topic == INPUT_TOPIC:
        print("Image query received.")
        query_image_data = base64.b64decode(msg.payload)
        query_image = Image.open(io.BytesIO(query_image_data)).convert("RGB")
        query_embedding = get_embedding(model, query_image, transform)

        database = load_database()
        top_matches = find_top_matches(query_embedding, database)

        client.publish(OUTPUT_TOPIC, json.dumps(top_matches))
        print("Top matches sent.")

def get_embedding(model, image, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(image_tensor)
    return embedding


def find_top_matches(query_embedding, database, top_k=5):
    distances = []
    for db_entry in database:
        db_embedding, db_path = db_entry
        distance = torch.nn.functional.pairwise_distance(query_embedding, db_embedding)
        distances.append((db_path, distance.item()))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]


def load_database():
    database = []
    return database

if __name__ == "__main__":
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_forever()
    except Exception as ex:
        print(f"Exception: {ex}")

