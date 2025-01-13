import requests
import json

def main():
    while True:
        user_input = input("Enter data to add to blockchain (or 'quit' to exit): ").strip()
        if user_input.lower() == 'quit':
            break

        r = requests.post('http://127.0.0.1:5000/add_block', data={'data': user_input})
        if r.status_code == 200:
            print("Mined Block:")
            print(json.dumps(r.json(), indent=4))
        else:
            print("Server response:", r.text)

        chain_response = requests.get('http://127.0.0.1:5000/chain')
        print("Current Blockchain:")
        print(json.dumps(chain_response.json(), indent=4))
        print("-" * 40)

if __name__ == '__main__':
    main()

