import socket
import csv
import re
import time
from dotenv import load_dotenv
import os, sys
import argparse
from datetime import datetime

load_dotenv()

parser = argparse.ArgumentParser(description="Scraping bot for stream")

parser.add_argument(
    '--channel',
    type=str,
    default=None,
    help="twitch channel name"
)

parser.add_argument(
    '--data_path', 
    type=str,
    default="./data/chatlogs",
    help="raw file to path"
)

args = parser.parse_args()

stored_time = datetime.now().strftime("%Y%m%d")
SERVER = 'irc.chat.twitch.tv'
PORT = 6667
OAUTH_TOKEN = os.getenv('OAUTH_TOKEN')
CHANNEL = args.channel

# Set up the socket connection
def create_connection():
    s = socket.socket()
    s.connect((SERVER, PORT))
    print(f"Connected to {SERVER} on port {PORT}")

    # Send the authentication details
    s.send("CAP REQ :twitch.tv/tags\r\n".encode('utf-8'))
    s.send(f"PASS {OAUTH_TOKEN}\r\n".encode('utf-8'))
    s.send(f"NICK {CHANNEL}\r\n".encode('utf-8'))
    s.send(f"JOIN #{CHANNEL}\r\n".encode('utf-8'))
    
    # Wait for a response from Twitch (it should send a welcome message or similar)
    response = s.recv(2048).decode('utf-8')
    
    if "Welcome" in response:
        print(f"Successfully connected to the channel #{CHANNEL}!")
    else:
        print(f"Failed to connect: {response}")
    
    return s

# Function to parse messages
def parse_message(message):

    print("Recieved ----")
    print(message)

    # this might not be correct because you want to consider othere motes first maybe w
    if 'emotes=' not in message:
        return None, None, None, None

    tag_split = message.split("PRIVMSG #")
    temp_user = tag_split[0].split(";display-name=")[-1].split(";")[0]
    username = temp_user
    content = tag_split[-1].split(f' :')
    message = content[-1]
    emotes_grab = tag_split[0].split(";emotes=")[-1].split(";")[0]
    emotes = []

    # grab emotes
    if emotes_grab:
        emotes_id = emotes_grab.split("/")

        for i in emotes_id:
            positions = i.split(":")[-1].split(",")
            for p in positions:
                single_pos = p.split("-")
                emotes.append(message[int(single_pos[0]):int(single_pos[-1]) + 1])

        # Return username, message, and any emotes
        return username, message, emotes_id, emotes

    return username, message, [], []

def stream_chat():
    s = create_connection()
    f_message = 0
    store_message = []
    chat_path = f'{args.data_path}/{stored_time}-{args.channel}.csv'
    
    # Make sure the directory exists
    import os
    os.makedirs(os.path.dirname(chat_path), exist_ok=True)
    
    # Open the file with a variable to reference
    with open(chat_path, 'a', encoding='utf-8') as file:
        rf = csv.writer(file, delimiter=',', lineterminator='\n')
        rf.writerow(['message', 'emotes', 'emotes_id', 'username'])
        
        while True:
            try:
                # Receive the message from Twitch server
                response = s.recv(2048).decode('utf-8')
               
                if response.startswith('PING'):
                    # Respond to the server's PING with PONG to keep connection alive
                    s.send("PONG\r\n".encode('utf-8'))
               
                else:
                    # Parse the message
                    username, message, emotes_id, emotes = parse_message(response)
                    KNOWN_BOTS = ['StreamElements', 'Nightbot', 'Moobot', 'Streamlabs', 'KofiStreamBot','Sery_bot']
                    
                    # Case clauses, if message is valid and not bot
                    if username and message and 'bot' not in username.lower() and len(message.split()) > 1 and username not in KNOWN_BOTS:
                        
                        print(f'{username}: {message}')
                        print(f'Emotes:{emotes}')
                        temp_results = [message.strip(), str(emotes), str(emotes_id), username]
                        
                        print("written")
                        print(temp_results)
                        rf.writerow(temp_results)
                        
                        # Flush the file after each write
                        file.flush()
                
                # Sleep to avoid overloading the server
                time.sleep(0.1)
           
            except socket.error:
                print("Connection lost, reconnecting...")
                time.sleep(5)
                s = create_connection()
                
                # Reopen the file if connection was lost
                file.close()
                file = open(chat_path, 'a')
                rf = csv.writer(file, delimiter=',', lineterminator='\n')


if __name__ == "__main__":
    stream_chat()