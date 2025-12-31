import re
import json
from datetime import datetime

class WhatsAppParser:
    def __init__(self):
        # Regex pattern to match WhatsApp messages
        # Handles formats like: "12/30/25, 10:30 AM - Name: Message"
        self.pattern = re.compile(
            r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[APap][Mm]?)\s?-\s?([^:]+):\s(.+)'
        )
        
    def parse_txt_file(self, txt_filepath):
        """Parse WhatsApp chat txt file"""
        conversations = []
        
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_message = None
        previous_sender = None
        user_messages = []
        bot_messages = []
        
        for line in lines:
            match = self.pattern.match(line.strip())
            
            if match:
                timestamp, sender, message = match.groups()
                
                # Clean up sender name
                sender = sender.strip()
                message = message.strip()
                
                # Skip system messages
                if self._is_system_message(message):
                    continue
                
                # Store message with metadata
                current_message = {
                    'timestamp': timestamp,
                    'sender': sender,
                    'message': message
                }
                
                # Group conversations by alternating senders
                if previous_sender and previous_sender != sender:
                    # This is a response
                    if user_messages and bot_messages:
                        conversations.append({
                            'user': user_messages[-1],
                            'assistant': message
                        })
                
                if sender == previous_sender:
                    # Same person continuing
                    if bot_messages:
                        bot_messages[-1] += ' ' + message
                else:
                    user_messages.append(message)
                    bot_messages.append('')
                
                previous_sender = sender
                
            else:
                # Continuation of previous message (multiline)
                if current_message and line.strip():
                    current_message['message'] += ' ' + line.strip()
                    if bot_messages:
                        bot_messages[-1] += ' ' + line.strip()
        
        return conversations
    
    def parse_and_extract_your_messages(self, txt_filepath, your_name):
        """
        Extract only YOUR messages as assistant responses
        and others' messages as user inputs
        """
        conversations = []
        
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        messages = []
        current_msg = None
        
        # First pass: extract all messages
        for line in lines:
            match = self.pattern.match(line.strip())
            
            if match:
                timestamp, sender, message = match.groups()
                sender = sender.strip()
                message = message.strip()
                
                if not self._is_system_message(message):
                    current_msg = {
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message
                    }
                    messages.append(current_msg)
            else:
                # Multiline message continuation
                if current_msg and line.strip():
                    current_msg['message'] += ' ' + line.strip()
        
        # Second pass: pair user questions with your responses
        for i in range(len(messages) - 1):
            current = messages[i]
            next_msg = messages[i + 1]
            
            # If someone asks and YOU respond
            if current['sender'] != your_name and next_msg['sender'] == your_name:
                conversations.append({
                    'user': current['message'],
                    'assistant': next_msg['message']
                })
        
        return conversations
    
    def _is_system_message(self, message):
        """Detect WhatsApp system messages"""
        system_keywords = [
            'Messages and calls are end-to-end encrypted',
            'added', 'removed', 'left', 'changed',
            'created group', 'image omitted', 'video omitted',
            'audio omitted', 'sticker omitted', 'document omitted',
            'GIF omitted', 'deleted this message'
        ]
        return any(keyword in message.lower() for keyword in system_keywords)
    
    def save_to_json(self, conversations, output_filepath):
        """Save parsed conversations to JSON"""
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(conversations)} conversations to {output_filepath}")
