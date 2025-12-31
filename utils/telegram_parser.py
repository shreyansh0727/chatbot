import json

class TelegramParser:
    def __init__(self):
        self.conversations = []
    
    def parse_json_file(self, json_filepath):
        """Parse Telegram JSON export"""
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        messages = data.get('messages', [])
        
        # Extract text messages only
        text_messages = []
        for msg in messages:
            if msg.get('type') == 'message':
                # Extract text content
                text = self._extract_text(msg)
                
                if text and text.strip():
                    text_messages.append({
                        'id': msg.get('id'),
                        'date': msg.get('date'),
                        'from': msg.get('from', 'Unknown'),
                        'from_id': msg.get('from_id', ''),
                        'text': text.strip()
                    })
        
        return text_messages
    
    def extract_conversations(self, json_filepath, your_name_or_id):
        """
        Extract conversations where others send messages 
        and YOU respond
        """
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = data.get('messages', [])
        conversations = []
        
        # Filter and parse messages
        parsed_messages = []
        for msg in messages:
            if msg.get('type') == 'message':
                text = self._extract_text(msg)
                sender = msg.get('from', '')
                sender_id = msg.get('from_id', '')
                
                if text and text.strip():
                    parsed_messages.append({
                        'sender': sender,
                        'sender_id': sender_id,
                        'text': text.strip(),
                        'date': msg.get('date')
                    })
        
        # Pair messages: someone asks, you respond
        for i in range(len(parsed_messages) - 1):
            current = parsed_messages[i]
            next_msg = parsed_messages[i + 1]
            
            # Check if you're the responder
            is_you_next = (next_msg['sender'] == your_name_or_id or 
                          str(next_msg['sender_id']) == str(your_name_or_id))
            is_you_current = (current['sender'] == your_name_or_id or 
                            str(current['sender_id']) == str(your_name_or_id))
            
            if not is_you_current and is_you_next:
                conversations.append({
                    'user': current['text'],
                    'assistant': next_msg['text']
                })
        
        return conversations
    
    def _extract_text(self, message):
        """Extract text from Telegram message (handles various formats)"""
        text_content = message.get('text', '')
        
        # Text can be string or array of objects
        if isinstance(text_content, str):
            return text_content
        elif isinstance(text_content, list):
            # Concatenate text pieces
            text_parts = []
            for item in text_content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text_parts.append(item.get('text', ''))
            return ''.join(text_parts)
        
        return ''
    
    def save_to_json(self, conversations, output_filepath):
        """Save parsed conversations to JSON"""
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(conversations)} conversations to {output_filepath}")
