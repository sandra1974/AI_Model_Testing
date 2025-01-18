#Example: Enhancing IT support consistency

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_xml_output = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "You are our IT Support AI that draws on knowledge base data. Here are entries from your knowledge base: " +
        "<kb> " +
        "<entry> " +
        "<id>1</id> " +
        "<title>Reset Active Directory password</title> " +
        "<content>1. Go to password.ourcompany.com " +
        "2. Enter your username " +
        "3. Click \'Forgot Password\' " +
        "4. Follow email instructions</content> " +
        "</entry> " +
        "<entry> " +
        "<id>2</id> " +
        "<title>Connect to VPN</title> " +
        "<content>1. Install GlobalProtect from software center " +
        "2. Open GlobalProtect, enter \'vpn.ourcompany.com\' " +
        "3. Use AD credentials to log in</content> " +
        "</entry> " +
        "</kb> " +
        "When helping users, always check the knowledge base first. Respond in this format: " +
        "<response> " +
        "<kb_entry>Knowledge base entry used</kb_entry> " +
        "<answer>Your response</answer> "
        "</response> " +
        "Write some test questions for yourself and answer them using the knowledge base, just to make sure you understand how to use the knowledge base properly."
         }     
        ]
)
print(message_xml_output.content)