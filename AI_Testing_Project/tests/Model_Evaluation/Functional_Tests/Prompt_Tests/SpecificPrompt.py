#Example: Crafting a marketing email campaign - Specific prompt

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_clear_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Your task is to craft a targeted marketing email for our Q3 AcmeCloud feature release. Instructions: " +
                        "1. Write for this target audience: Mid-size tech companies (100-500 employees) upgrading from on-prem to cloud. " +
                        "2. Highlight 3 key new features: advanced data encryption, cross-platform sync, and real-time collaboration. " +
                        "3. Tone: Professional yet approachable. Emphasize security, efficiency, and teamwork. " +
                        "4. Include a clear CTA: Free 30-day trial with priority onboarding. " +
                        "5. Subject line: Under 50 chars, mention \'security\' and \'collaboration\'. " +
                        "6. Personalization: Use {{COMPANY_NAME}} and {{CONTACT_NAME}} variables." +
                        "Structure: " +
                        "1. Subject line " +
                        "2. Email body (150-200 words) " +
                        "3. CTA button text"}     
            ]
)
print(message_clear_prompt.content)
