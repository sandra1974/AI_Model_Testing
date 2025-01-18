#Example: If an ML model is designed to classify emails as either "spam" or
#"not spam," it should consistently provide accurate classifications without
#errors. If emails that are clearly spam are classified as "not spam," it indicates
#a lack of functional completeness.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

classify_email = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1,
    messages=[
        {"role": "user", "content": "Please review this email \'Subject: Swift Response; From: Donor4 <bridg0-18@kiteedge.co.uk; Body: I a m Agnes S c h u l z e, a Munich, Germany r e s i d e n t, and a widow. At 51 years old, I have dedicated my life to humanitarian efforts. Regrettably, I am currently b a t t l i n g breast cancer, and my health i s declining rapidly. My m e d i c a l team has stressed t h e severity of my .condition.It is with a heavy heart that I share the loss of my husband and only child five years a g o My husband's death was politically motivated. He was a prosperous oil m a g n a t e in West Africa, engaged in the trade of oil, gold, and diamonds. After his p a s s i n g, I inherited his b u s i n e s s and wealth.Realizing the limited time I have left, I have chosen to allocate a portion of this wealth to assist the less fortunate in Africa, America, Asia, and Europe. While perusing the internet, I stumbled upon your email address. I have decided to donate $17,500,000.00USD for humanitarian and charitable causes. This sum is currently held in a private bank in Africa, specifically the Standard Trust Bank Ltd, under your supervision.If you are interested in partnering with me on this endeavor, p l e a s e reply for further information via my private email Reply to:  agnesschulz473@gmail.com Thank you, and may you receive abundant b l e s s i n g s. Warm regards,Agnes Schulze.Donor\' Please respond with (A) if email is spam or with (B) if email is not spam." },
        {"role": "assistant", "content": "The answer is ("}
    ]
)
print(classify_email)