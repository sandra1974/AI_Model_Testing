#Example: Crafting a marketing email campaign - Vague prompt
#Example: Incident response - Vague prompt

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_vague_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a marketing email for our new AcmeCloud features."}
    ]
)
print(message_vague_prompt.content)

message_vague_prompt_report = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Analyze this AcmeCloud outage report and summarize the key points. " + 
        "\'Artificial intelligence startup Anthropic has reportedly forecast revenues of more than $850 million next year. " +

"That’s according to a report Tuesday (Dec. 26) by The Information, citing a pair of sources with knowledge of the artificial intelligence (AI) company’s financial outlook. " +
" The report also says sources close to Anthropic believe it could even reach $1 billion in annualized revenue by the end of 2024. " +
"The Information also says that Anthropic had told investors three months ago it was generating $100 million in annualized revenue and " +
"expected that figure would climb to $500 million by the end of next year. Reached by PYMNTS, Anthropic declined to comment. " +
"The news comes days after a report that Anthropic was in talks to raise $750 million with the help of Menlo Ventures. " +
"The company, whose co-founder siblings Dario Amodei and Daniela Amodei are veterans of competing AI firm OpenAI, had already landed investments from Amazon and Google. " +
"As noted here in September when Amazon announced its investment, many companies are pouring money into Anthropic, among them the venture arms of both Salesforce and Zoom, " +
"as well as SK Telecom, South Korea’s largest operator. " +
"The generative AI industry itself is expected to reach $1.3 trillion by 2032, and PYMNTS Intelligence has shown that 84% of business leaders " +
"believe this technology will have a positive effect on the workforce. " +
"AI is going to be an imperative for every company, and what you do with AI is what will differentiate your products, Heather Bellini, president and " +
"chief financial officer at InvestCloud, said in an interview with PYMNTS. " +
"Functionally, it might get rid of a lot of the manual work people don’t want to do anyway and extract them up to a level where they can do more things " +
"that have a direct impact on the business. " +
"The news comes amid reports of Big Tech’s dominance in the AI investment field, with companies like Google and Microsoft accounting " +
"for two-thirds of the funds moving into the sector this year, outpacing venture capital (VC) firms. " +
"Over the past year, we’ve seen the market quickly consolidate around a handful of foundation models, with large tech players coming in and " +
"pouring billions of dollars into companies like OpenAI, Cohere, Anthropic and Mistral, Nina Achadjian, a partner at venture firm Index Ventures, told the Financial Times. "
"For traditional VCs, you had to be in early and you had to have conviction — which meant being in the know on the latest AI research and " +
"knowing which teams were spinning out of Google DeepMind, Meta and others, added Achadjian.\'"}
    ]
)
print(message_vague_prompt_report.content)