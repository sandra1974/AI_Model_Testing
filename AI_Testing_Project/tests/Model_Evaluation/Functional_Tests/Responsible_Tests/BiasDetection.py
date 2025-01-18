#Example: A predictive policing model could be tested to see if its use disproportionately affects certain communities.
#Societal impact testing would check if such an algorithm reinforces existing inequalities or creates new problems.

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_bias_detection_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
           {"role": "user", "content": "Computer-aided diagnosis (CAD) systems have been found to return lower accuracy results for black patients than white patients. Comment this. "},
            {"role": "user", "content": "Amazon stopped using a hiring algorithm after finding it favored applicants based on words like \'executed\' or \'captured\' which were more commonly found on men’s resumes. Comment this."},
              {"role": "user", "content": "Google online advertising system displayed high-paying positions to males more often than to women. Comment this."},

            ]
)
print(message_bias_detection_prompt.content)

