#Example: Incident response - Detailed prompt

import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv('$ANTHROPIC_API_KEY')
)

message_clear_prompt = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "AcmeCloud outage report. Skip the preamble. " +
                                    "Keep your response terse and write only the bare bones necessary information. List only: " +
                                    "1) Cause " +
                                    "2) Duration " +
                                    "3) Impacted services " +
                                    "4) Number of affected users " +
                                    "5) Estimated revenue loss. " +
                                    "Here’s the report: " +
                                    "\'ACME Cloud Centre (China) " +
                                    "ACME Universal Communications " +
                                    "5/F, GDC Building, 9 Gaoxin Central Avenue 3rd, Nanshan District " +
                                    "N/A Shenzhen " +
                                    "China " +
                                    "We are a leading service company in the fields of telecoms in China and is able to provide advice and " +
                                    "assistance with the conception and implementation of solutions involving the deployment of telecommunications technologies. " +
                                    "Our group has obtained China ISP License & Hong Kong ISP License which enables us to operate China ISP services, " +
                                    "Hong Kong ISP services and Cross broader Connectivity Services. " +
                                    "We offer both consulting, supporting and procurement services. " +
                                    "Cloud Centre and Computing: " +
                                    "*Cloud Hosting, Computing, Severs farm, Storage and United Communication System " +
                                    "*Internet Exchange [Neutral Data Center " +
                                    "*Co-location & Hosted IT Services [Dedicated, Virtual and SaaS] " +
                                    "*IT Infrastructure & Data Centre Turnkey Services " +
                                    "Connectivity Services: " +
                                    "*Cross Border Connectivity Services " +
                                    "*Metropolitan Network " +
                                    "*China DIA (Dedicated Internet Access) " +
                                    "*China IPLC, MPLS VPN " +
                                    "*China Static IP Broadband and Transit " +
                                    "Managed IT Professional Services: " +
                                    "*Private Cloud, Virtualization and Consolidation (Vmware, Hyper-v) " +
                                    "*Enterprise Storage, DR and Backup " +
                                    "*Managed Security Protection and Reporting " +
                                    "*United Communication Service and System " +
                                    "*7x24x4 Supports coverage Great China (Hong Kong, China, Macau, Taiwan) " +
                                    "Our Solution Partners " +
                                    "http://www.acmehk.net/partners/solution-partners/ " +
                                    "Our Backbone & Backhaul " +
                                    "http://www.acmehk.net/network/security-operation-centre-soc/ " +
                                    "Connectivity Service " +
                                    "http://www.acmehk.net/solutions/connectivity/global-connectivity-services/ " +
                                    "If you are looking for colocation, cloud, connectivity or other services in ACME Cloud Centre (China), " +
                                    "other data centers in Shenzhen or operated by ACME Universal Communications, please try our free quote service " +
                                    "or reach out for a free consultation about your data center needs.\' "}     
            ]
)
print(message_clear_prompt.content)