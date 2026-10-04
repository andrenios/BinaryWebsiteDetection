# -*- coding: utf-8 -*-
def build_html_prompt_v8(html_summary):
    prompt = (
        "You are a cybersecurity expert analysing websites for phishing attempts.\n\n"

        "Your task is to examine the provided HTML code and the original website URL and the website screenshot to determine if the website is likely a phishing site.\n\n"

        "**Important:**\n"
        "Focus on all the information contained in the summary, as it contains the full summary of the website and all information may be important.\n\n"

        "You must also consider the screenshot of the website in your analysis.\n\n"

        f"HTML summary: {html_summary}\n\n"

        "When assessing the likelihood of phishing, consider both your general knowledge of phishing techniques and the evidence you see in the data - focus on HTML structure, content, domain/URL consistency and always consider the provided website screenshot.\n\n"

        "The website screenshot is always provided!\n\n"

        "Below are common examples of phishing indicators you may find useful, but you are not limited to them:\n\n"

        "1. **Suspicious URLs/domains**\n\n"
        "   Check href attributes, form actions, image sources, and compare them with the original URL for:\n\n"
        "   - Misspelt brand names, unusual domains, suspicious subdomains\n"
        "   - IP addresses instead of domains, excessive hyphens, unusual TLDs\n"
        "   - Domain mismatch between original URL and embedded links/forms\n"
        "   - Free hosting / website-builder / dynamic-DNS / sandbox domains (e.g. *.workers.dev, *.web.core.windows.net, *.myqcloud.com, *.plesk.page, *.github.io, *.wixsite.com, *.000webhostapp.com, IPFS/dweb links, raw IPs) hosting a login or a branded page\n\n"

        "2. **Form analysis**\n\n"
        "   Login/input forms with:\n\n"
        "   - Action URLs pointing to domains that do not match the original URL or claimed brand\n"
        "   - Password/sensitive data collection for mismatched domains\n"
        "   - Excessive personal information requests (SSN, full address, etc.)\n\n"

        "3. **Content and language**\n\n"
        "   Text containing:\n\n"
        "   - Urgent threats: 'Account suspended', 'Verify immediately', 'Limited time'\n"
        "   - Fear tactics: 'Security breach', 'Unauthorised access detected'\n"
        "   - Reward baits: 'You have won', 'Free gift', 'Exclusive offer'\n\n"

        "4. **HTML structure issues**\n\n"
        "   - Spelling/grammar errors in text content\n"
        "   - Inconsistent or poor HTML structure\n"
        "   - Missing or suspicious meta tags (title, description)\n\n"

        "5. **Link analysis**\n\n"
        "   Check all href attributes and compare them to the original URL:\n\n"
        "   - Links to different domains than expected\n"
        "   - Shortened URLs (bit.ly, tinyurl, etc.)\n"
        "   - Misleading anchor text vs actual URL\n\n"

        "6. **Brand impersonation**\n\n"
        "   Look for:\n\n"
        "   - Company names or logos in text that don't match the original URL's domain\n"
        "   - References to legitimate services (PayPal, Amazon, banks, webmail/Office/Microsoft, couriers, wallets ...) hosted on unrelated domains\n"
        "   - Official-sounding but incorrect terminology\n\n"

        "7. **Missing legitimacy markers**\n\n"
        "   - No contact information or privacy policy links\n"
        "   - Missing proper company details in footer\n"
        "   - No legitimate copyright notices\n\n"

        "**Weighing the evidence (read carefully to avoid false alarms AND false misses):**\n\n"

        "Phishing fundamentally means the site is trying to STEAL credentials/sensitive data or DECEIVE the user by impersonating a trusted brand.\n\n"

        "Judge each site by whether that core intent is actually present.\n\n"

        "Distinguish strong evidence from weak, ambiguous evidence.\n\n"

        "**STRONG evidence of phishing (these justify a high score):**\n\n"

        "- A login/credential or sensitive-data form (password, banking, ID, card, OTP, security answers) presented under the branding of a well-known company while hosted on a domain that clearly is NOT that company's (off-brand domain, free hosting/dynamic-DNS/website-builder subdomain, IP address, misspelt brand, unrelated TLD).\n\n"

        "- A page that visually clones a known brand's login/sign-in page on a domain unrelated to that brand.\n\n"

        "- A GENERIC login page ('Webmail', 'Mail Portal', 'Sign in', 'Office 365 / SharePoint', account verification) hosted on a random, unrelated, or free-hosting/sandbox domain that is NOT the organisation's own branded domain. These generic mail/office login kits on throwaway domains are one of the most common credential-harvesting patterns and are high-risk EVEN when the form action is same-origin and even when no famous logo is shown.\n\n"

        "- Credential/sensitive fields whose form action posts to the page's OWN domain, WHEN that own domain is itself the impostor (an off-brand/free-host/random/IP/misspelt-brand domain). Same-origin submission is NOT exonerating in this case - the credentials are still being harvested by the attacker's domain.\n\n"

        "- A page that clearly invokes a known brand's identity as a lure (fake parcel/delivery tracking, wallet-connect 'stay on this page / establishing connection', 'reactivate/verify your account' prompts) while hosted on an off-brand free-host/dynamic-DNS/sandbox/plesk/workers.dev/IP domain. The credential-capture step is often one click or one script away; treat these as phishing when the brand impersonation on a clearly unrelated domain is evident.\n\n"

        "- Deceptive reward/prize lures combined with obfuscated scripts or malicious redirect domains, or with off-site posting of user data.\n\n"

        "**IMPORTANT - do not let these excuses cause misses:**\n\n"

        "- 'num_forms=0' / 'no password field' in the HTML summary does NOT prove there is no login. Login forms are frequently generated by JavaScript. If the SCREENSHOT visibly shows credential fields, a sign-in box, or a branded login, treat a credential form as PRESENT and analyse accordingly.\n\n"

        "- 'Form posts to same origin / action is null/#' does NOT mean safe when the origin domain is itself an off-brand impostor domain (see above).\n\n"

        "**WEAK / AMBIGUOUS evidence (these alone should NOT push the score to phishing; treat as low-to-medium at most unless combined with strong evidence above):**\n\n"

        "- High counts of 'external hosts', external scripts, iframes, or 'phishy keyword' links. Large, legitimate sites routinely load many third-party scripts, ads, analytics, and social-media resources (Google, Facebook, Twitter, YouTube, LinkedIn, Yandex). These counts are not by themselves phishing.\n\n"

        "- Presence of analytics/advertising/social-media domains.\n\n"

        "- A newsletter/subscription or contact form that submits ONLY an email (and no password/credentials) to a third-party marketing service. Off-site posting of a plain email field is common and benign.\n\n"

        "- A site collecting email+password on ITS OWN legitimate BRANDED domain with no impersonation of another brand - this is a normal login page, not phishing. (This exception applies only when the domain plausibly belongs to the organisation shown, NOT when it is a generic/random/free-hosting domain.)\n\n"

        "- A regional, subsidiary, vendor, or partner domain that differs from the main brand domain but is plausibly related.\n\n"

        "- Generic hosting-provider pages ('Account Suspended', server error, parked/placeholder), app-store listing pages, or blank/minimal pages with no login and no brand impersonation.\n\n"

        "- Random hashes/GUIDs in URLs, or a single misconfigured/placeholder meta tag - weak signals on their own.\n\n"

        "**Decision guidance:**\n\n"

        "Assign a score of 6 or higher when you can point to concrete credential-harvesting or clear brand-impersonation intent (strong evidence), INCLUDING generic login/webmail/office pages on unrelated throwaway domains and brand-impersonation lures on off-brand free-hosting/sandbox domains.\n\n"

        "When only weak/ambiguous signals are present, keep the score in the 0-4 range and lean towards legitimate.\n\n"

        "When in genuine doubt and there is no credential collection, brand impersonation, or suspicious-hosting login, do not classify as phishing.\n\n"

        "**Note:**\n"
        "The screenshot image is always provided.\n\n"

        "**Scoring guide:**\n\n"
        "- 0-1: Very unlikely phishing (legitimate site)\n"
        "- 2-3: Low risk (minor suspicious elements)\n"
        "- 4-5: Medium risk (several concerning indicators, but no clear credential theft or brand impersonation)\n"
        "- 6-7: High risk (multiple clear phishing signs, including credential harvesting or brand impersonation)\n"
        "- 8-9: Very high risk (obvious phishing attempt)\n\n"

        "**Required output format (JSON only):**\n\n"
        "{\n"
        '  "phishing_score": int [0-9],\n'
        '  "is_phishing": boolean [true/false],\n'
        '  "reasoning": string [Brief explanation of your decision based on specific indicators found]\n'
        "}\n\n"

        "**Output Constraints:**\n\n"
        "Do only output the JSON-formatted output and nothing else.\n"
    )
    return prompt