def build_html_prompt_v6_t_sc(html_summary):
    prompt = (
        "You are a cybersecurity expert analysing websites for phishing attempts. Your task is to examine the provided HTML code and the original website URL and the website screenshot to determine if the website is likely a phishing site.\n\n"

        "**Important:** Focus on all the information contained in the summary, as it contains the full summary of the website and all information may be important. You must also consider the screenshot of the website in your analysis.\n\n"
        f"HTML summary: {html_summary}\n"
        
        "When assessing the likelihood of phishing, consider both your general knowledge of phishing techniques and the evidence you see in the data - "
        "focus on HTML structure, content, domain/URL consistency and always consider the provided website screenshot.\n"
        "The website screenshot is always provided!\n"
        "Below are common examples of phishing indicators you may find useful, but you are not limited to them:\n"
        
        "1. **Suspicious URLs/domains** - Check href attributes, form actions, image sources, and compare them with the original URL for:\n"
        "   - Misspelt brand names, unusual domains, suspicious subdomains\n"
        "   - IP addresses instead of domains, excessive hyphens, unusual TLDs\n"
        "   - Domain mismatch between original URL and embedded links/forms\n"
        "2. **Form analysis** - Login/input forms with:\n"
        "   - Action URLs pointing to domains that do not match the original URL or claimed brand\n"
        "   - Password/sensitive data collection for mismatched domains\n"
        "   - Excessive personal information requests (SSN, full address, etc.)\n"
        "3. **Content and language** - Text containing:\n"
        "   - Urgent threats: 'Account suspended', 'Verify immediately', 'Limited time'\n"
        "   - Fear tactics: 'Security breach', 'Unauthorised access detected'\n"
        "   - Reward baits: 'You have won', 'Free gift', 'Exclusive offer'\n"
        "4. **HTML structure issues**:\n"
        "   - Spelling/grammar errors in text content\n"
        "   - Inconsistent or poor HTML structure\n"
        "   - Missing or suspicious meta tags (title, description)\n"
        "5. **Link analysis** - Check all href attributes and compare them to the original URL:\n"
        "   - Links to different domains than expected\n"
        "   - Shortened URLs (bit.ly, tinyurl, etc.)\n"
        "   - Misleading anchor text vs actual URL\n"
        "6. **Brand impersonation** - Look for:\n"
        "   - Company names or logos in text that don't match the original URL's domain\n"
        "   - References to legitimate services (PayPal, Amazon, banks, ...) hosted on unrelated domains\n"
        "   - Official-sounding but incorrect terminology\n"
        "7. **Missing legitimacy markers**:\n"
        "   - No contact information or privacy policy links\n"
        "   - Missing proper company details in footer\n"
        "   - No legitimate copyright notices\n\n"

        "**Note:** The screenshot image is always provided.\n"
         "**Scoring guide:**\n"
        "- 0-1: Very unlikely phishing (legitimate site)\n"
        "- 2-3: Low risk (minor suspicious elements)\n"
        "- 4-5: Medium risk (several concerning indicators)\n"
        "- 6-7: High risk (multiple clear phishing signs)\n"
        "- 8-9: Very high risk (obvious phishing attempt)\n\n"

        "**Required output format (JSON only):**\n"
        "{\n"
        '  "phishing_score": int [0-9],\n'
        '  "is_phishing": boolean [true/false],\n'
        '  "reasoning": string [Detailed explanation of your decision based on specific indicators found]\n'
        "}\n\n"
        
        "**Output Constraints:**\n"
        "Do only output the JSON-formatted output and nothing else.\n"
    )
    return prompt
