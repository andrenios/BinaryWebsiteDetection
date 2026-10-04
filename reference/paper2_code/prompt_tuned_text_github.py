# -*- coding: utf-8 -*-
def build_html_prompt_v8(html_summary):
    prompt = (
        "You are a cybersecurity expert analysing websites for phishing attempts.\n\n"

        "Your task is to examine the provided JSON-style summary of a website's HTML code and URL to determine if the website is likely a phishing site or not.\n\n"

        "**What phishing IS:**\n"
        "A site that impersonates a trusted brand, service, or person in order to steal credentials, payment details, or personal identity information (e.g. a fake login/verification/payment page, credential forms that post to an unrelated domain, homograph/typosquatted brand domains hosting login pages).\n\n"

        "**What phishing is NOT:**\n"
        "Low-quality, spammy, or distasteful content is not automatically phishing. Piracy/streaming sites, cracked-software or 'unlocker' sites, spyware/tracking-tool vendors, 'fake document' sellers, SEO spam, keyword-stuffed pages, ad-heavy pages, and generic customer-service/info pages may be undesirable or even scams, but should NOT be scored as phishing unless they also show clear credential/payment/identity harvesting via impersonation.\n\n"

        "Do not classify a site as phishing solely because it is untrustworthy or sells something dubious.\n\n"

        "**Important:**\n"
        "Focus on all the information contained in the summary, as it contains the full summary of the website and all information may be important.\n\n"

        f"HTML summary: {html_summary}\n\n"

        "When assessing the likelihood of phishing, consider both your general knowledge of phishing techniques and the evidence you see in the data - focus on HTML structure, content and domain/URL consistency.\n\n"

        "Below are common examples of phishing indicators you may find useful, but you are not limited to them:\n\n"

        "1. **Suspicious URLs/domains**\n\n"
        "   Check href attributes, form actions, image sources, and compare them with the original URL for:\n\n"
        "   - Misspelt/typosquatted brand names, homograph (punycode) domains, brand names placed in an unrelated domain or subdomain\n"
        "   - Domain mismatch between the original URL and embedded login/payment forms\n\n"

        "2. **Form analysis**\n\n"
        "   Login/input forms with:\n\n"
        "   - Action URLs pointing to domains that do not match the original URL or claimed brand (off-site credential posting is a strong signal)\n"
        "   - Password/sensitive data collection on a page impersonating a brand it does not belong to\n"
        "   - Excessive or unusual sensitive-data collection (SSN, full card details, multiple hidden credential fields)\n\n"

        "3. **Content and language**\n\n"
        "   Text containing:\n\n"
        "   - Urgent threats, fear tactics, or verification pressure ('Account suspended', 'Verify immediately', 'Wrong password')\n"
        "   - Instructions to log in to a brand while the page is hosted on an unrelated domain\n\n"

        "4. **HTML structure issues**\n\n"
        "   - Spelling/grammar errors combined with brand impersonation\n"
        "   - Obfuscated inline scripts (high entropy, atob/eval/unescape) on a login-style page\n\n"

        "5. **Brand impersonation**\n\n"
        "   Look for a mismatch between a claimed brand (title, logo asset, meta) and the actual hosting domain ON A PAGE THAT COLLECTS OR REQUESTS CREDENTIALS/PAYMENT.\n\n"
        "   A mere reference to a brand on an informational, help, or listing page (without credential collection) is a much weaker signal.\n\n"

        "**Signals that are WEAK on their own — do NOT drive a high score without corroborating impersonation + harvesting evidence:**\n\n"

        "- An unusual or cheap TLD (.tk, .gq, .cf, .ru, .top, .online, .xyz, .org, .eu.org, etc.). Legitimate sites use these too.\n"
        "- A non-brand, unfamiliar, or niche domain name, or hosting on a subdomain/platform (github.io, netlify.app, weebly, workers.dev, appspot). Suspicious only when combined with impersonation or credential harvesting.\n"
        "- A high count of 'phishy keyword links', many external hosts, or many third-party scripts. These metrics are noisy and also fire on legitimate banks, government portals, and content sites.\n"
        "- Missing legitimacy markers (no privacy policy/contact/footer) alone.\n"
        "- Executable file in the URL, or generic 404/parked pages, without impersonation.\n\n"

        "Treat these weak signals as raising suspicion modestly; require at least one strong signal (credential/payment form + brand or domain mismatch, off-site credential posting, typosquatted/homograph brand login domain, or explicit login-verification social engineering tied to a mismatched host) before scoring in the phishing range.\n\n"

        "**Scoring guide:**\n\n"
        "- 0-1: Very unlikely phishing (legitimate site)\n"
        "- 2-3: Low risk (only weak signals such as odd TLD, niche domain, keyword-link counts, or dubious-but-non-impersonating content)\n"
        "- 4-5: Medium risk (several concerning indicators, or one strong signal with ambiguity)\n"
        "- 6-7: High risk (clear brand/domain impersonation on a credential/payment page, or off-site credential posting)\n"
        "- 8-9: Very high risk (obvious, unambiguous credential/identity harvesting via impersonation)\n\n"

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
