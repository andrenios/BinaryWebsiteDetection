def build_html_prompt_v6_features(html_text, original_url):
    prompt = (
        "You are a cybersecurity expert assisting in website phishing detection. "
        "Your task is NOT to decide if a site is phishing or legitimate directly, "
        "but to extract structured, standardised **semantic and perceptual features** "
        "from the provided website data (HTML code, original URL, and website screenshot). "
        "A separate machine learning model will later analyse these features. \n\n"

        "**Important context:**\n"
        "- The HTML may be truncated to reduce cost; CSS styles and JavaScript may be missing.\n"
        "- Focus on meaningful HTML tags, visible text content, and linked resources.\n"
        "- The screenshot image of the website is always provided.\n"
        "- You will describe observed characteristics, not conclusions.\n\n"

        f"Original URL: {original_url}\n"
        f"HTML snippet (truncated): '{html_text}'\n\n"
        
        "When extracting features, carefully consider these categories:\n"
        "1. **Brand and identity cues** – look for visible or textual signs of brand impersonation.\n"
        "2. **Page intent** – what is the main purpose or action requested (login, shopping, info, etc.).\n"
        "3. **Trust or security indicators** – fake padlocks, 'secure' text, warning banners, or absence of contact info.\n"
        "4. **Domain-content consistency** – does the content match the brand or domain in the URL?\n"
        "5. **Cross-modal consistency** – does the brand and purpose shown in the screenshot agree with the HTML text?\n"
        "6. **Page structure and visual layout** – whether the page appears as a login form, banner, content page, etc.\n"
        "7. **Screenshot details** – presence of logos, login forms, buttons, or alerts visible in the image.\n\n"

        "Do not provide a risk score or classification — only fill in the required structured fields below.\n\n"

        "**Required JSON output schema:**\n"
        "{\n"
        '  "brand_mentioned": boolean,                    [Any brand or organisation mentioned or visually identifiable]\n'
        '  "logo_present": boolean,                       [Visible logo or brand symbol in screenshot]\n'
        '  "domain_vs_content_match": boolean,            [Whether brand/logo/content matches the original URL domain]\n'
        '  "external_favicon": boolean,                   [Presence of an external favicon in HTML]'
        '  "login_form_visible": boolean,                 [Presence of login/password form in screenshot or HTML]\n'
        '  "cta_button_present": boolean,                 [Prominent sign in, verify, continue, ... button]\n'
        '  "urgency_pressure": "none"|"mild"|"strong",    [Degree of manufactured urgency or threat in the wording, e.g. account suspended, verify immediately, ...]\n'
        '  "screenshot_text_brand_agreement": "agree"|"disagree"|"not_enough_info",  [Whether the brand/purpose shown in the screenshot matches the HTML text]\n'
        '  "visual_layout_type": "login_form_centered"|"banner"|"content"|"error"|"other"  [General layout structure]\n'
        "}\n\n"

        "**Guidelines:**\n"
        "- Use only the provided evidence (URL, HTML code, screenshot) — no external lookup.\n"
        "- Judge 'urgency_pressure' on the tone and wording only, not on whether the site seems trustworthy.\n"
        "- If a value cannot be confidently determined, set it to `null` (for strings) or `false` (for booleans).\n"
        "- Remain objective: do not infer intent beyond observable features.\n"
        "- Output only valid JSON conforming to the schema — no extra text, commentary, or formatting.\n"
    )
    return prompt
