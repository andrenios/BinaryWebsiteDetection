# -*- coding: utf-8 -*-


def build_teacher_prompt(used_prompt, performance_measures, reasons, modality):
    # What the student model had access to, injected so the teacher does not add
    # instructions about signals the student cannot see. This single line is the
    # only thing that differs across the text-only / image-only / both runs.
    modality_note = {
        "text":  "The student sees ONLY a summarised HTML representation of the website and its URL. "
                 "It does NOT see a screenshot. Do not add instructions that rely on visual appearance.",
        "image": "The student sees ONLY a screenshot image of the rendered website. "
                 "It does NOT see any HTML, text, or URL. Do not add instructions that rely on HTML, source code, or URL inspection.",
        "both":  "The student sees BOTH a summarised HTML representation of the website (plus its URL) AND a screenshot image of the rendered website. "
                 "You may use instructions that rely on either or both signals.",
    }[modality]

    
    prompt = (
        "You are an expert in designing prompts for other LLMs that are tasked with phishing website detection.\n"
        "Your job is to iteratively improve a prompt so that the other LLM (the 'student') detects phishing websites more accurately.\n\n"
 
        "You are given the prompt currently used by the student, the performance it achieved with that prompt, "
        "and the student's own reasoning on the cases it got wrong (plus some it got right).\n\n"
        
        f"=== WHAT THE STUDENT CAN SEE ===\n{modality_note}\n\n"
        
        "=== CURRENT STUDENT PROMPT ===\n"
        f"{used_prompt}\n\n"
 
        "=== PERFORMANCE ACHIEVED WITH THIS PROMPT ===\n"
        f"Accuracy: {performance_measures['accuracy']}\n"
        f"True Positive Rate (phishing correctly caught): {performance_measures['tpr']}\n"
        f"False Negative Rate (phishing missed): {performance_measures['fnr']}\n"
        f"True Negative Rate (benign correctly passed): {performance_measures['tnr']}\n"
        f"False Positive Rate (benign wrongly flagged): {performance_measures['fpr']}\n"
        f"Raw counts - TP: {performance_measures['tp']}, FP: {performance_measures['fp']}, "
        f"TN: {performance_measures['tn']}, FN: {performance_measures['fn']} "
        f"(out of {performance_measures['n_total']} sites: {performance_measures['n_phish']} phishing, {performance_measures['n_benign']} benign)\n\n"
 
        "=== STUDENT'S REASONING ON MISCLASSIFIED CASES ===\n"
        f"{reasons['errors']}\n\n"
 
        "=== STUDENT'S REASONING ON SOME CORRECTLY-CLASSIFIED CASES (do not break these) ===\n"
        f"{reasons['correct']}\n\n"
 
        "When improving the prompt, follow these principles:\n"
        "1. Reason about PATTERNS across the errors, not individual sites. Look for what the errors have in common "
        "(e.g. many missed phishing sites share 'no login form found', or many false alarms are legitimate login pages).\n"
        "2. Do NOT add narrow rules that memorise specific sites you were shown (e.g. 'if the domain contains X, flag it'). "
        "These help the shown examples but generalise badly. Prefer general, transferable guidance.\n"
        "3. You must also NOT circumvent principle 2. by including lists of examples of legitimate or illegitimate brands as this biases the results towards them.\n"
        "4. Weigh the trade-off using the raw counts. Making the student more suspicious reduces false negatives but "
        "increases false positives, and vice versa. Move only as far as the evidence supports.\n"
        "5. The correctly-classified cases show what the current prompt already gets right. Do not make changes that would flip those into errors.\n"
        "6. Keep the student's required OUTPUT FORMAT (its JSON schema) exactly as it is. Do not change the output contract.\n"
        "7. Prefer minimal, targeted edits over full rewrites. Keep what works.\n\n"
 
        "**Required output format (JSON only):**\n"
        "{\n"
        '  "analysis": string [the main failure pattern(s) you found and the trade-off you are making],\n'
        '  "changes": string [what you changed and why, in general terms],\n'
        '  "new_prompt": string [the full revised student prompt, ready to use]\n'
        "}\n\n"
 
        "**Output Constraints:**\n"
        "Do only output the JSON-formatted output and nothing else.\n"
    )
    return prompt

