"""Contrastive criteria for common vague yes/no judgements (W8, PLAN Task 4.6).

A vague question ("Is the candidate strong in Python?") is read literally by a small model. Criteria make it precise:
"Yes means" and "No means" each describe one side, so the model compares the state with two concrete descriptions.
Use one as a Noul's criteria: {"type": "noul", "instructions": ..., "criteria": LIBRARY["urgent"]["criteria"]}.

These were written once, before the E18 ablation, and were not changed after it. Several judgements (urgency,
refund, human agent, Python skill) match the documented Jev cases, so E18 on those cases is exploratory, not held out.
"""

from __future__ import annotations


def _c(question: str, yes: str, no: str) -> dict:
    return {"question": question, "criteria": {"true": yes, "false": no}}


LIBRARY: dict[str, dict] = {
    # Support and routing
    "urgent": _c("Does this message convey urgency?",
                 "Asks for action now or soon, or describes ongoing harm: money lost, service down, a deadline",
                 "Routine: a question, feedback or request with no time pressure"),
    "wants_human": _c("Is the customer asking for a human agent?",
                      "Asks to speak or write to a person, or rejects automated help",
                      "Asks a question or reports a problem without asking for a person"),
    "refund_request": _c("Is the customer asking for a refund?",
                         "Asks for money back, or for a charge to be reversed or removed",
                         "Asks for information, an exchange, a repair or a delivery, not money back"),
    "repeat_contact": _c("Has the customer contacted support about this before?",
                         "Mentions a prior message, call, ticket or attempt about the same issue",
                         "No sign of any earlier contact about this issue"),
    "complaint": _c("Is this a complaint?",
                    "Expresses dissatisfaction with a product, service or person",
                    "Neutral or positive: a question, a request or thanks"),
    "bug_report": _c("Does this report a software bug?",
                     "Describes software that behaves wrongly: an error, a crash, wrong output",
                     "A question about use, a feature request or feedback, with no fault described"),
    "feature_request": _c("Is this a feature request?",
                          "Asks for new behaviour or a capability that the product does not have",
                          "Reports a fault, asks how to use existing features, or gives feedback"),
    "cancel_intent": _c("Does the customer intend to cancel?",
                        "States a plan or wish to cancel, close the account or stop the subscription",
                        "Complains or asks questions, but does not say they will leave"),
    "security_issue": _c("Does this describe a security issue?",
                         "Unauthorised access, a leaked secret, phishing, or a vulnerability",
                         "An ordinary bug, outage or usability problem with no security impact"),
    # Tone
    "upset": _c("Is the writer upset?",
                "Anger, frustration or distress is clear in the words",
                "Calm or neutral, even if a problem is described"),
    "sarcastic": _c("Is the message sarcastic?",
                    "Says the opposite of what it means, to mock or criticise",
                    "Means what it says"),
    "polite": _c("Is the message polite?",
                 "Uses courteous words and a respectful tone throughout",
                 "Rude, demanding or insulting in any part"),
    # Skills and hiring
    "strong_python": _c("Is the candidate strong in Python?",
                        "Several years of regular professional Python work, or deep knowledge shown by their work",
                        "Little, occasional or only academic Python use, or none"),
    "senior": _c("Is the candidate senior?",
                 "Many years of experience with ownership: leads projects, designs systems, mentors others",
                 "Early career, or works on tasks that others define"),
    "management_experience": _c("Does the candidate have management experience?",
                                "Has managed people: hiring, reviews or direct reports",
                                "Has led work or mentored, but has had no direct reports"),
    # Content
    "actionable": _c("Is this request actionable?",
                     "States what is wanted clearly enough to act on it without more questions",
                     "Too vague or incomplete to act on without asking for more information"),
    "personal_data": _c("Does the text contain personal data?",
                        "Names a person together with contact details, an ID number, an address or health data",
                        "No information that identifies a private person"),
    "question": _c("Does the message ask a question?",
                   "Asks for information or help, with or without a question mark",
                   "Only states facts, feelings or thanks"),
    "positive_review": _c("Is this review positive?",
                          "Recommends the product or is mostly satisfied",
                          "Mixed, mostly unsatisfied, or warns others away"),
    "on_topic": _c("Is the message on topic for customer support?",
                   "About the company's products, orders, accounts or services",
                   "Spam, unrelated chat, or a topic the company does not handle"),
}
