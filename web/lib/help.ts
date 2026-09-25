// Tooltip text for every control. This is a learning tool, so each entry says, in plain words:
// what the thing is, what you will see change, and (last, only when useful) the maths.
// t = title, d = lines, e = effect: "instant" (re-scores in the page), "run" (applies on the next Run),
// "display" (changes no number).
export type Help = { t: string; d: string[]; e?: "instant" | "run" | "display" };

export const HELP = {
  // ---- global
  model: { t: "Model", e: "run", d: [
    "The small open AI model that answers your questions. It runs on this computer, not in the cloud.",
    "0.5B (half a billion parameters) is quick. 1.5B is about twice as slow but gives better answers.",
    "Switching takes 10–60 seconds while the new model loads, and it clears the current answers." ] },
  theme: { t: "Theme", e: "display", d: ["Switch between the dark and the light look."] },

  // ---- toolbar
  presets: { t: "Examples", e: "run", d: [
    "Load a ready-made example: some text plus some questions about it.",
    "It replaces what is in the editor. If you had your own text, an Undo button appears.",
    "Support ticket: one question of each kind. Jev doc cases: examples from Jev's documentation. GDPR: 13 questions about a long article." ] },
  undo: { t: "Undo", d: ["Bring back your own text and questions from before you loaded the example."] },
  evaluation: { t: "Evaluation", e: "run", d: [
    "Three ways to run the same calculation. They give the same answers; only the speed differs. You can check this yourself.",
    "naive: reads your text again for every question. Slow, but simple, so it is the reference.",
    "kv: reads your text once, remembers it, and reuses that memory for each question.",
    "packed: puts your text and all questions into one single pass, with walls so that no question can see another." ] },
  run: { t: "Run (⌘↵)", d: [
    "Ask the model. It reads your text and questions once and gives a probability for every allowed answer.",
    "It does not write any words back. That is the whole idea: read the answer out instead of generating it.",
    "After a run, the calibration dials change the answers instantly." ] },

  // ---- editor
  formJson: { t: "Form or JSON", d: [
    "Form: edit with boxes and buttons.",
    "JSON: see and edit the whole request as text, in the same format that Jev's API uses. Useful to copy or paste a request." ] },
  state: { t: "State (your text)", e: "run", d: [
    "The text that all the questions are about: a customer message, a review, a paragraph from an article.",
    "The model reads it only once, and every question reuses that reading. That is why many questions cost little extra.",
    "You can also paste JSON data here; the model then sees it as structured data." ] },
  addNoul: { t: "Add a yes/no question (Noul)", d: ["For example \"Is the customer angry?\". The answer is a probability from 0 (surely no) to 1 (surely yes)."] },
  addChoice: { t: "Add a multiple-choice question (Choice)", d: ["For example \"Which team should handle this?\" with the options billing / technical / sales. The answer is one option, plus how likely each option is."] },
  addScore: { t: "Add a scale question (Score)", d: ["For example \"How upset is the customer?\" on the scale calm → annoyed → angry. The answer is a position on the scale, like 1.9 (\"almost level 2\")."] },
  qid: { t: "Question name", d: ["A label for you, used as the key in the answer. The model never sees it, so the name cannot change the answer."] },
  qtype: { t: "Question kind", e: "run", d: [
    "Noul = yes/no. Choice = pick one option. Score = a position on an ordered scale.",
    "If you switch, the options carry over: Choice options become Score levels, and the other way round." ] },
  readoutMode: { t: "How the question is asked", e: "run", d: [
    "listwise: the model sees all options at once, labelled A, B, C, and picks a letter. One step, but small models like some letters more than others, so moving an option can change the answer. Try it.",
    "pointwise: the model judges each option on its own (\"Is 'billing' right? yes/no\") and the results are compared. The order cannot matter, but it costs one step per option.",
    "all orders averaged (multiple choice only): the list is asked once per rotation (ABCD, BCDA, …), all in the same pass, and each option's probability is averaged. Every option sits in every position once, so a favourite letter no longer helps one option. Costs one branch per option.",
    "asked: default: use the setting in the Calibration panel (multiple choice: listwise, scale: pointwise)." ] },
  branches: { t: "Branches", e: "display", d: [
    "How many separate mini-questions this adds to the model's pass. A yes/no or listwise question is 1. A pointwise question is one per option or level.",
    "Each branch sees your text and itself only. More branches = a little more work." ] },
  noulMeans: { t: "Yes means / No means", e: "run", d: [
    "Optional. Tell the model exactly what counts as yes and as no.",
    "For example, for \"Is this a refund request?\": Yes means \"asks for money back\". This often fixes vague questions." ] },
  optionName: { t: "Option name", e: "run", d: ["What the answer returns, for example \"billing\". The model sees it as \"A) billing\". Each name must be different."] },
  optionDesc: { t: "Option description", e: "run", d: ["Optional extra words the model sees after the name, for example \"billing: payments, invoices, refunds\". Clear descriptions help small models a lot."] },
  levels: { t: "Scale levels", e: "run", d: [
    "The steps of your scale, from lowest to highest. Level 0 is the first one.",
    "The answer is the expected position: if the model is split 50/50 between level 1 and 2, the score is 1.5." ] },
  moveQ: { t: "Order", e: "display", d: [
    "Moving a question changes nothing: each question is answered on its own.",
    "Moving a scale level does matter, because the order is the scale." ] },

  // ---- calibration
  calPresets: { t: "Fitted or raw", e: "instant", d: [
    "Fitted: temperatures learned on the TRAIN split of our frozen data (500 yes/no questions, 600 news articles), so that \"80% sure\" is right about 80% of the time. The val split chose between methods; the test split only reports.",
    "Raw: the model's own probabilities, no calibration.",
    "The fitted values belong to one model; switching the model loads its own file." ] },
  calProvenance: { t: "Where these numbers come from", e: "display", d: [
    "calibration/<model>.json, written by experiments.py heldout (E14). It records the dataset, the split and its size, and a checksum of the frozen splits file.",
    "docs/DATA.md and the Data page describe the data, and data.py check proves the claims from the files." ] },
  tempNoul: { t: "Temperature · yes/no", e: "instant", d: [
    "Makes the yes/no answers more or less sure of themselves.",
    "Above 1: answers move toward 0.5 (less sure). Below 1: they move toward 0 or 1 (more sure). A yes never turns into a no.",
    "Why: small models are often overconfident. We measured that 0.5B needs about 2.7 to be honest.",
    "The maths: P(yes) = sigmoid(evidence ÷ T + b)." ] },
  biasNoul: { t: "Bias · yes/no", e: "instant", d: [
    "Nudges every yes/no answer toward yes (positive) or toward no (negative).",
    "Unlike the temperature, it can flip close answers from no to yes. It corrects a model that leans one way.",
    "The maths: the b in P(yes) = sigmoid(evidence ÷ T + b), also called Platt scaling." ] },
  tempChoice: { t: "Temperature · multiple choice", e: "instant", d: [
    "Makes multiple-choice answers more or less sure of themselves.",
    "Above 1: the probability spreads out over the options, and the confidence ring shrinks. Below 1: the winner gets even more. The winner never changes." ] },
  tempScore: { t: "Temperature · scale", e: "instant", d: [
    "Makes scale answers more or less sure of themselves.",
    "Above 1: the bars even out, so the score moves toward the middle of the scale and the confidence drops. The tallest bar stays the tallest." ] },
  reset: { t: "Reset", e: "instant", d: ["Put this dial back to the fitted value (or to raw, when calibration is off)."] },
  defaultChoiceMode: { t: "Default for multiple choice", e: "run", d: [
    "How multiple-choice questions are asked when the question itself says \"asked: default\".",
    "listwise: one step, but the option order can change the answer. averaged: every order, averaged; fixes that for one branch per option. pointwise: each option judged alone."] },
  defaultScoreMode: { t: "Default for scales", e: "run", d: ["How scale questions are asked when the question itself says \"asked: default\". Pointwise (each level judged alone) is how Jev does it."] },
  labelMass: { t: "Warn below label mass", e: "display", d: [
    "Label mass = how much of the model's attention went to the allowed answers (Yes/No, A/B/C) rather than to other words.",
    "If it is low, the model \"wanted to say something else\": the question is probably confusing. Then the page shows a warning.",
    "This only controls the warning; it changes no answer." ] },
  envLine: { t: "Save as defaults", e: "display", d: ["Copy this line into poc/minijev.env to make these settings the defaults when the API starts."] },
  calProfiles: { t: "Calibration profiles", e: "instant", d: [
    "Quick presets that scale the fitted temperatures by a factor.",
    "Conservative: multiply by 0.8 (less confident, closer to 0.5). Balanced: use fitted values as-is (multiply by 1.0). Aggressive: multiply by 1.2 (more confident, closer to 0 or 1).",
    "These multiply all temperature dials by the same factor." ] },
  fittedECE: { t: "Fitted calibration quality (ECE)", e: "display", d: [
    "Expected Calibration Error measured on labeled data for this combination of model, primitive, and mode.",
    "Lower is better: 0.05 means the model's confidence is typically within 5% of the true accuracy.",
    "Values are from BoolQ (yes/no) or AG News (multiple choice) held-out splits." ] },
  unfittedWarning: { t: "Unfitted combination", e: "display", d: [
    "No calibration has been fitted for this specific combination of model, primitive type, and mode.",
    "The temperature shown is a generic default. For best calibration, use a fitted combination (see E11 in docs/RESEARCH.md)." ] },
  perModeTemp: { t: "Per-mode temperature override", e: "instant", d: [
    "Set a different temperature for each mode (listwise, pointwise, averaged).",
    "When set, this overrides the general temperature dial for questions using this mode.",
    "Leave blank to use the general temperature setting." ] },
  resetAllCal: { t: "Reset all to fitted defaults", e: "instant", d: [
    "Reset all calibration dials to the fitted defaults for the current model.",
    "This applies the temperatures measured on labeled data (see docs/RESEARCH.md §7.3 R6)." ] },

  // ---- response
  tabs: { t: "Views", e: "display", d: [
    "Answers: charts. JSON: the answer as data, as an app would receive it.",
    "cURL: the command to ask the same thing from a terminal. Raw logits: the model's raw numbers before calibration." ] },
  legend: { t: "Calibrated vs raw", e: "display", d: ["The solid bar is the answer after your calibration dials. The dashed outline is the model's raw answer. The gap shows what the dials changed."] },
  pyes: { t: "P(yes)", d: ["How likely the answer is yes, from 0 to 1. The line in the middle is 0.5: to the right means \"more likely yes\"."] },
  choiceConfidence: { t: "Confidence", d: [
    "How clear-cut the choice is. 0 = all options equally likely (a coin toss). 1 = one option has everything.",
    "The probabilities are the real answer; this is a one-number summary. It uses TypeSafe's own formula." ] },
  scoreConfidence: { t: "Scale confidence", d: [
    "How focused the answer is on one spot of the scale. 1 = all on one level. 0 = as spread out as a random guess.",
    "Being torn between neighbours (annoyed or frustrated?) counts less against it than being torn between the ends (calm or angry?)." ] },
  expected: { t: "Score (expected level)", d: ["The average position on the scale, weighted by the probabilities. 1.92 means \"almost level 2, a little toward 1\"."] },
  usage: { t: "Cost of this request", e: "display", d: [
    "input: how many tokens (word pieces) the model read. Your text is counted once, plus each question.",
    "output: always 0, because the model writes nothing. latency: how long the calculation took on this computer." ] },
  massChip: { t: "Did the questions work?", e: "display", d: ["Near 1.0 means the model put almost all of its attention on the allowed answers, so it understood what kind of answer you wanted."] },
  stale: { t: "Out of date", d: ["You changed the text, the questions, or a setting that needs the model. The answers shown are for the old request. Press Run. (Moving a calibration dial never needs a run.)"] },

  // ---- compare
  cmpMethods: { t: "Methods to compare", e: "run", d: [
    "Different ways to get answers out of the same model. Tick the ones to run.",
    "To make them comparable, every question is asked as multiple choice (yes/no becomes the options yes and no; scale levels become options).",
    "The readout (minijev) always runs, as the reference." ] },
  cmpLogprobs: { t: "Reads out, one question per request", d: [
    "What you can do with an AI API that shows its probabilities: let it write just one token and read how likely each option was.",
    "It is really the same calculation as minijev, just one question at a time." ] },
  cmpGenCached: { t: "Writes each answer", d: [
    "The normal chatbot way: the model writes the answer as words (\"technical\"), and code turns the words back into an option.",
    "One request per question. The text is read once and remembered between requests, as many AI APIs do (prompt caching)." ] },
  cmpJson: { t: "Writes all answers as one JSON", d: ["One message with all questions; the model writes all the answers as JSON in one go. Answers can influence each other, and the JSON can come out broken."] },
  cmpUncached: { t: "Writes each answer, resends the text (slow)", d: ["The simplest way: a separate request per question, each sending the whole text again. Slow, because the text is read again every time."] },
  cmpRun: { t: "Run comparison", d: ["Runs your request once with every ticked method on this computer and times them. Times vary a bit between runs (about ±10–20%)."] },
  cmpRecorded: { t: "Measured earlier", e: "display", d: ["Results we recorded: 13 questions on a 500-token article, each method run 3 times. The thin line shows the fastest and slowest of the 3 runs."] },
  cmpWaterfall: { t: "Where the time goes", e: "display", d: [
    "Starting from the slow way, remove one cost at a time:",
    "1) remember the text instead of reading it again for every question, 2) read the answer out instead of writing it, 3) put all questions in one pass.",
    "Most of the saving comes from step 1." ] },
  agreement: { t: "Agreement", d: ["How often this way picked the same answer as minijev. It is the same model, so a lower number means the way of asking changed its answer."] },

  // ---- same format
  sameFormat: { t: "Same answer format", e: "display", d: [
    "The fairest speed test: same model, same text, same questions, and both sides return the same JSON.",
    "minijev reads the probabilities out and builds the JSON in code. The model writing it has to type every brace, name and digit, and each typed token costs a full pass of the model.",
    "Tokens are word pieces: \"0.97\" is about 3 tokens." ] },
  sameFormatRun: { t: "Run both", d: ["Runs your Playground request both ways, once each, on this laptop. The writing side takes seconds to minutes, depending on how many questions and options you have."] },
  sameFormatDiffer: { t: "When the answers differ", d: [
    "Both sides are the same model, so a difference means the way of asking changed its answer.",
    "Without an answer key, neither side is known to be right. The last column shows how likely minijev found the written answer: a low number means the two really disagree.",
    "Section 2 uses questions with known answers to show which way is right more often." ] },

  structured: { t: "Format enforced (structured output)", d: [
    "What AI APIs call structured output or JSON mode. The program types every fixed part of the JSON itself (braces, names, quotes), and the model may choose only valid content: the digits of a number, or one of your option names.",
    "So the reply is always complete and readable. It is also faster than free writing, because the model does not spend a pass on every brace.",
    "It cannot make the numbers good: those still come from what the model writes." ] },
  unusable: { t: "Unusable answer", d: [
    "The model was asked to write its answers in an exact JSON shape. For this question, the reply cannot be read: a field is missing, a question is nested in the wrong place, a name is not one of the options, or a number is not a probability.",
    "Code cannot guess what a broken reply meant, so this answer is not counted. The raw reply is shown below the table.",
    "minijev cannot fail this way: it writes nothing, and its answer is always one of the allowed options." ] },

  // ---- quality (E11)
  quality: { t: "Quality on questions with known answers", e: "display", d: [
    "The same small model answers the same questions in four ways. Because the right answers are known, we can see which way is more often right, and which way is honest about how sure it is.",
    "Measured once on this laptop and saved; run it again with: uv run python experiments.py quality." ] },
  qTask: { t: "Question set", e: "display", d: [
    "Yes/no · BoolQ: 200 real questions about Wikipedia paragraphs, e.g. \"Is Canada in North America?\".",
    "Multiple choice · AG News: 120 news articles; pick the topic out of World, Sports, Business, Technology." ] },
  qReadout: { t: "Reads out (minijev)", d: ["minijev: one pass, and the probability of every allowed answer is read directly. Nothing is written."] },
  qReadoutCal: { t: "Reads out, calibrated", d: [
    "The same readout, then the temperature dial set on half of the questions and tested on the other half, so it is never tested on what it learned from.",
    "It can only change how sure the answers are, not which answer wins." ] },
  qWritesAnswer: { t: "Writes the answer", d: ["The chatbot way: the model writes \"Yes\" or \"Sports\" as text, and code reads the word back. You get an answer but no probability."] },
  qWritesProb: { t: "Writes a probability", d: ["The model is asked to write how sure it is, e.g. \"0.8\" or a JSON list of probabilities. Small models often write the wrong format, or always the same number."] },
  qAccuracy: { t: "Accuracy", e: "display", d: [
    "How often the answer was right. The orange line is what you get by always guessing the most common answer.",
    "The thin line is the 95% uncertainty of the measurement: if two lines overlap, the difference is not clear." ] },
  qHonesty: { t: "Calibration error (lower = more honest)", e: "display", d: [
    "How far \"how sure it says it is\" is from \"how often it is right\". 0 is perfectly honest: \"80% sure\" is right 80% of the time.",
    "Writing the answer gives no probability, so it has no score here." ] },
  qTime: { t: "Time per question", e: "display", d: ["Average seconds per question on this laptop's CPU, for this way of asking. Writing costs time for every word the model writes."] },
  qWords: { t: "Tokens written", e: "display", d: ["How many tokens (word pieces) the model wrote per question, and how many replies could not be read back into an answer."] },
  qScatter: { t: "Speed vs accuracy", e: "display", d: ["Each dot is one way of asking, for one model. Up = more often right. Left = faster. The best place is the top left."] },

  // ---- order flaw (E13)
  orderFlaw: { t: "The order flaw", e: "display", d: [
    "When a multiple-choice question lists its options as A, B, C, D, a small model partly answers by letter, not by content: it likes some letters more than others.",
    "So the same question, with the same options in another order, can get a different answer. This section measures how often that happens, and tests three fixes." ] },
  orderLetters: { t: "Favourite letters", e: "display", d: [
    "Orange: how often the model picked each letter. Grey: how often the right answer was actually at that letter (the orders are shuffled, so about a quarter each).",
    "If orange is taller than grey for a letter, the model picks that letter more often than the content justifies." ] },
  orderFlips: { t: "Answer flips", e: "display", d: ["The share of articles where the answer changed when only the option order changed. 0 means the order never matters."] },
  orderCost: { t: "Cost", e: "display", d: [
    "How many branches (mini-questions) one multiple-choice question needs. The text is still read only once, so extra branches cost only the question's own tokens.",
    "Debiased needs 1 branch at run time, but it needs a one-time measurement of the model's letter liking first." ] },

  // ---- data
  dataClaims: { t: "Verified claims", e: "display", d: [
    "Each line is one claim of the data card, checked against the files when this page loaded: checksums, split sizes, no overlap between splits, label balance.",
    "The same checks run in a terminal with: cd poc && uv run python data.py check." ] },
  dataHeldout: { t: "Held-out results", e: "display", d: [
    "The only numbers to use when you defend a claim. The temperatures were fitted on the train split, the method was chosen on the val split, and the test split was used once, to report.",
    "The lines around each number are 95% uncertainty intervals. Grey rows are the Choice modes that val did not choose." ] },

  // ---- hood
  hoodReadout: { t: "What is a readout?", e: "display", d: [
    "Instead of asking the model to generate an answer token by token (\"b\", \"i\", \"l\", \"l\", \"i\", \"n\", \"g\"), minijev reads the probabilities of the allowed answer tokens (A, B, C) from one forward pass.",
    "Every language model computes probabilities for all ~150,000 tokens at every position. Generation samples from those probabilities and loops. A readout uses them directly and stops.",
    "The answer IS the probabilities: the option with the highest probability is the predicted choice, and you get calibrated confidence for all options." ] },
  hoodBranch: { t: "Branches", e: "display", d: ["Each box is one mini-question the model answers. Click one to follow it through the picture, the grid, and the token list."] },
  hoodMask: { t: "Who can see what", e: "display", d: [
    "Each row is a token, each column is a token it is allowed to look at. Blue = allowed.",
    "Every question can see all of your text, but never another question. That is why asking questions together or one by one gives the same answers." ] },
  hoodPositions: { t: "Positions", e: "display", d: ["Every question is numbered as if it came right after your text, as if it were the only question. The model cannot tell that other questions exist."] },
  hoodTemplate: { t: "Template tokens", e: "display", d: ["The fixed wrapper around your text (a system instruction and the word \"STATE:\"). The model reads it too, and it is the same for every request."] },
  hoodRun: { t: "Run for label mass", d: ["Asks the model once so that each branch can show how well it worked (its label mass). The picture itself only needs the text to be split into tokens."] },

  // ---- findings
  fSize: { t: "Model size", e: "display", d: ["Show the results for the small (0.5B) or the larger (1.5B) model."] },
  fView: { t: "Before or after calibration", e: "display", d: ["Raw: the model's own confidence. Temperature: after calibration. Dots on the dashed line mean \"as sure as it is right\"."] },
  fReliability: { t: "Is the model honest about how sure it is?", e: "display", d: [
    "Each dot groups answers by how sure the model said it was (left to right), and shows how often they were actually right (bottom to top).",
    "Dots below the dashed line: the model was overconfident. Bigger dot = more answers." ] },
  fEce: { t: "Calibration error (ECE)", e: "display", d: [
    "The average gap between how sure the model says it is and how often it is right. 0 is perfect.",
    "The line around each dot is the uncertainty of the measurement. If two lines overlap, the difference is not clear." ] },
} satisfies Record<string, Help>;

export type HelpKey = keyof typeof HELP;
