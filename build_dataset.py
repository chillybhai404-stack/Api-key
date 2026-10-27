"""Build the included synthetic training dataset for AXU Interaction Intelligence v2.

This creates a larger starter corpus from reviewed phrase patterns. It is synthetic,
not harvested from real users, and is intended for learning/prototyping.
"""
from __future__ import annotations
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "interactions.csv"
RNG = random.Random(20261024)

# Each string is a single intent example. Slot templates below provide controlled
# variation for intents whose wording naturally contains entities or symptoms.
SEEDS = {
"greeting": [
"Hi", "Hello", "Hey there", "Good morning", "Good afternoon", "Good evening",
"Yo, what's up?", "Hey, are you there?", "Hello assistant", "Hi, can you help me?",
"Hey AXU", "Is anyone there?", "Greetings", "Hi there", "Morning!", "Hey, how are you?",
"Hello again", "I'm back", "What's up?", "Are you online?", "Hey, let's start",
"Hi, I have a question", "Hello, I need a hand", "Good day", "Hey friend"
],
"farewell": [
"Bye", "Goodbye", "See you later", "Talk to you soon", "I'm leaving now", "Catch you later",
"That's all for now", "I'm done, thanks", "Have a good day", "Good night", "See you tomorrow",
"I'll be back later", "Signing off", "End this conversation", "Bye for now", "I have to go",
"Thanks, I'm heading out", "Talk later", "See ya", "Close the chat for now"
],
"gratitude": [
"Thanks", "Thank you so much", "Thanks for the help", "That was useful, thank you",
"I appreciate it", "You're a lifesaver", "Much appreciated", "Thanks a ton",
"This solved my problem, thanks", "Really helpful answer", "Thanks for explaining",
"I owe you one", "Perfect, thank you", "Cheers for that", "Thanks for being patient",
"That helps a lot", "Thank you for your time", "Awesome, thanks", "Grateful for the help",
"Thanks for sorting this out"
],
"apology": [
"Sorry", "I'm sorry about that", "My mistake", "I apologize", "Sorry for the confusion",
"That was my fault", "Please forgive the mix-up", "I didn't mean to do that",
"Sorry, I misunderstood", "Apologies for the delay", "I should have explained better",
"Oops, sorry", "I made an error", "Sorry for asking again", "My bad, I got that wrong",
"I apologize for the inconvenience", "Sorry about my earlier message", "Let me correct that, sorry"
],
"small_talk": [
"How are you doing?", "How's your day going?", "What's new?", "I'm a bit bored",
"Tell me something interesting", "It's a quiet day today", "I'm just checking in",
"What are you up to?", "Let's chat for a bit", "I had a long day", "Today has been strange",
"I don't know what to do right now", "Say something fun", "Can we talk?", "How's everything?",
"I'm in a good mood", "I'm feeling curious today", "Let's have a casual conversation"
],
"joke_request": [
"Tell me a joke", "Make me laugh", "Give me a funny joke", "Do you know any clean jokes?",
"I need a laugh", "Say something hilarious", "Tell me a silly pun", "Give me a one-line joke",
"Make a joke about coding", "Got a dad joke?", "Tell me a school joke", "Can you be funny?",
"Give me a harmless roast of my messy code", "I want a funny riddle", "Share a quick pun"
],
"identity_question": [
"Who are you?", "What should I call you?", "Are you a bot?", "Are you an AI assistant?",
"Who made you?", "What is your name?", "Are you human?", "Tell me about yourself",
"What model am I talking to?", "Are you AXU's model?", "Who built this assistant?",
"Introduce yourself", "What's your identity?", "What are you exactly?"
],
"capability_question": [
"What can you do?", "How can you help me?", "What features do you support?",
"What kinds of tasks can you handle?", "Can you classify user messages?", "Can you help with Python?",
"Do you understand multiple intents?", "What are your limits?", "Can you answer questions?",
"What can I use this model for?", "Which tasks are supported?", "Can you work with text?",
"Can you analyze feedback?", "What does this API return?"
],
"question": [
"Can you answer a question for me?", "I have a question", "Could you explain this?",
"Why does this happen?", "What does this mean?", "How does it work?", "Can I ask you something?",
"Is there a reason for this?", "When should I use it?", "Which option is better?",
"Where can I find the setting?", "What is the difference between these two?",
"Does this support offline use?", "Is that possible?", "Would this work on Android?",
"Can you clarify the rule?", "What caused the issue?", "How much does it cost?"
],
"information_request": [
"Give me more information about this feature", "I need the details", "Explain the available options",
"Show me the specifications", "Tell me more about the service", "I want to know the requirements",
"What are the important details?", "Provide an overview", "Can I get the full information?",
"List the main points", "I need the facts before deciding", "Describe how the system works",
"Give me a quick summary of the options", "Share the relevant details"
],
"how_to": [
"How do I get started?", "Show me how to do this", "Give me step by step instructions",
"Can you walk me through it?", "What steps should I follow?", "Teach me how to set it up",
"Explain the process in simple steps", "How can I do this correctly?", "I need a tutorial",
"Guide me through the setup", "What is the first step?", "Please demonstrate the workflow",
"How can a beginner do this?", "Give me instructions I can follow",
"How do I change the picture on my profile?", "Show me the steps to update my profile photo",
"How can I change my account settings?", "Walk me through updating my avatar",
"How do I update my account details?", "Give me steps for changing my profile image"
],
"coding_help": [
"Help me debug this Python code", "Explain this JavaScript error", "Write a Python function for me",
"My HTML page is not rendering correctly", "How do I call this API from JavaScript?",
"Explain this SQL query", "Help me build a small REST API", "Why is my loop not working?",
"Can you review this function?", "Show an example of a Python class",
"Help me understand async code", "I need help with CSS layout", "How do I parse JSON in Python?",
"Fix this syntax issue", "Explain what this stack trace means", "Help me structure my project",
"How do I save a model with joblib?", "I need a database connection example"
],
"writing_help": [
"Help me rewrite this paragraph", "Proofread my description", "Make this message more professional",
"Improve the wording without changing the meaning", "Draft a short project introduction",
"Help me write a polite reply", "Make this clearer and easier to read",
"Can you edit my announcement?", "Suggest a better title", "Fix the grammar in this text",
"Turn these notes into a paragraph", "Help me write a project proposal",
"Make this less repetitive", "Write a concise product description"
],
"translation_request": [
"Translate this into English", "Translate this message to Urdu", "How do I say this in Hindi?",
"Convert this paragraph to Kashmiri", "Translate the following sentence into Arabic",
"Can you translate this into simple English?", "Keep the meaning but change the language",
"What is the English translation of this phrase?", "Translate this caption to Hindi",
"I need a translation into Urdu", "Translate my response into French"
],
"summarization_request": [
"Summarize this article", "Give me the key points only", "Make a short summary of these notes",
"Explain this in a few sentences", "Condense this paragraph", "Summarize the main argument",
"Can you shorten this without losing important information?", "Give me a quick recap",
"Turn this long text into bullet points", "What are the main takeaways?",
"Summarize this chapter for revision", "Make this easier to review",
"Turn these notes into a concise recap", "Make a short summary from the text", "Condense this into a brief overview",
"Summarize my notes without rewriting every detail", "Give me a concise recap of the main points",
"Can you summarize the text in three sentences?", "Create a short summary of this document",
"Please shorten this report into its key ideas", "Give me the gist of this chapter",
"I need a summary, not a rewritten version", "Extract the main points from the passage",
"Write a compact recap of these meeting notes", "Reduce this long text to a short summary"
],
"math_help": [
"Help me solve this equation", "Explain how to calculate the percentage", "Check my algebra steps",
"Solve this quadratic equation", "I don't understand this geometry problem",
"Show the steps for this calculation", "Can you check my maths answer?",
"Explain the formula in simple words", "Help me with linear equations", "Why is my answer wrong?",
"Teach me how to calculate the mean", "Give me a hint for this problem",
"Explain this trigonometry question", "Show another example of this math concept"
],
"creative_request": [
"Give me a fresh app idea", "Brainstorm names for my project", "Write a short science fiction scene",
"Suggest a unique website concept", "Create a list of design directions",
"Help me develop a character", "Think of unusual features for this app",
"Give me ideas for a motion design intro", "Come up with a project name",
"Generate concepts for a creative campaign", "Help me make this idea more original",
"Suggest a visual theme for my website"
],
"learning_help": [
"Teach me the basics of machine learning", "Explain this topic for a beginner",
"Make a study plan for this chapter", "Quiz me on this subject", "Give me practice questions",
"Help me remember these concepts", "Explain the difference between training and testing",
"Teach me Python from the beginning", "Give me a simple learning roadmap",
"Check whether I understood the lesson", "Explain it using a simple example",
"Make a revision checklist", "What should I learn before APIs?"
],
"command": [
"Open the settings page", "Show the dashboard", "Sort the list by date", "Save this item",
"Delete the selected draft", "Export the current report", "Switch to dark mode",
"Turn notifications off", "Create a new project", "Rename this workspace",
"Refresh the page", "Clear the current search", "Pin this item", "Archive the old record",
"Move this file to the project folder", "Log me out", "Go back to the previous screen",
"Add this to my favourites", "Generate the report now", "Start a new session"
],
"feature_request": [
"Please add a dark mode", "I would like a search filter", "Could you add offline support?",
"Please support exporting to CSV", "It would be useful to have keyboard shortcuts",
"Can you add a dashboard widget?", "I want custom themes", "Please include bulk editing",
"Could you add a way to duplicate projects?", "A calendar view would be helpful",
"Please add an undo button", "Can we have an import feature?", "Add a way to pin important items",
"I wish there were better sorting options", "Can you add multi-language support?",
"It needs a better mobile layout", "Please allow custom categories", "A history panel would be useful",
"I want an option to back up my data", "Could the app have saved filters?",
"I need the app to work without an internet connection", "Please let me use this when I'm offline",
"Offline mode would be useful", "Can the app keep working when there is no network?",
"Add support for using the app without Wi-Fi", "I want to open my saved work offline",
"It should not require internet for basic features", "Please add an offline-first mode"
],
"positive_feedback": [
"I love this app", "This is really useful", "Great job on the new update", "The design looks clean",
"Everything feels much easier now", "This feature is excellent", "I'm happy with this experience",
"The app works perfectly for me", "This is exactly what I needed", "Nice work on the improvements",
"I like how simple it is", "The latest version is much better", "That feature saves me time",
"The support was excellent", "I'm impressed with the result", "This workflow is smooth"
],
"negative_feedback": [
"I don't like this update", "The new design is confusing", "This feature isn't useful to me",
"The app is harder to use now", "I'm disappointed with this change", "The interface feels cluttered",
"This isn't what I expected", "The experience has become worse", "The new layout is frustrating",
"I preferred the previous version", "The result is disappointing", "This feature misses the point",
"The update made my workflow slower", "I'm not happy with this", "This needs a lot of improvement"
],
"complaint": [
"I've been waiting for a response for days", "The service has been unreliable lately",
"I keep getting charged unexpectedly", "This process is unnecessarily difficult",
"I have reported this issue twice already", "The instructions are unclear",
"The support team hasn't replied", "This keeps wasting my time", "I had a poor experience",
"The same issue keeps happening", "I'm frustrated with the service", "Nothing has improved after my report",
"The app keeps interrupting my work", "This is not acceptable service", "I want this problem addressed"
],
"bug_report": [
"The app crashes when I sign in", "The save button does nothing", "I found a bug in the latest version",
"The screen goes blank after tapping submit", "The page throws an error when I upload a file",
"My changes disappear after I reopen the app", "The app freezes on the loading screen",
"The search results are incorrect", "The dropdown is broken on mobile",
"The app closes unexpectedly", "The login flow crashes even though my password is correct",
"I get an error every time I press save", "The form refuses to submit valid data",
"The back button sends me to the wrong screen", "The app doesn't remember my settings",
"The notification opens the wrong page", "The exported file is corrupted", "The counter shows the wrong value",
"The image preview fails to load", "My account page shows a technical error"
],
"account_login": [
"I can't log into my account", "My sign-in keeps failing", "The login page won't accept my details",
"I'm locked out of my account", "I can't access my profile", "It says my account doesn't exist",
"My login session expires immediately", "I get stuck on the sign-in screen",
"The app won't let me sign in with email", "I cannot access my account after the update",
"My account login is not working", "I keep getting redirected back to login",
"Sign-in works on the website but not on my phone", "I need help accessing my account"
],
"password_reset": [
"I forgot my password", "Send me a password reset link", "How can I change my password?",
"The reset email never arrived", "My password reset link has expired", "I can't create a new password",
"The reset page says the token is invalid", "Help me recover my account password",
"I need to update my login password", "The password recovery process isn't working",
"I don't remember the password I used", "Where can I reset my password?"
],
"billing_payment": [
"My payment failed", "I was charged twice", "The invoice amount is incorrect", "Where can I download my receipt?",
"My card was declined", "The subscription payment didn't go through", "Why was I billed today?",
"I need a copy of my invoice", "The checkout page won't accept my payment",
"My payment is still pending", "The amount due looks wrong", "I paid but the account still shows unpaid",
"Can I change my billing method?", "My renewal payment failed",
"I received two receipts for a single purchase", "I got duplicate receipts for the same payment",
"You sent me two receipts for one order", "My payment has two separate receipts",
"The receipt was sent twice", "I have a duplicate billing receipt",
"I was charged twice for one purchase", "The invoice lists the wrong amount",
"I received a receipt but did not make a payment", "My transaction history has duplicate entries"
],
"refund_request": [
"I want a refund", "Please refund my last payment", "How do I request my money back?",
"I'd like to return this purchase and get a refund", "I was charged for something I didn't use",
"Can you reverse this payment?", "I need a refund for the duplicate charge",
"Please explain the refund process", "My refund hasn't arrived", "I want to dispute this charge",
"I cancelled and still need my refund", "Can I get my payment returned?"
],
"cancellation": [
"Cancel my subscription", "I want to close my account", "How do I stop automatic renewal?",
"Please cancel my plan", "I don't want to continue the service", "Turn off my subscription renewal",
"I need to cancel my booking", "Remove me from this paid plan", "Stop my membership from renewing",
"I want to delete my account permanently", "Please end my trial before it charges me",
"How can I cancel this service?"
],
"order_status": [
"Where is my order?", "Track my purchase", "Has my order shipped yet?", "What is the status of my order?",
"My order confirmation hasn't arrived", "Can I see the progress of my order?",
"Did you receive my order?", "My order status hasn't changed", "I need an update on order 1234",
"Is my order still being processed?", "When will the order be dispatched?"
],
"delivery_shipping": [
"My package is late", "The delivery hasn't arrived", "The courier couldn't find my address",
"Can I change the shipping address?", "My parcel arrived damaged", "Delivery was marked complete but I got nothing",
"I need to reschedule the delivery", "Shipping is taking longer than expected",
"Can you tell me the delivery date?", "The package was sent to the wrong address",
"I want to know the shipping cost", "The tracking number isn't updating"
],
"product_recommendation": [
"Recommend a good budget phone", "Which laptop would suit a student?", "Help me choose between these two products",
"Suggest a beginner-friendly coding app", "What's a good option for video editing?",
"Recommend a simple note-taking tool", "Which product offers better value?",
"I need a recommendation for a small project", "Suggest something reliable under my budget",
"What should I buy for learning Python?", "Help me compare these options"
],
"booking_scheduling": [
"Book an appointment for Friday", "Schedule a meeting for tomorrow", "Move my appointment to 3 PM",
"Are there any available slots next week?", "Help me arrange a call", "Can I reschedule my booking?",
"Set up a meeting with the team", "I need to reserve a time slot", "Cancel the meeting on Monday",
"Find a free time for a short call", "Schedule this for next month"
],
"reminder_request": [
"Remind me to submit the report", "Set a reminder for 6 PM", "Don't let me forget my appointment",
"Remind me tomorrow morning to call home", "Create a weekly reminder to study",
"I need a reminder before the deadline", "Notify me when it's time to leave",
"Remind me every Monday to back up my files", "Can you remind me in two hours?",
"Set a reminder for my project review", "Help me remember to renew this"
],
"search_request": [
"Search for Python tutorials", "Find all files named report", "Look up the latest project entry",
"Search the notes for the word budget", "Find a record from last week", "Show me matching results",
"Search for projects tagged design", "Find the message about the API", "Look for similar items",
"Find the settings page", "Search this database for my records"
],
"navigation_help": [
"Where is the settings menu?", "How do I get back to the home screen?", "I can't find the profile page",
"Where can I see my saved projects?", "How do I open the dashboard?", "Which menu contains exports?",
"I got lost in the app", "How do I reach the account page?", "Where are archived records?",
"Show me where to manage notifications", "How can I get to the help section?"
],
"confirmation": [
"Yes", "Yes, that's correct", "Exactly", "Confirmed", "That is what I meant", "Go ahead",
"Proceed with that", "Sounds good", "That's right", "I agree with this option", "Okay, do it",
"Correct", "Yes please", "That works for me", "Approved", "Let's do that"
],
"rejection": [
"No", "No thanks", "That is not correct", "Don't do that", "Cancel that action", "I disagree",
"That's not what I meant", "Please stop", "Not this option", "I don't want that",
"No, try something else", "Reject this change", "Leave it as it is", "That won't work for me",
"No, that's wrong", "I refuse this suggestion"
],
"clarification": [
"What do you mean by that?", "Can you explain that part again?", "I meant the other option",
"Let me clarify what I asked", "To be clear, I want the API only", "That's not quite what I meant",
"Can you be more specific?", "I should explain the context", "Here's what I meant instead",
"Could you clarify the last step?", "Let me rephrase the question", "I meant on Android, not iOS"
],
"confusion": [
"I'm confused", "I don't understand this", "This makes no sense to me", "I'm lost here",
"I can't follow these instructions", "I don't get why that happened", "This is confusing",
"I have no idea what this means", "I'm not sure what to do next", "I can't understand the output",
"Why is this so complicated?", "I'm stuck and don't know why", "The explanation went over my head"
],
"urgency": [
"This is urgent", "I need a solution right now", "Please handle this as soon as possible",
"The deadline is in ten minutes", "I need this fixed today", "This is time sensitive",
"Please prioritize this", "I can't wait until next week", "I'm blocked until this is resolved",
"This needs immediate attention", "Can someone respond quickly?", "I have a critical deadline"
],
"privacy_concern": [
"What data do you collect about me?", "Can I delete my personal data?", "Who can see my information?",
"I don't want my data shared", "How long do you keep my messages?", "Can I use this without tracking?",
"Does the app sell user data?", "I want to download my personal data", "How is my data used?",
"Can I disable analytics?", "I am concerned about my privacy", "Please explain the data policy"
],
"security_concern": [
"I think someone accessed my account", "Is my data encrypted?", "I received a suspicious login alert",
"How do I enable two-factor authentication?", "I found a security vulnerability", "My account may be compromised",
"Someone changed my password without permission", "How can I secure my account?",
"I got a suspicious email claiming to be from you", "Can I review active sessions?",
"I think this link may be a phishing attempt", "Help me report a security issue"
],
"accessibility_request": [
"Please add screen reader support", "The text is too small to read", "Can you improve keyboard navigation?",
"I need higher contrast", "Add captions to the videos", "Can the interface support larger text?",
"I need an option to reduce motion", "The buttons are difficult to reach", "Please improve accessibility",
"Can you add voice control?", "I need clearer focus indicators", "Support users who navigate by keyboard",
"Can I use every part of the app with a keyboard?", "The controls are hard to use without a mouse",
"Make the buttons easier to tap on a phone", "The tap targets are too small",
"Please make the interface accessible for keyboard-only users", "I need larger touch targets",
"The screen reader cannot identify the buttons", "Can the app be easier to use with limited dexterity?",
"Please improve focus order when tabbing", "The controls are difficult to tap accurately"
],
"language_request": [
"Can I change the app language?", "Add Urdu language support", "I want the interface in Hindi",
"Can you reply in simple English?", "Please switch to Kashmiri", "How do I set my preferred language?",
"The translation is inaccurate", "Can I choose another language?", "Use English for the next answer",
"Please respond in Urdu", "I need support for multiple languages"
],
"troubleshooting": [
"Help me troubleshoot this problem", "What should I try first?", "Walk me through diagnosing the issue",
"How can I fix this error?", "Give me steps to get this working again",
"I need help finding the cause", "Let's debug this step by step", "What checks should I run?",
"Can you help me narrow down the problem?", "Suggest a safe way to test the issue",
"How do I restore the app to a working state?", "Help me diagnose why this isn't working"
],
"data_export": [
"Export my data to CSV", "Download all my records", "Can I export this report as PDF?",
"Create a backup of my projects", "I need my data in JSON format", "Where is the export button?",
"Save a copy of my account data", "Can I download my history?", "Export the current table",
"I want to back up everything before updating", "Give me a downloadable copy of these records"
],
"performance_issue": [
"The app is extremely slow", "The page takes forever to load", "Scrolling is laggy on my phone",
"The app uses too much memory", "The dashboard freezes when I open it", "The website is slow on mobile",
"It takes a long time to save changes", "The animations are stuttering", "The app drains my battery",
"Search is taking too long", "The page is unresponsive", "The app performance got worse after the update"
],
"weather_request": [
"What's the weather today?", "Will it rain tomorrow?", "Give me the weather forecast for this weekend",
"Is it going to be cold tonight?", "What's the temperature outside?", "Do I need an umbrella today?",
"Check the forecast for Srinagar", "Will the weather be sunny tomorrow?"
],
"opinion_request": [
"What do you think of this idea?", "Give me your assessment of this design",
"Which version seems clearer?", "What are the strengths and weaknesses?", "Review this concept honestly",
"Does this approach make sense?", "Tell me what could be improved", "Which option would you recommend and why?"
],
"comparison_request": [
"Compare these two options", "What are the differences between A and B?", "Make a pros and cons comparison",
"Which is better for my situation?", "Compare the price and features", "Show the tradeoffs between these approaches",
"How do these products differ?", "Can you compare the two designs side by side?"
],
"other": [
"The blue box in the second screen looks odd", "Here is a random thought", "I'm testing the classifier",
"This sentence doesn't fit a clear category", "A quick note for later", "Just sending a sample message",
"The meeting room has four chairs", "I wrote this to test the model", "This text is only a placeholder",
"The sky looks different today", "Sample input number seven", "This message may be unrelated to the app"
],
}

# Structured slots deliberately widen wording while keeping the same intended label.
SLOT_TEMPLATES = {
"bug_report": {
 "prefixes": ["", "I noticed that ", "There is a bug: ", "I'm reporting that ", "Something is wrong: "],
 "problems": ["the app crashes", "the app freezes", "the button does nothing", "the screen goes blank", "the form shows an error", "the page won't load", "my changes disappear", "the search returns wrong results", "the image won't open", "the app closes unexpectedly", "the settings reset themselves", "the upload fails", "the app gets stuck", "the link opens the wrong page", "the counter shows the wrong number", "the export file is broken"],
 "ends": ["", " after I tap save", " when I sign in", " on my Android phone", " after the latest update", " whenever I submit the form", " even with valid information", " every time I open the page"]},
"feature_request": {
 "prefixes": ["Please add ", "Could you add ", "I'd love to have ", "It would help if there were ", "Can we get ", "I suggest adding "],
 "features": ["dark mode", "offline access", "a search bar", "advanced filters", "CSV export", "PDF export", "custom themes", "bulk editing", "an undo button", "a calendar view", "saved searches", "keyboard shortcuts", "a backup option", "multi-language support", "a dashboard widget", "version history", "a duplicate action", "custom categories", "a pin feature", "a better mobile layout", "screen reader support", "reduced-motion settings", "an import tool", "a history panel", "a data recovery option", "a compact view", "multi-select", "local-only mode", "a quick-add button", "a command palette"],
 "ends": ["", " in the next update", " for mobile users", " in the settings", " to make the workflow faster", " so it is easier to use", " if possible"]},
"account_login": {
 "prefixes": ["I can't ", "I'm unable to ", "I keep failing to ", "The app won't let me ", "Please help me ", "I am having trouble trying to "],
 "actions": ["log in", "sign in", "access my account", "open my profile", "authenticate with email", "stay signed in", "access my account on mobile", "sign in after the update"],
 "ends": ["", ". It keeps returning me to the login screen.", " even though my details are correct.", " and I need help.", "; the screen just reloads."]},
"password_reset": {
 "prefixes": ["I need to ", "Can you help me ", "How do I ", "Please help me ", "I am trying to "],
 "actions": ["reset my password", "recover my password", "change my login password", "get a password reset email", "create a new password", "recover access to my account"],
 "ends": ["", " because I forgot the old one.", ". The reset link does not work.", ", but the email never arrives.", " on my phone."]},
"billing_payment": {
 "prefixes": ["My ", "I have a problem with my ", "There is an issue with the ", "Please check my "],
 "items": ["payment", "invoice", "subscription charge", "card payment", "renewal payment", "checkout", "billing amount", "receipt", "pending transaction", "duplicate charge"],
 "ends": [" failed.", " is incorrect.", " is still pending.", " was charged twice.", " does not appear in my account.", " needs to be checked.", " went through but my account is unpaid."]},
"refund_request": {
 "prefixes": ["I want a refund for ", "Please return the money for ", "Can I get a refund for ", "I'd like my payment back for ", "I need to request a refund for "],
 "items": ["my last payment", "a duplicate charge", "an accidental purchase", "a cancelled order", "a subscription renewal", "a product that never arrived", "an incorrect invoice", "a service I couldn't use", "the wrong item", "my recent transaction"],
 "ends": [".", ". Please tell me the next steps.", ". The charge was unexpected.", ". I have the receipt."]},
"cancellation": {
 "prefixes": ["Please cancel ", "I want to cancel ", "How can I cancel ", "Stop renewing ", "I'd like to end ", "Help me cancel "],
 "items": ["my subscription", "my paid plan", "automatic renewal", "my membership", "my booking", "the free trial", "my order", "this service", "the upcoming appointment", "my account"],
 "ends": [".", " before the next charge.", " as soon as possible.", " and confirm when it is done."]},
"order_status": {
 "prefixes": ["Can you check ", "I want to track ", "Please show me ", "What's the status of ", "I need an update about "],
 "items": ["my order", "my recent purchase", "order number 1042", "the shipment", "the item I bought", "my latest order", "the package I paid for"],
 "ends": ["?", "; it has not updated.", " because it is taking longer than expected.", " before I contact support."]},
"delivery_shipping": {
 "prefixes": ["My delivery ", "The package ", "The courier ", "Shipping for my order ", "The tracking page "],
 "problems": ["is late", "has not arrived", "went to the wrong address", "is marked delivered but is missing", "arrived damaged", "has no tracking updates", "needs a new delivery time", "has an incorrect address", "is taking too long", "was returned unexpectedly"],
 "ends": [".", ". What should I do?", ". Please help me resolve it.", " and I need an update."]},
"reminder_request": {
 "prefixes": ["Remind me to ", "Set a reminder to ", "Please remind me when it's time to ", "Don't let me forget to ", "Create a reminder for me to "],
 "tasks": ["submit my report", "study maths", "call my family", "back up my files", "review the project", "pay the bill", "attend my appointment", "send the email", "check the deadline", "renew my subscription", "drink some water", "finish the assignment"],
 "times": ["", " tomorrow", " at 6 PM", " every Monday", " in two hours", " next week", " before the deadline"]},
"search_request": {
 "prefixes": ["Search for ", "Find ", "Look up ", "Locate ", "Can you search for "],
 "items": ["Python tutorials", "my saved projects", "the API documentation", "records from last week", "the word budget in my notes", "files named report", "messages about login", "items tagged design", "the settings page", "similar entries", "the latest project update", "my archived notes"],
 "ends": [".", " in my workspace.", " and show the matching results.", " please."]},
"data_export": {
 "prefixes": ["Export ", "Download ", "Create a backup of ", "Give me a copy of ", "Save a downloadable version of "],
 "items": ["my data", "all project records", "this report", "my account history", "the current table", "my notes", "the full dataset", "the selected records", "my project list"],
 "ends": [".", " as CSV.", " as JSON.", " as a PDF.", " before I update the app."]},
"performance_issue": {
 "prefixes": ["The app is ", "My page is ", "The website feels ", "Loading has become ", "The dashboard is ", "Scrolling is "],
 "problems": ["very slow", "laggy on mobile", "taking too long to load", "freezing under load", "using too much memory", "stuttering during animations", "draining my battery", "unresponsive after a few minutes", "slower after the update", "delayed when saving changes"],
 "ends": [".", " on my Android phone.", " compared with the previous version.", " even on a stable connection."]},
"accessibility_request": {
 "prefixes": ["Please support ", "Could you improve ", "I need ", "Please add ", "It would help to have "],
 "features": ["screen readers", "larger text", "high contrast", "keyboard navigation", "reduced motion", "captions", "voice control", "clear focus indicators", "better touch targets", "accessible form labels"],
 "ends": [".", " for the mobile app.", " so the app is easier to use.", " in the next update."]},
"language_request": {
 "prefixes": ["Please switch the interface to ", "Can you support ", "I want to use ", "Please reply in ", "Add support for "],
 "languages": ["English", "Urdu", "Hindi", "Kashmiri", "Arabic", "French", "Spanish", "simple English", "multiple languages"],
 "ends": [".", " throughout the app.", " for my account.", " if possible."]},
"booking_scheduling": {
 "prefixes": ["Schedule ", "Book ", "Move ", "Reschedule ", "Arrange "],
 "items": ["a meeting", "an appointment", "a call with the team", "a review session", "a time slot", "a consultation", "a project discussion"],
 "times": [" for tomorrow", " for Friday", " at 3 PM", " next week", " for Monday morning", " in the afternoon", " for next month"],
 "ends": [".", " if there is availability.", " and send me the details."]},
"product_recommendation": {
 "prefixes": ["Recommend ", "Suggest ", "Help me choose ", "Find ", "I'd like a recommendation for "],
 "items": ["a budget phone", "a laptop for coding", "a beginner Python course", "a video editing app", "a simple database tool", "a note-taking app", "a free design tool", "a reliable power bank", "a lightweight code editor", "a project management app"],
 "ends": [".", " for a student.", " under a limited budget.", " and explain why."]},
"weather_request": {
 "prefixes": ["What's the weather like ", "Check the forecast ", "Will it rain ", "Tell me the temperature ", "Do I need a jacket "],
 "times": ["today", "tomorrow", "this evening", "this weekend", "next Monday", "tonight", "in Srinagar", "in my area"],
 "ends": ["?", " and mention the chance of rain.", " please."]},
}

MULTI = [
("I love this app, but the login keeps crashing", ["positive_feedback", "bug_report", "complaint", "account_login"]),
("The design is great, but please add offline mode", ["positive_feedback", "feature_request"]),
("Thanks, but my payment failed again", ["gratitude", "billing_payment", "complaint"]),
("I can't log in and the password reset email never arrives", ["account_login", "password_reset", "bug_report"]),
("The app is slow and the save button freezes", ["performance_issue", "bug_report", "complaint"]),
("Please cancel my subscription because I was charged twice", ["cancellation", "billing_payment", "complaint"]),
("Can you export my data and explain how you store it?", ["data_export", "privacy_concern", "question"]),
("I like the new dashboard, but I need larger text", ["positive_feedback", "accessibility_request", "feature_request"]),
("I'm confused because the invoice amount is wrong", ["confusion", "billing_payment", "complaint"]),
("The package hasn't arrived; can I get a refund?", ["delivery_shipping", "refund_request", "complaint"]),
("Please schedule a call and remind me an hour before", ["booking_scheduling", "reminder_request", "command"]),
("Thanks for explaining; can you show another Python example?", ["gratitude", "coding_help", "learning_help", "question"]),
("I want to delete my account and all personal data", ["cancellation", "privacy_concern", "command"]),
("This is urgent: the app crashes whenever I export a PDF", ["urgency", "bug_report", "data_export"]),
("Could you add Urdu, and make the screen reader labels better?", ["language_request", "accessibility_request", "feature_request"]),
("No, that's not what I meant; I was asking about Android", ["rejection", "clarification"]),
("The app looks nice, but the animation makes it lag", ["positive_feedback", "performance_issue", "complaint"]),
("I forgot my password and think someone accessed my account", ["password_reset", "security_concern", "account_login"]),
("Can you compare these laptops and recommend one for coding?", ["comparison_request", "product_recommendation", "coding_help"]),
("I don't understand this error; help me debug the Python function", ["confusion", "coding_help", "troubleshooting"]),
("Please stop the renewal and send me the invoice for last month", ["cancellation", "billing_payment", "command"]),
("The update is useful, but my saved changes disappeared", ["positive_feedback", "bug_report", "complaint"]),
("Could you explain your privacy policy and let me download my data?", ["privacy_concern", "data_export", "information_request"]),
("I'm happy with the service, but support hasn't replied for a week", ["positive_feedback", "complaint"]),
("Good morning! Can you help me solve this equation?", ["greeting", "math_help", "question"]),
("No thanks, please don't delete the project", ["rejection", "command"]),
("My order is late, and the tracking page is broken", ["order_status", "delivery_shipping", "bug_report", "complaint"]),
("I need a quick answer because the deadline is today", ["urgency", "question"]),
("Please switch to dark mode and save that as my default", ["feature_request", "command"]),
("I appreciate the help; I'm still confused about the API", ["gratitude", "confusion", "coding_help"]),
("This feature is terrible, and the app closes without warning", ["negative_feedback", "bug_report", "complaint"]),
("Can I book a meeting for Friday and get a reminder?", ["booking_scheduling", "reminder_request", "question"]),
("Translate this note to Urdu, then summarize it", ["translation_request", "summarization_request", "command"]),
("Who are you, and what kinds of tasks can you handle?", ["identity_question", "capability_question"]),
("The payment went through but my account still says unpaid", ["billing_payment", "bug_report", "complaint"]),
("The text is too small and the contrast is hard to read", ["accessibility_request", "complaint"]),
("Please add a JSON export option; CSV alone isn't enough", ["feature_request", "data_export", "negative_feedback"]),
("I'm sorry, I sent the wrong file. Can I upload another?", ["apology", "question", "command"]),
("The reset link expired; how do I create a new one?", ["password_reset", "how_to", "question"]),
("I received a suspicious login email and need to secure my account", ["security_concern", "account_login", "urgency"]),
("I need the app to work without a connection", ["feature_request"]),
("The invoice sent two receipts for one payment", ["billing_payment", "complaint"]),
("Could you turn these notes into a short recap?", ["summarization_request"]),
("The keyboard navigation is difficult and needs improvement", ["accessibility_request", "complaint", "feature_request"]),
("Please explain how to change my profile picture", ["how_to", "question"]),
("I appreciate the update, but I still need offline access", ["positive_feedback", "feature_request"]),
("The buttons are too small to tap on my mobile screen", ["accessibility_request", "complaint"]),
("Please give me a short summary of this report", ["summarization_request"]),
]


def add(rows, text, labels):
    text = " ".join(str(text).split()).strip()
    labels = sorted(set(labels))
    if text and labels:
        rows.append((text, "|".join(labels)))


def build():
    rows = []
    for label, examples in SEEDS.items():
        for text in examples:
            add(rows, text, [label])
    # Generated, slot-based variations.
    from itertools import product
    for label, spec in SLOT_TEMPLATES.items():
        # Concatenate each ordered slot group (prefix, phrase, optional ending).
        # Each spec is intentionally written in grammatical order above.
        for parts in product(*spec.values()):
            add(rows, "".join(parts), [label])
    for text, labels in MULTI:
        add(rows, text, labels)
        # Safe casing/punctuation variants, deduplicated below.
        add(rows, text + ".", labels)
    # Balanced general style variants for a subset, avoiding a universal prefix bias.
    # They preserve the underlying task while changing request phrasing.
    paraphrase_rules = {
      "coding_help": ["Can you help me with this: {}", "I'm stuck on this coding task: {}"],
      "writing_help": ["Could you help with the wording? {}", "Please help me polish this: {}"],
      "math_help": ["Can you walk me through this? {}", "I need a step-by-step maths explanation: {}"],
      "learning_help": ["Help me learn this: {}", "Can you teach me this topic? {}"],
      "question": ["Quick question: {}", "I'd like to ask: {}"],
      "feature_request": ["Feature idea: {}", "My suggestion is: {}"],
      "bug_report": ["Bug report: {}", "I'm seeing this issue: {}"],
      "complaint": ["I'd like to raise a concern: {}", "I'm unhappy because {}"],
      "privacy_concern": ["Question about privacy: {}", "I have a data privacy concern: {}"],
      "security_concern": ["Security concern: {}", "I want to report a security issue: {}"],
      "data_export": ["I need a data copy: {}", "Please help me download this: {}"],
      "product_recommendation": ["Help me pick something: {}", "What would you recommend? {}"],
    }
    # Use only seed examples to make paraphrases, not generated sentence fragments.
    for label, wrappers in paraphrase_rules.items():
        for base in SEEDS.get(label, [])[:14]:
            for wrapper in wrappers:
                add(rows, wrapper.format(base.rstrip(".!?")), [label])
    # Deterministic shuffle before de-duplication.
    # Keep first copy of text; if repeated text occurs with multiple labels, merge labels.
    merged = {}
    for text, labelstr in rows:
        labels = set(labelstr.split("|"))
        key = text.casefold().strip()
        if key not in merged:
            merged[key] = [text, labels]
        else:
            merged[key][1].update(labels)
    out = [(v[0], "|".join(sorted(v[1]))) for v in merged.values()]
    RNG.shuffle(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "labels"])
        w.writerows(out)
    print(f"Wrote {len(out)} examples to {OUT}")
    print(f"Intents: {len(SEEDS)}")
    print(f"Multi-label examples: {sum('|' in labels for _, labels in out)}")

if __name__ == "__main__":
    build()
