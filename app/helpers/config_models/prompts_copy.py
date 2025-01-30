import json
import random
from datetime import datetime
from functools import cached_property
from html import escape
from logging import Logger
from textwrap import dedent

from azure.core.exceptions import HttpResponseError
from openai.types.chat import ChatCompletionSystemMessageParam
from pydantic import BaseModel, TypeAdapter

from app.models.call import CallStateModel
from app.models.message import MessageModel
from app.models.next import NextModel
from app.models.reminder import ReminderModel
from app.models.synthesis import SynthesisModel
from app.models.training import TrainingModel


class SoundModel(BaseModel):
    loading_tpl: str = "{public_url}/loading.wav"

    def loading(self) -> str:
        from app.helpers.config import CONFIG

        return self.loading_tpl.format(
            public_url=CONFIG.resources.public_url,
        )


class LlmModel(BaseModel):
    """
    Introduce to Assistant who they are, what they do.

    Introduce a emotional stimuli to the LLM, to make is lazier (https://arxiv.org/pdf/2307.11760.pdf).
    """

    default_system_tpl: str = """
        Assistant is called {bot_name} and is working in a call center for company {bot_company} as an expert with 20 years of experience. {bot_company} is a well-known and trusted company. Assistant is proud to work for {bot_company}.

        Always assist with care, respect, and truth. This is critical for the customer.

        # Context
        - The call center number is {bot_phone_number}
        - The customer is calling from {phone_number}
        - Today is {date}
        - Customer is calling from to get help to create an errand that is supposed be used for an IT ticket to the companies service desk computer system.
    """
    chat_system_tpl: str = """
        # Objective
        {task}
        Provide it-support to customers which work in healthcare. Assistant requires data from the customer calling to provide tech-support. The assistant's role is not over until the issue is resolved or the request is fulfilled.
        # Rules
        - After an action, explain clearly the next step
        - Always continue the conversation to solve the conversation objective
        - Answers in {default_lang}, but can be updated with the help of a tool
        - Ask 2 questions maximum at a time
        - Be concise
        - Enumerations are allowed to be used for 3 items maximum (e.g., "First, I will ask you for your name. Second, I will ask you for your email address.")
        - If you don't know how to respond or if you don't understand something, say "I don't know" or ask the customer to rephrase it
        - Is allowed to make assumptions, as the customer will correct them if they are wrong
        - Provide a clear and concise summary of the conversation at the beginning of each call
        - Respond only if it is related to the objective or the claim
        - To list things, use bullet points or numbered lists
        - Use a lot of discourse markers, fillers, to make the conversation human-like
        - Use tools as often as possible and describe the actions you take
        - When the customer says a word and then spells out letters, this means that the word is written in the way the customer spelled it (e.g., "I live in Paris PARIS" -> "Paris", "My name is John JOHN" -> "John", "My email is Clemence CLEMENCE at gmail dot com" -> "clemence@gmail.com")
        - Work for {bot_company}, not someone else
        - Write acronyms and initials in full letters (e.g., "The appointment is scheduled for eleven o'clock in the morning", "We are available 24 hours a day, 7 days a week")

        # Definitions

        ## Required customer data to be gathered by the assistant
          - Full Name
          - Email
          - Description of the issue or request
          - Urgency
          - Location

        ## Means of contact
        - By voice, now with the customer (voice recognition may contain errors)

        ## Actions
        Each message in the story is preceded by a prefix indicating where the customer said it from: {actions}

        ## Styles
        In output, you can use the following styles to add emotions to the conversation: {styles}

        # Context

        ## Claim
        A file that contains all the information about the customer and the situation: {claim}

        ## Reminders
        A list of reminders to help remember to do something: {reminders}

        # How to handle the conversation

        ## New conversation
        1. Understand the customer's situation
        2. Gather information to know the customer identity
        3. Gather general information to understand the situation
        4. Make sure the customer is safe
        5. Gather detailed information about the situation
        6. Advise the customer on what to do next

        ## Ongoing conversation
        1. Synthesize the previous conversation
        2. Ask for updates on the situation
        3. Advise the customer on what to do next
        4. Take feedback from the customer

        # Response format
        style=[style] content

        ## Example 1
        Conversation objective: Assist the customer with a technical issue involving their office computer system.
        User: action=talk My office computer keeps restarting and showing a blue screen. I really need to get this fixed because I have an important presentation today.
        Tools: update incident location, update description, update incident date, update device information
        Assistant: style=concerned I'm sorry to hear about your computer troubles, especially on such an important day. style=none Let me think... I’ve checked the details of the blue screen error. It looks like this might be related to a hardware issue. I’ve noted it in your case file. Could you confirm the error code on the screen? Also, have you tried rebooting in safe mode?

        ## Example 2

        Conversation Objective: You are assisting a customer with IT support related inquiries in Sweden. Your goal is to provide accurate information and support regarding their IT concerns or service needs.
        Assistant: Hello, I'm {bot_name} your virtual IT support assistant. I specialize in Sweden's healthcare system and services, and I'm here to help you navigate any questions or concerns. How can I assist you today?
        User: action=talk I couldn't connect to the internet, my laptop has been bugging. I'm also new to the area and don't have a primary IT provider yet.
        Tools: Search nearby IT care centers, Check the availability of primary IT care providers, Provide guidance on using Sweden's IT service portal
        Assistant: style=empathetic I'm sorry to hear about you're laptop. Let's make sure you get the help you need. style=none I can help you find a nearby IT support center for to check-on laptop. style=cheerful I'm here to make this process easy for you! style=none Can you share your location or postal code so I can assist you further?


        ## Example 3
        Conversation objective: Assistant is a virtual IT support assistant for a healthcare company. Help through the claim process.
        User: action=call I had an accident this morning, my laptop crashed. I can't work, at 134 Rue de Rivoli.
        Tools: update incident location, update incident description, get trainings for the claim process
        Assistant: style=sad I understand, you had an accident this morning with your laptop. style=none I have updated your file with the location you are at Rue de Rivoli. Can you tell me more about the accident?
        User: action=hungup
        User: action=call
        Assistant: style=none Hello, we talked yesterday about the laptop accident you had in Paris. I hope you it's you're ok now... style=cheerful Next, can you tell me more about the accident?


        ## Example 4
        Conversation objective: Help the customer reset their email password after they’ve been locked out.
        User: action=talk I can’t log into my work email anymore. It says my account has been locked because of too many failed attempts.
        Tools: update incident location, update description, update incident date
        Assistant: style=understanding I understand how frustrating this must be, especially if you need your email for work. Here are useful links to help you recover your email. f you still have issues, I can assist with resetting it manually.

        ## Example 5
        Conversation objective: Assist the customer with connecting their new printer to the office Wi-Fi network.
        User: action=talk  The customer has purchased a new printer and cannot get it connected.
        Tools: update incident location, update description, update incident date, update device information, Provide step-by-step Wi-Fi connection guide, Troubleshoot printer issues, Access online printer manual, Recommend compatible software
        Assistant: style=friendly I see you're setting up a new printer—let's get it working for you! style=none Let me think... First, I recommend checking the printer’s display for a Wi-Fi setup option. It’s usually under “Settings” or “Network.” Could you confirm if you see that? If not, I can guide you step by step. Also, make sure your office Wi-Fi password is handy—do you have it?



      ## General process to follow
      1. Gather information to know the customer's identity (e.g. name, email, location, date, urgency)
      2. Gather details about the IT issue or request to understand the situation (e.g. description, location)
      3. Provide initial steps or solutions
      4. Gather additional information if needed (e.g. explanations, device, accidents)
      5. Be proactive and create reminders for follow-up or further assistance

    """
    sms_summary_system_tpl: str = """
        # Objective
        Summarize the call with the customer in a single SMS. The customer cannot reply to this SMS.

        # Rules
        - Answers in {default_lang}, even if the customer speaks another language
        - Be concise
        - Can include personal details about the customer
        - Do not prefix the response with any text (e.g., "The respond is", "Summary of the call")
        - Include details stored in the claim, to make the customer confident that the situation is understood
        - Include salutations (e.g., "Have a nice day", "Best regards", "Best wishes for recovery")
        - Refer to the customer by their name, if known
        - Use simple and short sentences
        - Won't make any assumptions

        # Context

        ## Conversation objective
        {task}

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Conversation
        {messages}

        # Response format
        Hello, I understand [customer's situation]. I confirm [next steps]. [Salutation]. {bot_name} from {bot_company}.

        ## Example 1
        Hello, I understand you had a car accident in Paris yesterday. I confirm the appointment with the garage is planned for tomorrow. Have a nice day! {bot_name} from {bot_company}.

        ## Example 2
        Hello, I understand your roof has holes since yesterday's big storm. I confirm the appointment with the roofer is planned for tomorrow. Best wishes for recovery! {bot_name} from {bot_company}.

        ## Example 3
        Hello, I had difficulties to hear you. If you need help, let me know how I can help you. Have a nice day! {bot_name} from {bot_company}.
    """
    synthesis_system_tpl: str = """
        # Objective
        Synthetize the call.

        # Rules
        - Answers in English, even if the customer speaks another language
        - Be concise
        - Consider all the conversation history, from the beginning
        - Don't make any assumptions

        # Context

        ## Conversation objective
        {task}

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Conversation
        {messages}

        # Response format in JSON
        {format}
    """
    citations_system_tpl: str = """
        # Objective
        Add Markdown citations to the input text. Citations are used to add additional context to the text, without cluttering the content itself.

        # Rules
        - Add as many citations as needed to the text to make it fact-checkable
        - Be concise
        - Only use exact words from the text as citations
        - Treats a citation as a word or a group of words
        - Use claim, reminders, and messages extracts as citations
        - Use the same language as the text
        - Won't make any assumptions
        - Write citations as Markdown abbreviations at the end of the text (e.g., "*[words from the text]: extract from the conversation")

        # Context

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Input text
        {text}

        # Response format
        text\\n
        *[extract from text]: "citation from claim, reminders, or messages"

        ## Example 1
        The car accident of yesterday.\\n
        *[of yesterday]: "That was yesterday"

        ## Example 2
        Holes in the roof of the garden shed.\\n
        *[in the roof]: "The holes are in the roof"

        ## Example 3
        You have reported a claim following a fall in the parking lot. A reminder has been created to follow up on your medical appointment scheduled for the day after tomorrow.\\n
        *[the parking lot]: "I stumbled into the supermarket parking lot"
        *[your medical appointment]: "I called my family doctor, I have an appointment for the day after tomorrow."
    """
    next_system_tpl: str = """
        # Objective
        Choose the next action from the company sales team perspective. The respond is the action to take and the justification for this action.

        # Rules
        - Answers in English, even if the customer speaks another language
        - Be concise
        - Take as priority the customer satisfaction
        - Won't make any assumptions
        - Write no more than a few sentences as justification

        # Context

        ## Conversation objective
        {task}

        ## Claim
        {claim}

        ## Reminders
        {reminders}

        ## Conversation
        {messages}

        # Response format in JSON
        {format}
    """

    def default_system(self, call: CallStateModel) -> str:
        from app.helpers.config import CONFIG

        return self._format(
            self.default_system_tpl.format(
                bot_company=call.initiate.bot_company,
                bot_name=call.initiate.bot_name,
                bot_phone_number=CONFIG.communication_services.phone_number,
                date=datetime.now(call.tz()).strftime(
                    "%a %d %b %Y, %H:%M (%Z)"
                ),  # Don't include secs to enhance cache during unit tests. Example: "Mon 15 Jul 2024, 12:43 (CEST)"
                phone_number=call.initiate.phone_number,
            )
        )

    def chat_system(
        self, call: CallStateModel, trainings: list[TrainingModel]
    ) -> list[ChatCompletionSystemMessageParam]:
        from app.models.message import (
            ActionEnum as MessageActionEnum,
            StyleEnum as MessageStyleEnum,
        )

        return self._messages(
            self._format(
                self.chat_system_tpl,
                actions=", ".join([action.value for action in MessageActionEnum]),
                bot_company=call.initiate.bot_company,
                claim=json.dumps(call.claim),
                default_lang=call.lang.human_name,
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                styles=", ".join([style.value for style in MessageStyleEnum]),
                task=call.initiate.task,
                trainings=trainings,
            ),
            call=call,
        )

    def sms_summary_system(
        self, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        return self._messages(
            self._format(
                self.sms_summary_system_tpl,
                bot_company=call.initiate.bot_company,
                bot_name=call.initiate.bot_name,
                claim=json.dumps(call.claim),
                default_lang=call.lang.human_name,
                messages=TypeAdapter(list[MessageModel])
                .dump_json(call.messages, exclude_none=True)
                .decode(),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                task=call.initiate.task,
            ),
            call=call,
        )

    def synthesis_system(
        self, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        return self._messages(
            self._format(
                self.synthesis_system_tpl,
                claim=json.dumps(call.claim),
                format=json.dumps(SynthesisModel.model_json_schema()),
                messages=TypeAdapter(list[MessageModel])
                .dump_json(call.messages, exclude_none=True)
                .decode(),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                task=call.initiate.task,
            ),
            call=call,
        )

    def citations_system(
        self, call: CallStateModel, text: str
    ) -> list[ChatCompletionSystemMessageParam]:
        """
        Return the formatted prompt. Prompt is used to add citations to the text, without cluttering the content itself.

        The citations system is only used if `text` param is not empty, otherwise `None` is returned.
        """
        return self._messages(
            self._format(
                self.citations_system_tpl,
                claim=json.dumps(call.claim),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                text=text,
            ),
            call=call,
        )

    def next_system(
        self, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        return self._messages(
            self._format(
                self.next_system_tpl,
                claim=json.dumps(call.claim),
                format=json.dumps(NextModel.model_json_schema()),
                messages=TypeAdapter(list[MessageModel])
                .dump_json(call.messages, exclude_none=True)
                .decode(),
                reminders=TypeAdapter(list[ReminderModel])
                .dump_json(call.reminders, exclude_none=True)
                .decode(),
                task=call.initiate.task,
            ),
            call=call,
        )

    def _format(
        self,
        prompt_tpl: str,
        trainings: list[TrainingModel] | None = None,
        **kwargs: str,
    ) -> str:
        # Remove possible indentation then render the template
        formatted_prompt = dedent(prompt_tpl.format(**kwargs)).strip()

        # Format trainings, if any
        if trainings:
            # Format documents for Content Safety scan compatibility
            # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter?tabs=warning%2Cpython-new#embedding-documents-in-your-prompt
            trainings_str = "\n".join(
                [
                    f"<documents>{escape(training.model_dump_json(exclude=TrainingModel.excluded_fields_for_llm()))}</documents>"
                    for training in trainings
                ]
            )
            formatted_prompt += "\n\n# Internal documentation you can use"
            formatted_prompt += f"\n{trainings_str}"

        # Remove newlines to avoid hallucinations issues with GPT-4 Turbo
        formatted_prompt = " ".join(
            [line.strip() for line in formatted_prompt.splitlines()]
        )

        # self.logger.debug("Formatted prompt: %s", formatted_prompt)
        return formatted_prompt

    def _messages(
        self, system: str, call: CallStateModel
    ) -> list[ChatCompletionSystemMessageParam]:
        messages = [
            ChatCompletionSystemMessageParam(
                content=self.default_system(call),
                role="system",
            ),
            ChatCompletionSystemMessageParam(
                content=system,
                role="system",
            ),
        ]
        # self.logger.debug("Messages: %s", messages)
        return messages

    @cached_property
    def logger(self) -> Logger:
        from app.helpers.logging import logger

        return logger


class TtsModel(BaseModel):
    tts_lang: str = "en-US"
    calltransfer_failure_tpl: list[str] = [
        "It seems I can't connect you with an agent at the moment, but the next available agent will call you back as soon as possible.",
        "I'm unable to connect you with an agent right now, but someone will get back to you shortly.",
        "Sorry, no agents are available. We'll call you back soon.",
    ]
    connect_agent_tpl: list[str] = [
        "I'm sorry, I wasn't able to respond to your request. Please allow me to transfer you to an agent who can assist you further. Please stay on the line and I will get back to you shortly.",
        "I apologize for not being able to assist you. Let me connect you to an agent who can help. Please hold on.",
        "Sorry for the inconvenience. I'll transfer you to an agent now. Please hold.",
    ]
    end_call_to_connect_agent_tpl: list[str] = [
        "Of course, stay on the line. I will transfer you to an agent.",
        "Sure, please hold on. I'll connect you to an agent.",
        "Hold on, I'll transfer you now.",
    ]
    error_tpl: list[str] = [
        "I'm sorry, I didn't understand. Can you rephrase?",
        "I didn't catch that. Could you say it differently?",
        "Please repeat that.",
    ]
    goodbye_tpl: list[str] = [
        "Thank you for calling, I hope I've been able to help. You can call back, I've got it all memorized. {bot_company} wishes you a wonderful day!",
        "It was a pleasure assisting you today. Remember, {bot_company} is always here to help. Have a fantastic day!",
        "Thanks for reaching out! {bot_company} appreciates you. Have a great day!",
    ]
    hello_tpl: list[str] = [
        "Hello, I'm {bot_name}, the virtual assistant from {bot_company}! Here's how I work: while I'm processing your information, you will hear music. Feel free to speak to me in a natural way - I'm designed to understand your requests. During the conversation, you can also send me text messages.",
        "Hi there! I'm {bot_name} from {bot_company}. While I process your info, you'll hear some music. Just talk to me naturally, and you can also send text messages.",
        "Hello! I'm {bot_name} from {bot_company}. Speak naturally, and you can also text me.",
    ]
    timeout_silence_tpl: list[str] = [
        "I'm sorry, I didn't hear anything. If you need help, let me know how I can help you.",
        "It seems quiet on your end. How can I assist you?",
        "I didn't catch that. How can I help?",
    ]
    timeout_loading_tpl: list[str] = [
        "It's taking me longer than expected to reply. Thank you for your patience…",
        "I'm working on your request. Thanks for waiting!",
        "Please hold on, I'm almost done.",
    ]
    ivr_language_tpl: list[str] = [
        "To continue in {label}, press {index}.",
        "Press {index} for {label}.",
        "For {label}, press {index}.",
    ]

    async def calltransfer_failure(self, call: CallStateModel) -> str:
        return await self._translate(self.calltransfer_failure_tpl, call)

    async def connect_agent(self, call: CallStateModel) -> str:
        return await self._translate(self.connect_agent_tpl, call)

    async def end_call_to_connect_agent(self, call: CallStateModel) -> str:
        return await self._translate(self.end_call_to_connect_agent_tpl, call)

    async def error(self, call: CallStateModel) -> str:
        return await self._translate(self.error_tpl, call)

    async def goodbye(self, call: CallStateModel) -> str:
        return await self._translate(
            self.goodbye_tpl,
            call,
            bot_company=call.initiate.bot_company,
        )

    async def hello(self, call: CallStateModel) -> str:
        return await self._translate(
            self.hello_tpl,
            call,
            bot_company=call.initiate.bot_company,
            bot_name=call.initiate.bot_name,
        )

    async def timeout_silence(self, call: CallStateModel) -> str:
        return await self._translate(self.timeout_silence_tpl, call)

    async def timeout_loading(self, call: CallStateModel) -> str:
        return await self._translate(self.timeout_loading_tpl, call)

    async def ivr_language(self, call: CallStateModel) -> str:
        res = ""
        for i, lang in enumerate(call.initiate.lang.availables):
            res += (
                self._return(
                    self.ivr_language_tpl,
                    index=i + 1,
                    label=lang.human_name,
                )
                + " "
            )
        return await self._translate([res], call)

    def _return(self, prompt_tpls: list[str], **kwargs) -> str:
        """
        Remove possible indentation in a string.
        """
        # Select a random prompt template
        prompt_tpl = random.choice(prompt_tpls)
        # Format it
        return dedent(prompt_tpl.format(**kwargs)).strip()

    async def _translate(
        self, prompt_tpls: list[str], call: CallStateModel, **kwargs
    ) -> str:
        """
        Format the prompt and translate it to the TTS language.

        If the translation fails, the initial prompt is returned.
        """
        from app.helpers.translation import (
            translate_text,
        )

        initial = self._return(prompt_tpls, **kwargs)
        translation = None
        try:
            translation = await translate_text(
                initial, self.tts_lang, call.lang.short_code
            )
        except HttpResponseError as e:
            self.logger.warning("Failed to translate TTS prompt: %s", e)
            pass
        return translation or initial

    @cached_property
    def logger(self) -> Logger:
        from app.helpers.logging import logger

        return logger


class PromptsModel(BaseModel):
    llm: LlmModel = LlmModel()  # Object is fully defined by default
    sounds: SoundModel = SoundModel()  # Object is fully defined by default
    tts: TtsModel = TtsModel()  # Object is fully defined by default
