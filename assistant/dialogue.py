"""
Görevi: Kullanıcı ile konuşma akışını yönetir -- teyit, ilerleme anlatımı,
takip soruları ve "bulamadım" cevabı.

Why this exists, and why in this shape: the reference work (Wei et al., CHI '26)
found four things about talking to an assistive drone that this project had
none of.

  Confirm before acting. They added a confirmation step after transcription
  errors turned directly into task failures ("plant" heard as "clamp"). A wrong
  command that flies is worse than one that asks.

  Say what you are doing. P6: "while the drone is locating, it tells me where
  it is looking. So at least I have an orientation, which I find very, very
  useful."

  Expect follow-ups. 35% of all their user queries were follow-up questions
  about a previous answer, so a single-shot request/response is the wrong
  shape.

  Say less, not more. Hearing is a safety channel for a blind person -- P1:
  "as a blind person, I have to listen to the surroundings, otherwise I cannot
  know if someone, maybe a cyclist, is coming." Their v3 deliberately capped
  response length. Narration here is rate-limited for the same reason.

To which this adds one thing they did not need: a bounded give-up. Their drone
visited a fixed list of waypoints and stopped; ours searches autonomously and
would otherwise search forever.
"""
from __future__ import annotations

import re
import time

from assistant.command_parser import normalize_target, parse_command

# Minimum gap between spoken progress updates. Narration is meant to give the
# user a sense of orientation, not to fill the audio channel they need for
# situational awareness.
NARRATION_MIN_GAP = 8.0     # seconds

# A repeat of the previous update has to wait much longer. Observed in flight:
# the search ran for 40 s inside one direction bucket and said "hafif sağınızda
# arıyorum" five times, which carries no information after the first and is
# exactly the audio overload the rate limit exists to prevent. Repeats are not
# dropped outright, though -- silence reads as a crash, so a periodic reminder
# that the search is still running is worth the interruption.
NARRATION_REPEAT_GAP = 30.0     # seconds

# How long to search before admitting failure, when the caller does not say.
# A caller that knows its own coverage pattern should pass a timeout derived
# from it instead: this default is a guess, and a guess shorter than one sweep
# turns "I have not looked there yet" into "it is not there".
SEARCH_TIMEOUT = 300.0      # seconds

_YES = {'e', 'evet', 'y', 'yes', 'ok', 'tamam', 'onay', 'onaylıyorum', ''}
_NO = {'h', 'hayır', 'hayir', 'n', 'no', 'iptal', 'vazgeç', 'vazgec'}
# A rejection often arrives attached to the correction -- "hayır, plant bul".
# Matched as a word with any punctuation after it, because people type
# "no,phone" as readily as "no phone": a literal "no " prefix missed the
# comma form and sent the drone looking for an object called "no,phone".
_NO_PREFIX_RE = re.compile(r'^(hayır|hayir|yok|no|not)\b[\s,;.]*', re.I)
_QUIT = {'q', 'çıkış', 'cikis', 'quit', 'exit'}

# Follow-up detection. The reference system routed intent through the LLM; that
# costs a call per utterance, and the free tier allows 20 a day, so this is a
# deliberate cheap stand-in. It only has to fire when there is already an answer
# on the table, which narrows what an utterance is likely to mean.
_QUESTION_WORDS = {
    'ne', 'neler', 'nerede', 'nerde', 'nasıl', 'nasil', 'kaç', 'kac',
    'hangi', 'kim', 'niye', 'neden', 'mi', 'mı', 'mu', 'mü',
    'what', 'where', 'how', 'which', 'who', 'why',
}
# Substrings that make an utterance a question about the last answer even
# without a question word of its own ("üzerinde yazı var", "rengi").
_QUESTION_HINTS = (
    'var mı', 'var mi', 'yazıyor', 'yaziyor', 'yazı', 'yazi', 'renk', 'reng',
    'başka', 'baska', 'kaç tane', 'is there', 'are there', 'colour', 'color',
    'anlat', 'tarif',
)


class Dialogue:
    """Conversation state for one flight.

    The node calls submit() with whatever the user typed or said and acts on
    the returned dict; everything about *when* to speak lives here.
    """

    def __init__(self, speak, log):
        self._speak = speak
        self._log = log
        self.target = None
        self.pending_target = None      # awaiting confirmation
        self.history = []               # (role, text) for follow-up context
        self.have_answer = False        # a result has been delivered
        self._search_started = None
        self._timeout = SEARCH_TIMEOUT
        self._last_narration = 0.0
        self._last_narration_text = None

    # ---------------- speaking ----------------

    def say(self, text, log_prefix=">"):
        self._log(f"{log_prefix} {text}")
        print(f"\n[{log_prefix}] {text}")
        self._speak(text)
        self.history.append(("drone", text))

    def narrate(self, text):
        """Progress update, dropped if one was spoken too recently.

        Saying the same thing again is held to a longer interval than saying
        something new, so the user hears the search *change* rather than the
        search continue.
        """
        now = time.time()
        gap = (NARRATION_REPEAT_GAP if text == self._last_narration_text
               else NARRATION_MIN_GAP)
        if now - self._last_narration < gap:
            return False
        self._last_narration = now
        self._last_narration_text = text
        self.say(text, log_prefix="~")
        return True

    # ---------------- search lifecycle ----------------

    def start_search(self, target, timeout=None):
        self.target = target
        self._timeout = timeout or SEARCH_TIMEOUT
        self.pending_target = None
        self.have_answer = False
        self._search_started = time.time()
        self._last_narration = 0.0
        self._last_narration_text = None
        self.say(f"Looking for the {target}.")

    def extend_search(self, seconds, reason=""):
        """Give the current search more time, because something changed.

        The deadline is derived from how long one sweep of the area takes, so
        it is the right bound for sweeping and the wrong one for anything else.
        Going to inspect surfaces close up is slower per square metre and is
        the fallback for exactly the targets the sweep cannot find -- cutting
        it off on the sweep's clock means the expensive search never gets to
        finish, having already spent the cheap one.
        """
        if self.target is None or self._search_started is None:
            return
        self._timeout += max(0.0, seconds)
        return reason

    def search_expired(self):
        return (self.target is not None and self._search_started is not None
                and time.time() - self._search_started > self._timeout)

    def give_up(self):
        """An explicit, bounded failure. Silence or an endless hunt is not an
        answer a user can act on."""
        t = self.target
        self.target = None
        self.pending_target = None
        self._search_started = None
        self.have_answer = True
        self.say(f"I could not find the {t}. I have searched the whole area. "
                 f"You can try somewhere else, or ask for something different.")

    def abort(self, reason):
        """Stop the search for a reason that is not "it is not there".

        Kept separate from give_up() because the two are different answers and
        a user who cannot look has no other way to tell them apart. A run was
        observed where every vision call failed with a retired-model 404 and
        the drone went on sweeping in silence: it could not see at all, but was
        on course to report the object missing.

        have_answer stays False -- there is no result here to ask questions
        about.
        """
        self.target = None
        self.pending_target = None
        self._search_started = None
        self.have_answer = False
        self.say(reason)

    def record_answer(self, text):
        self.target = None
        self.pending_target = None
        self._search_started = None
        self.have_answer = True
        self.say(text)

    # ---------------- input ----------------

    def submit(self, text):
        """Route one utterance.

        Returns a dict with 'action' in:
          quit | confirm | start | cancel | followup | ignored
        """
        raw = (text or "").strip()
        low = raw.lower()

        if low in _QUIT:
            return {"action": "quit"}

        # "hayır, telefon" is a rejection and a correction in one breath, and
        # it arrives in both states: while a target awaits confirmation, and
        # just after an answer the user disagrees with. Handled before anything
        # else, because with nothing pending it fell through to the parser,
        # which found no verb and made the whole utterance the target -- the
        # drone was then sent looking for "hayır, telefon".
        m = _NO_PREFIX_RE.match(raw)
        if m:
            rest = raw[m.end():].strip(' ,')
            cancelled, self.pending_target = self.pending_target, None
            if rest:
                return self._propose(rest)
            self.say(f"Search for the {cancelled} cancelled. What should I look for?"
                     if cancelled else
                     "Tamam, iptal ettim. Ne aramamı istersiniz?")
            return {"action": "cancel"}

        # Awaiting a yes/no on a proposed target.
        if self.pending_target is not None:
            if low in _NO:
                cancelled, self.pending_target = self.pending_target, None
                self.say(f"Search for the {cancelled} cancelled. What should I look for?")
                return {"action": "cancel"}
            if low in _YES:
                # Cleared here, not left to the caller. start_search() also
                # clears it, but a dialogue that only leaves the confirmation
                # state when the node remembers to call back is one stuck
                # question away from treating every later utterance -- including
                # follow-ups -- as a correction to a target already accepted.
                confirmed, self.pending_target = self.pending_target, None
                return {"action": "start", "target": confirmed}
            # Anything else is treated as a correction rather than an answer.
            return self._propose(raw)

        if not raw:
            return {"action": "ignored"}

        # A bare "iptal" with nothing pending means stop what you are doing.
        # Without this it was parsed as a command and became the target.
        if low in _NO:
            self.target = None
            self._search_started = None
            self.say("All right, stopping. What should I look for?")
            return {"action": "cancel"}

        # With an answer already given, a question is about that answer.
        if self.have_answer and self._looks_like_question(low):
            self.history.append(("user", raw))
            return {"action": "followup", "question": raw}

        return self._propose(raw)

    # Words that mean the utterance is being said *to* the drone rather than
    # naming a thing for it to find. An object name is a short noun phrase;
    # "thank you", "tell me about pizza" and "do you want me to come back" are
    # not, and each of them was accepted as a search target during a rehearsal
    # -- the drone dutifully asked whether to look for "the thank you".
    _NOT_A_TARGET = {
        'thank', 'thanks', 'please', 'hello', 'hi', 'hey', 'ok', 'okay',
        'you', 'your', 'yours', 'me', 'my', 'i', 'we', 'us', 'he', 'she',
        'they', 'it', 'do', 'does', 'did', 'can', 'could', 'would', 'will',
        'shall', 'should', 'tell', 'say', 'talk', 'come', 'go', 'about',
        'sorry', 'good', 'well', 'now', 'again', 'back',
        'tesekkur', 'tesekkurler', 'sagol', 'merhaba', 'selam', 'lutfen',
        'bana', 'sen', 'sana', 'ben', 'bize', 'anlat', 'soyle', 'gel', 'git',
    }
    # An object name is short. Four words is already a sentence.
    _MAX_TARGET_WORDS = 3

    @classmethod
    def _looks_like_target(cls, raw: str) -> bool:
        words = [w.strip('.,!?;:"\'').lower() for w in raw.split()]
        words = [w for w in words if w]
        if not words or len(words) > cls._MAX_TARGET_WORDS:
            return False
        return not any(w in cls._NOT_A_TARGET for w in words)

    def _propose(self, raw):
        # Fall back to the lexicon, not to the raw string: an utterance with no
        # verb is still usually just the object's name.
        parsed = parse_command(raw).get("target")
        if parsed:
            target = parsed
        else:
            # normalize_target returns unknown text unchanged, so it cannot be
            # used to tell an object name from a sentence -- checking its
            # result was the first version of this guard and it never fired.
            # The utterance itself has to be judged before falling back to it.
            if not self._looks_like_target(raw):
                self.history.append(("user", raw))
                self.say("I did not catch an object in that. "
                         "What should I look for?")
                return {"action": "ignored"}
            target = normalize_target(raw) or raw
        self.pending_target = target
        self.history.append(("user", raw))
        self.say(f"Please confirm: you want me to look for the {target}. "
                 f"Press Enter for yes, h to cancel.", log_prefix="?")
        return {"action": "confirm", "target": target}

    @staticmethod
    def _looks_like_question(low):
        if low.endswith('?'):
            return True
        if any(h in low for h in _QUESTION_HINTS):
            return True
        # Turkish puts the question particle last, so check word by word rather
        # than searching for a substring -- 'ne' inside 'nerede' is not a match
        # for the word 'ne', and 'rengi ne' ends with it.
        return bool(_QUESTION_WORDS & set(low.replace('?', '').split()))

    def context(self, limit=6):
        """Recent turns, for giving a follow-up something to refer back to."""
        return self.history[-limit:]
