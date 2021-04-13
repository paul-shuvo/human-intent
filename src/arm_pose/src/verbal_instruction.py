#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import speech_recognition as sr

import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words -= {'give', 'Give', 'put', 'Put',
                            'this', 'This', 'that', 'That',
                            'here', 'Here', 'there', 'There'}


matcher = Matcher(nlp.vocab)

# r = sr.Recognizer()
# mic = sr.Microphone(device_index=9)
# with mic as source:
#     audio = r.listen(source)
# instruction = r.recognize_google(audio)

# TODO: Add docstring
# TODO: Add language pattern.


class VerbalInstruction:
    """
    Class for the verbal instruction node. It records the verbal
    instruction, transcribes and publishes it.
    """

    def __init__(self, device_index: int = 9):
        """
        Initializes the instance.

        Parameters
        ----------
        device_index : int, optional
            Index for the audion input that the speech recognizer
            would listen to, by default 9
        """
        self.speech_recognizer = sr.Recognizer()
        self.mic = sr.Microphone(device_index=device_index)

        rospy.init_node("verbal_instruction_pub_node", anonymous=False)
        self.verbal_instruction_pub = rospy.Publisher(
            "/verbal_instruction", String, queue_size=10
        )
        self.instruction_msg = ""
        # Run the publisher after initiation
        self.run()

    def run(self):
        """
        Runs the publisher node. Publishes verbal instructions
        received from the user.
        """
        rate = rospy.Rate(1)  # 1hz
        while not rospy.is_shutdown():
            with self.mic as source:
                audio = self.speech_recognizer.listen(source)
            # if no instruction is received, go to the
            # next iteration.
            try:
                self.instruction_msg = self.speech_recognizer.recognize_google(audio)
                if self.instruction_msg in ["exit", "quit"]:
                    rospy.signal_shutdown("User command")
            except sr.UnknownValueError:
                rospy.loginfo("Speaker is quiet")
                continue
            rospy.loginfo(self.instruction_msg)
            self.extract_object_info(self.instruction_msg)
            self.verbal_instruction_pub.publish(self.instruction_msg)
            rate.sleep()

    def extract_object_info(self, instruction_msg):
        matcher.add("action", None, [{"POS": "VERB"},
                                     {"POS": "PRON", "OP": "*"},
                                     {},
                                     {"POS": "ADJ", "OP": "*"},
                                     {"POS": "NOUN"}])

        matcher.add("attr", None, [{"POS": "ADJ", "OP": "+"}, {"POS": "NOUN"}])
        matcher.add("pos", None, [{"LEMMA": {"IN": ["right", "left", "front", "back"]}}])

        doc = nlp(instruction_msg)
        matches = matcher(doc)
        object_info = {}
        object_name = ""
        action = ""
        attr = []
        pos = ""

        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            print(string_id)
            if string_id == "action":
                object_name = doc[end-1]
                action = doc[start]

            if string_id == "attr":
                attr.append(doc[start])

            if string_id == "pos":
                pos = doc[start]
            # span = doc[start:end]  # The matched span
            # print(string_id, start, end, span.text)

        object_info[object_name] = {}
        object_info[object_name]["action"] = action
        object_info[object_name]["attr"] = attr
        object_info[object_name]["pos"] = pos
        print(object_info)


if __name__ == "__main__":
    VerbalInstruction(device_index=1)
