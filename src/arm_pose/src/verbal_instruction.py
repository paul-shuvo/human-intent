#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import speech_recognition as sr
# r = sr.Recognizer()
# mic = sr.Microphone(device_index=9)
# with mic as source:
#     audio = r.listen(source)
# instruction = r.recognize_google(audio)

class VerbalInstruction:
    def __init__(self, device_index=9):
        self.speech_recognizer = sr.Recognizer()
        self.mic = sr.Microphone(device_index=device_index)
        
        rospy.init_node('verbal_instruction_pub_node', anonymous=False)
        self.verbal_instruction_pub = rospy.Publisher('/verbal_instruction', 
                                                      String, 
                                                      queue_size=10)
        self.instruction_msg = ''
        self.run()
    
    def run(self):
        rate = rospy.Rate(1) # 10hz
        while not rospy.is_shutdown():
            with self.mic as source:
                audio = self.speech_recognizer.listen(source)
                print('here')
            
            try:
                self.instruction_msg = self.speech_recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                rospy.loginfo('Speaker is quiet')
                continue
            rospy.loginfo(self.instruction_msg)
            self.verbal_instruction_pub.publish(self.instruction_msg)
            rate.sleep()

if __name__ == '__main__':
    VerbalInstruction(device_index=9)