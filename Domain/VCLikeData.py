from pykson import JsonObject, IntegerField, StringField, ObjectListField, DateTimeField
from ML.Domain.LearningKnowledge import LearningKnowledge

class VCLikeData(JsonObject):
    # def __init__(self, learning_settings, learning_settings2):
    #     self.learning_settings = learning_settings
    #     self.learning_settings2 = learning_settings2
    
    learning_settings = LearningKnowledge()
    learning_settings2 = LearningKnowledge()
