from pykson import JsonObject, IntegerField, StringField, ObjectListField, DateTimeField
from ML.Domain.LearningKnowledge import LearningKnowledge
from ML.Domain.IndexedLearningKnowledge import IndexedLearningKnowledge
from ML import BoxLearner

class VCLikeData(JsonObject):
    # def __init__(self, learning_settings, learning_settings2):
    #     self.learning_settings = learning_settings
    #     self.learning_settings2 = learning_settings2
    
    # learning_settings = LearningKnowledge()
    # learning_settings2 = LearningKnowledge()

    learning_settings_85_singlescale = ObjectListField(IndexedLearningKnowledge)
    learning_settings_85_multiscale = LearningKnowledge()
    learning_settings_64_singlescale = ObjectListField(IndexedLearningKnowledge)
    learning_settings_64_multiscale = LearningKnowledge()
