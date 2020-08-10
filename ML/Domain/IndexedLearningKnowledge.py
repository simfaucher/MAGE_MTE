from pykson import JsonObject, IntegerField, StringField, ObjectListField, DateTimeField
from ML.Domain.LearningKnowledge import LearningKnowledge

class IndexedLearningKnowledge(JsonObject):
    scale = IntegerField()
    learning_knowledge = LearningKnowledge()