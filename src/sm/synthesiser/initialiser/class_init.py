from .match_methods import match_methods_class

class init_class(match_methods_class):
    def __init__(self, method="faker"):
        self.outputpooling = {}
        self.word_list, methods_map = self.list_match_methods(method)
        self.word_tokens = {word: word.split("_") for word in self.word_list}
        self.word_tokens_set = {word: set(word.split("_")) for word in self.word_list}
        self.resolved_methods = self.make_resolved_methods(self.word_list, methods_map)