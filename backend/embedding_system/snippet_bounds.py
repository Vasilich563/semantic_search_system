class SnippetBounds:

    def __init__(self, start_index_in_text, after_last_index_in_text):
        self._start_index_in_text = start_index_in_text
        self._after_last_index_in_text = after_last_index_in_text


    @property
    def snippet_start_index(self):
        """Index from the original text, corresponding to the first char of the snippet """
        return self._start_index_in_text

    @property
    def snippet_end_index(self):
        """Index from the original text, corresponding to the next char to the last char of the snippet (last in the snippet + 1)"""
        return self._start_index_in_text


