import translators as ts
def translate_text_with_translators(text, source_language, target_language):
    translated_text = ts.translate_text(text, translator='bing',
                                        from_language=source_language,
                                        to_language=target_language)
    return translated_text

