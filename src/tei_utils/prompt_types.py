# src/tei_utils/prompt_types.py

class PromptType:
    """
    Типы промптов для TEI.

    ВАЖНО: тип промптра - это НЕ prompt_name для сервера, 
    а просто указание на желаемый промпт, 
    имя которого должно быть выяснено отдельно
    """
    QUERY = "query"
    DOCUMENT = "document"