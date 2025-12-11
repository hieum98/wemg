"""
Test suite for the improved extract_info_from_text function.
"""

import pytest
from wemg.utils.parsing import extract_info_from_text


class TestExtractInfoFromText:
    """Test suite for extract_info_from_text parsing functionality."""
    
    def test_valid_json_parsing(self):
        """Test parsing of valid, complete JSON."""
        text = '{"name": "John", "age": 30, "active": true, "tags": ["python", "testing"]}'
        keys = ["name", "age", "active", "tags"]
        value_types = ["str", "int", "bool", "list"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["name"] == "John"
        assert result["age"] == 30
        assert result["active"] is True
        assert result["tags"] == ["python", "testing"]
        print("✓ Valid JSON parsing works")
    
    def test_incomplete_json(self):
        """Test parsing of incomplete or malformed JSON."""
        text = '{"name": "Alice", "age": 25, "tags": ["ai", "ml"'
        keys = ["name", "age", "tags"]
        value_types = ["str", "int", "list"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["name"] == "Alice"
        assert result["age"] == 25
        assert "ai" in result["tags"] and "ml" in result["tags"]
        print("✓ Incomplete JSON parsing works")
    
    def test_json_within_text(self):
        """Test extracting JSON from within larger text."""
        text = '''
        Here is some analysis:
        {"sentiment": "positive", "score": 8, "confident": true}
        That's my assessment.
        '''
        keys = ["sentiment", "score", "confident"]
        value_types = ["str", "int", "bool"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["sentiment"] == "positive"
        assert result["score"] == 8
        assert result["confident"] is True
        print("✓ JSON within text extraction works")
    
    def test_missing_fields(self):
        """Test handling of missing fields with appropriate defaults."""
        text = '{"name": "Bob"}'
        keys = ["name", "age", "active", "tags"]
        value_types = ["str", "int", "bool", "list"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["name"] == "Bob"
        assert result["age"] == 0  # Default for int
        assert result["active"] is False  # Default for bool
        assert result["tags"] == []  # Default for list
        print("✓ Missing fields get appropriate defaults")
    
    def test_regex_fallback_strings(self):
        """Test regex extraction when JSON parsing fails - strings."""
        text = '''
        The question is "What is 2+2?"
        The answer is "4"
        confidence level is "high"
        '''
        keys = ["question", "answer", "confidence"]
        value_types = ["str", "str", "str"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert "2+2" in result["question"]
        assert result["answer"] == "4"
        assert result["confidence"] == "high"
        print("✓ Regex fallback for strings works")
    
    def test_regex_fallback_numbers(self):
        """Test regex extraction for numbers."""
        text = '''
        "score": 95
        "temperature": 0.7
        "count": -5
        '''
        keys = ["score", "temperature", "count"]
        value_types = ["int", "float", "int"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["score"] == 95
        assert result["temperature"] == 0.7
        assert result["count"] == -5
        print("✓ Regex fallback for numbers works")
    
    def test_regex_fallback_boolean(self):
        """Test regex extraction for booleans."""
        text = '''
        "is_valid": true
        "has_error": false
        "enabled": True
        '''
        keys = ["is_valid", "has_error", "enabled"]
        value_types = ["bool", "bool", "bool"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["is_valid"] is True
        assert result["has_error"] is False
        assert result["enabled"] is True
        print("✓ Regex fallback for booleans works")
    
    def test_regex_fallback_lists(self):
        """Test regex extraction for lists."""
        text = '''
        "tags": ["python", "testing", "ai"]
        "numbers": [1, 2, 3]
        '''
        keys = ["tags", "numbers"]
        value_types = ["list", "list"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert len(result["tags"]) == 3
        assert "python" in result["tags"]
        assert "testing" in result["tags"]
        print("✓ Regex fallback for lists works")
    
    def test_llm_output_with_markdown(self):
        """Test parsing LLM output that includes markdown formatting."""
        text = '''
        ```json
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "confidence": "high"
        }
        ```
        '''
        keys = ["question", "answer", "confidence"]
        value_types = ["str", "str", "str"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert "France" in result["question"]
        assert result["answer"] == "Paris"
        assert result["confidence"] == "high"
        print("✓ LLM output with markdown parsing works")
    
    def test_malformed_quotes(self):
        """Test handling of mismatched or escaped quotes."""
        text = r'''
        {"name": "John \"The Expert\" Doe", "title": "Engineer"}
        '''
        keys = ["name", "title"]
        value_types = ["str", "str"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert "John" in result["name"]
        assert result["title"] == "Engineer"
        print("✓ Malformed quotes handling works")
    
    def test_newline_separated_lists(self):
        """Test extraction of lists with newline separators."""
        text = '''
        "items": [
            "first item",
            "second item",
            "third item"
        ]
        '''
        keys = ["items"]
        value_types = ["list"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert len(result["items"]) >= 2
        assert any("first" in item for item in result["items"])
        print("✓ Newline-separated lists work")
    
    def test_type_conversion_errors(self):
        """Test graceful handling of type conversion errors."""
        text = '''
        "score": "not a number"
        "active": "maybe"
        '''
        keys = ["score", "active"]
        value_types = ["int", "bool"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        # Should fall back to defaults when conversion fails
        assert isinstance(result["score"], int)
        assert isinstance(result["active"], bool)
        print("✓ Type conversion error handling works")
    
    def test_empty_text(self):
        """Test handling of empty or whitespace-only text."""
        text = "   "
        keys = ["name", "value"]
        value_types = ["str", "int"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["name"] == ""
        assert result["value"] == 0
        print("✓ Empty text handling works")
    
    def test_multiple_json_objects(self):
        """Test extraction when multiple JSON objects are present."""
        text = '''
        First object: {"name": "Alice", "age": 30}
        Second object: {"name": "Bob", "age": 25}
        '''
        keys = ["name", "age"]
        value_types = ["str", "int"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        # Should extract from the first valid JSON object found
        assert result["name"] in ["Alice", "Bob"]
        assert result["age"] in [30, 25]
        print("✓ Multiple JSON objects handling works")
    
    def test_case_insensitive_booleans(self):
        """Test case-insensitive boolean parsing."""
        text = '''
        "flag1": TRUE
        "flag2": False
        "flag3": true
        '''
        keys = ["flag1", "flag2", "flag3"]
        value_types = ["bool", "bool", "bool"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["flag1"] is True
        assert result["flag2"] is False
        assert result["flag3"] is True
        print("✓ Case-insensitive boolean parsing works")
    
    def test_scientific_notation(self):
        """Test parsing of numbers in scientific notation."""
        text = '''
        "large": 1.5e10
        "small": 2.3e-5
        '''
        keys = ["large", "small"]
        value_types = ["float", "float"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["large"] == 1.5e10
        assert result["small"] == 2.3e-5
        print("✓ Scientific notation parsing works")
    
    def test_mixed_quotes_and_no_quotes(self):
        """Test extraction with mixed quoting styles."""
        text = '''
        name: "Alice"
        'age': 30
        "city": London
        status: active
        '''
        keys = ["name", "age", "city", "status"]
        value_types = ["str", "int", "str", "str"]
        
        result = extract_info_from_text(text, keys, value_types)
        
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["city"] == "London" or result["city"] == ""
        print("✓ Mixed quoting styles work")
    
    def test_pydantic_schema_integration(self):
        """Test extraction that would work with Pydantic model validation."""
        from pydantic import BaseModel, Field
        
        class SentimentAnalysis(BaseModel):
            text: str = Field(description="The analyzed text")
            sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
            score: int = Field(description="Sentiment score from 0 to 10")
        
        # Simulate LLM output that's partially structured
        text = '''
        Based on the analysis:
        "text": "I love this product!"
        "sentiment": "positive"
        "score": 9
        '''
        
        keys = list(SentimentAnalysis.model_fields.keys())
        value_types = [
            field.annotation.__name__
            for field in SentimentAnalysis.model_fields.values()
        ]
        
        result = extract_info_from_text(text, keys, value_types)
        
        # Should be able to create a Pydantic model from the result
        model = SentimentAnalysis(**result)
        assert model.sentiment == "positive"
        assert model.score == 9
        print("✓ Pydantic schema integration works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
