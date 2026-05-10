import time
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.messages import BaseMessage
import tiktoken

class TokenRateLimiter:
    """
    Manages token rate limiting to prevent exceeding API limits.
    Falls back to summaries when approaching the limit.
    """
    
    def __init__(self, max_tokens_per_minute: int = 12000, buffer_percent: float = 0.85):
        """
        Args:
            max_tokens_per_minute: Max TPM allowed by API (default: 12000 for Groq)
            buffer_percent: Use only 85% of limit to have safety margin
        """
        self.max_tokens_per_minute = max_tokens_per_minute
        self.safe_limit = int(max_tokens_per_minute * buffer_percent)
        self.token_usage = []  # Track token usage timestamps
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Compatible encoding
        self.logger = None
    
    def set_logger(self, logger):
        """Set logger instance"""
        self.logger = logger
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: list) -> int:
        """Count tokens in LangChain messages"""
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total += self.count_tokens(msg.content)
            else:
                total += self.count_tokens(str(msg))
        return total
    
    def get_tokens_used_this_minute(self) -> int:
        """Get tokens used in current minute window"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Remove old entries
        self.token_usage = [(ts, tokens) for ts, tokens in self.token_usage 
                       if ts > one_minute_ago]
        
        return sum(tokens for _, tokens in self.token_usage)
    
    def can_proceed(self, estimated_tokens: int) -> tuple[bool, dict]:
        """
        Check if we can proceed with the API call.
        Returns: (can_proceed: bool, status: dict)
        """
        current_usage = self.get_tokens_used_this_minute()
        projected_usage = current_usage + estimated_tokens
        
        status = {
            "current_usage": current_usage,
            "estimated_tokens": estimated_tokens,
            "projected_usage": projected_usage,
            "safe_limit": self.safe_limit,
            "can_proceed": projected_usage <= self.safe_limit,
            "usage_percent": (projected_usage / self.safe_limit) * 100
        }
        
        if self.logger:
            self.logger.info("Token check", **status)
        
        return status["can_proceed"], status
    
    def record_tokens(self, tokens: int):
        """Record token usage"""
        self.token_usage.append((datetime.now(), tokens))
    
    def wait_if_needed(self, estimated_tokens: int) -> bool:
        """
        Wait if necessary before proceeding.
        Returns: True if waited, False if proceeded immediately
        """
        current_usage = self.get_tokens_used_this_minute()
        projected = current_usage + estimated_tokens
        
        if projected > self.safe_limit:
            # Calculate wait time
            oldest_usage = min(ts for ts, _ in self.token_usage) if self.token_usage else datetime.now()
            one_minute_ago = datetime.now() - timedelta(minutes=1)
            
            if oldest_usage > one_minute_ago:
                wait_time = (oldest_usage - one_minute_ago).total_seconds() + 1
                if self.logger:
                    self.logger.warning(f"Rate limit approaching. Waiting {wait_time:.2f}s", 
                                      current_usage=current_usage, estimated=estimated_tokens)
                time.sleep(wait_time)
                return True
        
        return False