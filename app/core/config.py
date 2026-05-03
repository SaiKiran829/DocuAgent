import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock

load_dotenv()

class Config:
    ANTHROPIC_KEY: str = os.getenv("ANTHROPIC_KEY", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-2")
    
    # model settings
    # BEDROCK_MODEL_ID: str = "global.twelvelabs.pegasus-1-2-v1:0"
    BEDROCK_MODEL_ID: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"
    TEMPERATURE: float = 0.0

    @classmethod
    def validate_bedrock(cls):
        if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS credentials not set in .env file")

    @classmethod
    def validate_anthropic(cls):
        if not cls.ANTHROPIC_KEY:
            raise ValueError("ANTHROPIC_KEY not set in .env file")

class LLM_Client:
    _instance = None
    _provider = "anthropic"  # switch between "bedrock" and "anthropic"
    
    @classmethod
    def get(cls) -> ChatBedrock | ChatAnthropic:
        print(f"[llm] provider is: {cls._provider}")  # add this line

        if cls._instance is not None:
            return cls._instance

        if cls._provider == "bedrock":
            cls._instance = cls._create_bedrock()
        else:
            cls._instance = cls._create_anthropic()

        return cls._instance
    
    @classmethod
    def _create_bedrock(cls):
        Config.validate_bedrock()
        print("[llm] using AWS Bedrock")
        import boto3
        session = boto3.Session(
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION
        )
        return ChatBedrock(
            model_id=Config.BEDROCK_MODEL_ID,
            client=session.client("bedrock-runtime"),
            model_kwargs={"temperature": Config.TEMPERATURE}
        )

    @classmethod
    def _create_anthropic(cls):
        Config.validate_anthropic()
        print("[llm] using Anthropic directly")
        return ChatAnthropic(
            api_key=Config.ANTHROPIC_KEY,
            model=Config.ANTHROPIC_MODEL,
            temperature=Config.TEMPERATURE
        )

    @classmethod
    def switch(cls, provider: str):
        """Switch between 'bedrock' and 'anthropic' at runtime."""
        cls._provider = provider
        cls._instance = None  # reset singleton so new provider is used
        print(f"[llm] switched to {provider}")
    