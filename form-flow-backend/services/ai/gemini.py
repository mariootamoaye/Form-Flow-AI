"""
Gemini AI Service (LangChain Enhanced)

Provides integration with Google's Gemini AI for conversational form flow generation
using LangChain for structured reasoning and output parsing.

Uses:
    - langchain-google-genai for LLM integration
    - google-genai SDK (wrapped by langchain)
    - LangChain output parsers for reliable JSON extraction

Usage:
    from services.ai.gemini import GeminiService, SmartFormFillerChain
    
    service = GeminiService()
    result = service.generate_conversational_flow(
        extracted_fields={"name": "John"},
        form_schema=[{"fields": [...]}]
    )
    
    # Magic Fill
    filler = SmartFormFillerChain(service.llm)
    filled = await filler.fill(user_profile, form_schema)
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dotenv import load_dotenv  # <--- Add this

load_dotenv()

from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from utils.logging import get_logger, log_api_call
from utils.exceptions import AIServiceError

logger = get_logger(__name__)


# --- Pydantic Models for Structured Output ---

class FormFieldSuggestion(BaseModel):
    """Suggestion for a single form field."""
    field_name: str = Field(description="The exact name of the form field")
    value: str = Field(description="The suggested value for this field")
    confidence: float = Field(description="Confidence score between 0 and 1")
    source: str = Field(description="Where this value came from (profile, inferred, default)")


class MagicFillResult(BaseModel):
    """Result of Magic Fill operation."""
    filled_fields: List[FormFieldSuggestion] = Field(description="List of filled field suggestions")
    unfilled_fields: List[str] = Field(description="Field names that couldn't be filled")
    summary: str = Field(description="Brief summary of what was filled")


class FieldFill(BaseModel):
    """Single-field fill output."""
    value: Optional[str] = ""
    confidence: float = 0.0


class ConversationalFlow(BaseModel):
    """Structure for conversational flow."""
    acknowledgment: str = Field(description="Message acknowledging captured data")
    questions: List[Dict[str, Any]] = Field(description="List of questions to ask")
    completion_message: str = Field(description="Message when all fields are collected")


# --- LangChain Enhanced Service ---

class GeminiService:
    """
    Service for interacting with Google Gemini AI via LangChain.
    
    Generates conversational flows for form completion based on
    extracted user data and remaining form fields.
    
    Attributes:
        llm: ChatGoogleGenerativeAI instance
        model: Model name (default: gemini-2.0-flash)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini service with LangChain.
        
        Args:
            api_key: Google or OpenRouter API key. 
                     Checks GEMMA_API_KEY, then GOOGLE_API_KEY.
            model: Model to use.
            
        Raises:
            ValueError: If no API key is provided or found.
        """
        # Prioritize GEMMA_API_KEY as requested
        self.api_key = api_key or os.getenv('GEMMA_API_KEY') or os.getenv('OPENROUTER_API_KEY') or os.getenv('GOOGLE_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError("No API key found. Set GEMMA_API_KEY, OPENROUTER_API_KEY, or GOOGLE_API_KEY.")
        
        # Check if it's an OpenRouter key (Gemma)
        if self.api_key and self.api_key.startswith("sk-or-"):
            logger.info("Detected OpenRouter API key. Switching to OpenRouter provider.")
            # Default to Gemma 2 9B if using OpenRouter and default model was passed
            if self.model in ("gemini-2.0-flash", "gemma-2-9b-it"):
                self.model = "google/gemma-2-9b-it" # Use free tier
            
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.3
            )
        else:
            # Standard Google Gemini
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.3,  # Lower temp for more consistent outputs
                convert_system_message_to_human=True
            )
        
        logger.info(f"GeminiService initialized, model: {self.model}")

    def generate_conversational_flow(
        self,
        extracted_fields: Dict[str, str],
        form_schema: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a conversational flow for collecting remaining form fields.
        
        Uses LangChain for structured output parsing.
        
        Args:
            extracted_fields: Dictionary of already captured {field_name: value}
            form_schema: Form schema from parser (list of forms with fields)
            
        Returns:
            dict: Result with conversational_flow, remaining_fields, and success flag
        """
        try:
            logger.info(f"Generating flow for {len(extracted_fields)} extracted fields")
            
            remaining_fields = self._get_remaining_fields(extracted_fields, form_schema)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI assistant that creates conversational flows for form completion.
                Return ONLY valid JSON matching this structure:
                {
                    "acknowledgment": "Brief message",
                    "questions": [{"field_name": "...", "question": "...", "field_type": "...", "required": true}],
                    "completion_message": "Done message"
                }"""),
                ("human", """EXTRACTED DATA: {extracted}
                
REMAINING FIELDS: {remaining}

Create a friendly conversational flow to collect the remaining fields.""")
            ])
            
            # Create chain with JSON parser
            parser = JsonOutputParser(pydantic_object=ConversationalFlow)
            chain = prompt | self.llm | parser
            
            # Invoke
            flow_data = chain.invoke({
                "extracted": json.dumps(extracted_fields, indent=2),
                "remaining": json.dumps([{
                    'name': f.get('name'),
                    'type': f.get('type'),
                    'label': f.get('label'),
                    'required': f.get('required', False)
                } for f in remaining_fields], indent=2)
            })
            
            log_api_call("Gemini-LangChain", "generate_content", success=True)
            logger.info(f"Generated flow with {len(remaining_fields)} remaining fields")
            
            return {
                "success": True,
                "conversational_flow": flow_data,
                "remaining_fields": remaining_fields
            }
            
        except Exception as e:
            logger.error(f"Gemini/LangChain API error: {e}")
            log_api_call("Gemini-LangChain", "generate_content", success=False, error=str(e))
            
            return {
                "success": False,
                "error": str(e),
                "conversational_flow": self._get_fallback_flow()
            }

    def _get_fallback_flow(self) -> Dict[str, Any]:
        """Get a fallback conversational flow when parsing fails."""
        return {
            "acknowledgment": "Thank you for the information provided.",
            "questions": [],
            "completion_message": "All required information has been collected."
        }

    def _get_remaining_fields(
        self,
        extracted_fields: Dict[str, str],
        form_schema: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get list of fields that still need to be collected."""
        remaining = []
        
        for form in form_schema:
            for field in form.get('fields', []):
                field_name = field.get('name')
                
                if (field_name not in extracted_fields and
                    not field.get('hidden', False) and
                    field.get('type') != 'submit'):
                    
                    remaining.append({
                        'name': field_name,
                        'type': field.get('type'),
                        'label': field.get('label'),
                        'required': field.get('required', False)
                    })
        
        return remaining


class SmartFormFillerChain:
    """
    LangChain-powered intelligent form filler.
    
    Analyzes user profile against form schema and fills as many fields
    as possible in a single LLM call ("Magic Fill").
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=MagicFillResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent form-filling assistant.
Your task is to match user profile data to form fields intelligently.

RULES:
1. Match fields by semantic meaning, not just exact name match.
2. For "Years of Experience", infer from job history if available.
3. For "Skills", extract from resume/projects.
4. For addresses, infer city/state/country from full address.
5. For names, split "Full Name" into first/last if needed.
6. Set confidence based on how direct the match is (1.0 = exact, 0.5 = inferred).
7. Do NOT fill password, secrets, or payment fields.

Return ONLY valid JSON matching this schema:
{format_instructions}"""),
            ("human", """USER PROFILE:
{profile}

FORM SCHEMA (fields to fill):
{schema}

Fill as many fields as possible from the user's profile. Be intelligent about mapping data.""")
        ])
        # Lightweight single-field prompt for parallel inference
        self.single_field_prompt = ChatPromptTemplate.from_messages([
            ("system", """You fill ONE form field using a provided user profile.
Return strict JSON: {"value": "...", "confidence": 0-1}. If unsure, leave value empty and confidence 0."""),
            ("human", """PROFILE:
{profile}

FIELD:
Name: {field_name}
Label: {field_label}
Type: {field_type}
Required: {required}
Options: {options}

Provide the best value for this field from the profile or inferred context.""")
        ])
        self.single_field_parser = JsonOutputParser(pydantic_object=FieldFill)
    
    async def fill(
        self, 
        user_profile: Dict[str, Any], 
        form_schema: List[Dict[str, Any]],
        min_confidence: float = 0.5,
        progress_cb: Optional[Callable[[str, Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Perform "Magic Fill" - fill entire form from user profile.
        
        Args:
            user_profile: User's profile data (name, email, resume, etc.)
            form_schema: Form schema with all fields
            min_confidence: Minimum confidence to include a suggestion
            progress_cb: Optional async callback invoked per field filled
            
        Returns:
            dict: {
                "filled": {field_name: value, ...},
                "unfilled": [field_names],
                "summary": "Filled X of Y fields"
            }
        """
        try:
            # Extract fillable fields from schema
            fillable_fields = []
            for form in form_schema:
                for field in form.get('fields', []):
                    if not field.get('hidden') and field.get('type') not in ['submit', 'button', 'hidden']:
                        fillable_fields.append({
                            'name': field.get('name'),
                            'type': field.get('type', 'text'),
                            'label': field.get('label', ''),
                            'required': field.get('required', False),
                            'options': field.get('options', [])[:5]  # Limit options for context
                        })

            # STEP 1: Instant fill from obvious profile fields (non-blocking)
            instant_filled = self._instant_fill_profile(user_profile, fillable_fields)
            filled: Dict[str, Any] = {**instant_filled}
            if progress_cb:
                for fname, val in instant_filled.items():
                    await progress_cb(fname, val)

            remaining_fields = [f for f in fillable_fields if f.get("name") not in filled]

            # STEP 2: Fire AI calls in parallel for remaining fields
            async def infer_single(field: Dict[str, Any]):
                try:
                    payload = {
                        "profile": json.dumps(user_profile, default=str),
                        "field_name": field.get("name"),
                        "field_label": field.get("label") or field.get("name"),
                        "field_type": field.get("type", "text"),
                        "required": field.get("required", False),
                        "options": json.dumps(field.get("options", []), default=str)
                    }
                    chain = self.single_field_prompt | self.llm | self.single_field_parser
                    result: FieldFill = await chain.ainvoke(payload)
                    return field.get("name"), result.value, result.confidence
                except Exception as e:
                    logger.warning(f"Single-field inference failed for {field.get('name')}: {e}")
                    return field.get("name"), None, 0.0

            tasks = []
            semaphore = asyncio.Semaphore(5)  # prevent LLM overload

            async def run_with_semaphore(field):
                async with semaphore:
                    return await infer_single(field)

            for field in remaining_fields:
                tasks.append(asyncio.create_task(run_with_semaphore(field)))

            for task in asyncio.as_completed(tasks):
                fname, val, conf = await task
                if val and conf >= min_confidence:
                    filled[fname] = val
                    if progress_cb:
                        await progress_cb(fname, val)

            unfilled = [f.get("name") for f in fillable_fields if f.get("name") not in filled]

            logger.info(f"Magic Fill: {len(filled)} filled, {len(unfilled)} unfilled")

            return {
                "success": True,
                "filled": filled,
                "unfilled": unfilled,
                "summary": f"Filled {len(filled)} of {len(fillable_fields)} fields automatically"
            }
            
        except Exception as e:
            logger.error(f"SmartFormFillerChain error: {e}")
            return {
                "success": False,
                "error": str(e),
                "filled": {},
                "unfilled": [],
                "summary": "Magic fill failed, please fill manually"
            }

    def _instant_fill_profile(self, profile: Dict[str, Any], fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fast, deterministic profile mapping for common fields.
        Mirrors frontend instantFill utility to ensure DOM injection happens immediately.
        """
        if not profile:
            return {}

        profile_data = {
            "first_name": profile.get("first_name") or profile.get("firstName") or "",
            "last_name": profile.get("last_name") or profile.get("lastName") or "",
            "full_name": profile.get("fullname") or profile.get("full_name") or f"{profile.get('first_name','')} {profile.get('last_name','')}".strip(),
            "email": profile.get("email") or "",
            "phone": profile.get("mobile") or profile.get("phone") or profile.get("contact") or "",
            "city": profile.get("city") or "",
            "state": profile.get("state") or "",
            "country": profile.get("country") or "",
            "zip": profile.get("zip") or profile.get("zipcode") or profile.get("pincode") or "",
            "address": profile.get("address") or profile.get("street") or "",
        }

        patterns = {
            "first_name": ["first_name", "firstname", "fname", "given_name", "givenname", "name_first"],
            "last_name": ["last_name", "lastname", "lname", "surname", "family_name", "familyname", "name_last"],
            "full_name": ["full_name", "fullname", "name", "your_name", "yourname", "applicant_name", "applicantname", "complete_name"],
            "email": ["email", "e-mail", "mail", "emailaddress", "email_address", "e_mail", "emailid", "email_id", "user_email"],
            "phone": ["phone", "mobile", "cell", "telephone", "tel", "phonenumber", "phone_number", "mobile_number", "contact_number", "cellphone", "mobilenumber", "contact"],
            "city": ["city", "town", "municipality", "locality", "city_name"],
            "state": ["state", "province", "region", "state_province", "stateprovince"],
            "country": ["country", "nation", "country_name"],
            "zip": ["zip", "zipcode", "zip_code", "postal", "postalcode", "postal_code", "pincode", "pin_code", "pin"],
            "address": ["address", "street", "street_address", "address_line", "addressline", "address1", "address_1", "location"],
        }

        def normalize(name: str) -> str:
            return (
                (name or "")
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
                .replace(".", "_")
            )

        filled = {}
        for field in fields:
            if not field.get("name") or field.get("type") in ["submit", "button", "hidden", "file", "image"]:
                continue
            fname = normalize(field.get("name"))
            flabel = normalize(field.get("label") or "")
            for key, pat_list in patterns.items():
                norm_patterns = [normalize(p) for p in pat_list]
                if fname in norm_patterns or flabel in norm_patterns or any(p in fname for p in norm_patterns):
                    if profile_data.get(key):
                        filled[field.get("name")] = profile_data[key]
                    break

        return filled


# --- Singleton ---
_service_instance: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Get singleton GeminiService instance."""
    global _service_instance
    print("Getting Gemini Service Instance")
    api_key = os.getenv('GEMMA_API_KEY') or os.getenv('OPENROUTER_API_KEY') or os.getenv('GOOGLE_API_KEY')
    print(f"Using API Key: {api_key[:8]}...")  # Print first 8 chars for debugging
    if _service_instance is None:
        try:
            _service_instance = GeminiService()
        except ValueError as e:
            logger.warning(f"Could not initialize GeminiService: {e}")
            return None
    return _service_instance
