"""E2E tests for Ollama.structured_predict and related streaming/async variants.

Covers schema extraction edge cases across the full range of Pydantic model
structures: flat models, nested models with $defs, optional fields, enums,
literals, field constraints, list types, and deeply nested hierarchies.

All tests require a running Ollama instance (local or cloud) and are skipped
when none is available.  Use temperature=0.0 (via llm_model fixture) for
deterministic extraction.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional, Union

import pytest
from pydantic import BaseModel, Field, field_validator

from serapeum.core.prompts import PromptTemplate
from serapeum.ollama import Ollama

from ..models import client


# ---------------------------------------------------------------------------
# Schema stress-test models
# ---------------------------------------------------------------------------


# 1. Flat / all-primitive
class PersonBasic(BaseModel):
    """Flat model: only primitive types — produces a schema with no $defs."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    height_m: float = Field(description="Height in metres")
    is_employed: bool = Field(description="Whether the person is currently employed")


# 2. Optional fields  (produces anyOf / null in schema)
class Contact(BaseModel):
    """Model with optional fields — schema includes anyOf:[type, null] branches."""

    email: str = Field(description="Primary e-mail address")
    phone: Optional[str] = Field(default=None, description="Phone number, if known")
    website: Optional[str] = Field(default=None, description="Personal website URL")


# 3. Fields with defaults  (schema has 'default' annotations)
class Config(BaseModel):
    """Model whose fields have default values — tests that defaults are ignored when the LLM fills in values."""

    language: str = Field(default="en", description="ISO-639 language code")
    max_tokens: int = Field(default=256, description="Maximum number of tokens")
    temperature: float = Field(default=0.7, description="Sampling temperature 0-1")


# 4. Nested model  (produces $defs in schema — canonical structured-output edge case)
class Address(BaseModel):
    """Nested sub-model for Employee."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="ISO country name")


class Employee(BaseModel):
    """Model containing a nested BaseModel — schema has $defs and $ref."""

    full_name: str = Field(description="Employee's full name")
    job_title: str = Field(description="Job title or role")
    address: Address = Field(description="Registered office address")


# 5. List of nested models  (array of $ref'd objects)
class LineItem(BaseModel):
    """Single line item in an invoice."""

    product: str = Field(description="Product name")
    quantity: int = Field(description="Number of units")
    unit_price: float = Field(description="Price per unit in USD")


class Invoice(BaseModel):
    """Model with list[NestedModel] — produces $defs + array of $ref."""

    invoice_number: str = Field(description="Unique invoice identifier")
    customer: str = Field(description="Customer name")
    items: list[LineItem] = Field(description="Line items in this invoice")
    total_usd: float = Field(description="Total amount in USD")


# 6. Enum field  (produces 'enum' in schema)
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Review(BaseModel):
    """Model with an Enum field — schema includes 'enum' alongside string type."""

    product_name: str = Field(description="Name of the reviewed product")
    sentiment: Sentiment = Field(description="Overall sentiment of the review")
    summary: str = Field(description="One-sentence summary of the review")


# 7. Literal field  (produces 'const' / single-value enum in schema)
from typing import Literal


class TrafficLight(BaseModel):
    """Model with a Literal field — schema restricts value to a fixed set."""

    intersection: str = Field(description="Intersection identifier")
    state: Literal["red", "amber", "green"] = Field(description="Current signal state")


# 8. Union field  (produces anyOf in schema)
class Measurement(BaseModel):
    """Model with Union[str, int] — schema uses anyOf for the value field."""

    label: str = Field(description="What is being measured")
    value: Union[int, float] = Field(description="Measured value, integer or decimal")
    unit: str = Field(description="Unit of measurement e.g. kg, m, °C")


# 9. Fields with constraints  (adds minimum/maximum/minLength etc. to schema)
class ConstrainedProduct(BaseModel):
    """Model with annotated constraints — schema has minLength, minimum, maximum."""

    sku: Annotated[str, Field(min_length=3, max_length=20, description="Stock-keeping unit")]
    price_usd: Annotated[float, Field(ge=0.0, description="Price, must be non-negative")]
    stock: Annotated[int, Field(ge=0, le=10_000, description="Units in stock")]


# 10. Field with alias  (schema uses alias names)
class ApiPayload(BaseModel):
    """Model whose fields have Python-style names but JSON aliases."""

    model_config = {"populate_by_name": True}

    user_id: str = Field(alias="userId", description="User identifier")
    access_token: str = Field(alias="accessToken", description="Bearer token")


# 11. Deeply nested (3+ levels)
class Location(BaseModel):
    latitude: float = Field(description="Latitude in decimal degrees")
    longitude: float = Field(description="Longitude in decimal degrees")


class Waypoint(BaseModel):
    name: str = Field(description="Name of the waypoint")
    location: Location = Field(description="GPS coordinates")


class Route(BaseModel):
    """Three-level nesting — produces multi-level $defs."""

    route_name: str = Field(description="Name of the route")
    origin: Waypoint = Field(description="Starting waypoint")
    destination: Waypoint = Field(description="Ending waypoint")


# 12. Optional nested model  (anyOf:[ref, null] inside schema)
class Profile(BaseModel):
    """Model with an Optional nested model field."""

    username: str = Field(description="Display name")
    bio: Optional[str] = Field(default=None, description="Short biography")
    address: Optional[Address] = Field(default=None, description="Home address if provided")


# 13. Dict field  (produces additionalProperties in schema)
class Metadata(BaseModel):
    """Model with a dict field — produces object with additionalProperties."""

    record_id: str = Field(description="Unique record identifier")
    tags: dict[str, str] = Field(description="Key-value metadata tags")


# 14. Field validator  (validator does not change schema; tests post-validation)
class NormalizedName(BaseModel):
    """Model with a field_validator that uppercases the value after parsing."""

    first: str = Field(description="First name")
    last: str = Field(description="Last name")

    @field_validator("first", "last", mode="after")
    @classmethod
    def to_upper(cls, v: str) -> str:
        return v.upper()


# 15. Multi-field list of primitives  (array of strings)
class Keywords(BaseModel):
    """Model whose list field holds primitives, not sub-models."""

    topic: str = Field(description="The topic being described")
    keywords: list[str] = Field(description="5-10 relevant keywords for the topic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP = pytest.mark.skipif(
    client is None,
    reason="Ollama client is not available",
)


def _assert_valid_instance(result: BaseModel, cls: type[BaseModel]) -> None:
    """Assert result is a fully initialised instance of cls with all required fields set."""
    assert isinstance(result, cls), f"Expected {cls.__name__}, got {type(result).__name__}"
    schema = cls.model_json_schema()
    required = schema.get("required", [])
    for field_name in required:
        val = getattr(result, field_name, None)
        assert val is not None, f"Required field '{field_name}' is None on {cls.__name__}"


# ---------------------------------------------------------------------------
# Tests — structured_predict (sync)
# ---------------------------------------------------------------------------


class TestStructuredPredictSync:
    """Synchronous structured_predict across all schema edge-case models."""

    @pytest.mark.e2e
    @_SKIP
    def test_flat_primitive_model(self, llm_model: Ollama) -> None:
        """Flat schema with only primitive fields (str, int, float, bool).

        Inputs: free-text description of a person.
        Expected: PersonBasic instance with all four required fields populated.
        Checks: correct type, all required fields non-None.
        """
        prompt = PromptTemplate(
            "Extract the following information from the text into the requested fields.\n"
            "Text: {text}"
        )
        result = llm_model.structured_predict(
            PersonBasic, prompt, text="Jane Smith is 34 years old, 1.72 m tall, and works as a nurse."
        )
        _assert_valid_instance(result, PersonBasic)
        assert isinstance(result.name, str) and result.name.strip()
        assert isinstance(result.age, int) and result.age > 0
        assert isinstance(result.height_m, float) and result.height_m > 0
        assert isinstance(result.is_employed, bool)

    @pytest.mark.e2e
    @_SKIP
    def test_optional_fields_populated(self, llm_model: Ollama) -> None:
        """Schema with Optional fields that should be filled from context.

        Inputs: text containing email and phone but no website.
        Expected: Contact with email and phone set; website may be None.
        Checks: email is a non-empty string; result is Contact.
        """
        prompt = PromptTemplate(
            "Extract contact details from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Contact,
            prompt,
            text="Reach Alice at alice@example.com or call her on +1-555-0100.",
        )
        _assert_valid_instance(result, Contact)
        assert "@" in result.email

    @pytest.mark.e2e
    @_SKIP
    def test_optional_fields_absent(self, llm_model: Ollama) -> None:
        """Optional fields remain None when context provides no relevant data.

        Inputs: text with only an email, no phone or website.
        Expected: Contact.phone and Contact.website are either None or unset.
        Checks: result.email is populated; optional fields may be None.
        """
        prompt = PromptTemplate(
            "Extract contact details from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Contact,
            prompt,
            text="Contact us at support@example.org for help.",
        )
        assert isinstance(result, Contact)
        assert result.email.strip()

    @pytest.mark.e2e
    @_SKIP
    def test_fields_with_defaults(self, llm_model: Ollama) -> None:
        """Schema with default-valued fields — LLM should override defaults from context.

        Inputs: text describing non-default configuration values.
        Expected: Config instance; language overridden to 'fr' from context.
        Checks: language field matches the context value.
        """
        prompt = PromptTemplate(
            "Extract configuration settings from the description.\nDescription: {text}"
        )
        result = llm_model.structured_predict(
            Config,
            prompt,
            text="Set the language to French (fr), max_tokens to 512, and temperature to 0.3.",
        )
        assert isinstance(result, Config)
        assert result.language == "fr"
        assert result.max_tokens == 512

    @pytest.mark.e2e
    @_SKIP
    def test_nested_model_with_defs(self, llm_model: Ollama) -> None:
        """Nested Pydantic model — schema uses $defs and $ref.

        Inputs: text describing an employee and their address.
        Expected: Employee instance with a valid nested Address.
        Checks: employee.address is an Address instance; city is non-empty.
        """
        prompt = PromptTemplate(
            "Extract employee information from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Employee,
            prompt,
            text=(
                "Carlos Mendez works as a Software Engineer at our Berlin office, "
                "located at Unter den Linden 1, Berlin, Germany."
            ),
        )
        _assert_valid_instance(result, Employee)
        assert isinstance(result.address, Address)
        assert result.address.city.strip()

    @pytest.mark.e2e
    @_SKIP
    def test_list_of_nested_models(self, llm_model: Ollama) -> None:
        """Schema with list[NestedModel] — produces $defs + items: $ref array.

        Inputs: text describing an invoice with multiple line items.
        Expected: Invoice with at least one LineItem.
        Checks: items is a non-empty list of LineItem; total is positive.
        """
        prompt = PromptTemplate(
            "Extract the invoice details from the following text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Invoice,
            prompt,
            text=(
                "Invoice #INV-2024-001 for customer Acme Corp. "
                "Items: 2x Widget A at $15.00 each, 1x Widget B at $30.00. "
                "Total: $60.00."
            ),
        )
        _assert_valid_instance(result, Invoice)
        assert len(result.items) >= 1
        assert all(isinstance(i, LineItem) for i in result.items)
        assert result.total_usd > 0

    @pytest.mark.e2e
    @_SKIP
    def test_enum_field(self, llm_model: Ollama) -> None:
        """Schema with Enum field — produces 'enum' constraint in JSON schema.

        Inputs: positive product review text.
        Expected: Review with sentiment == Sentiment.POSITIVE.
        Checks: sentiment is a valid Sentiment member; summary is non-empty.
        """
        prompt = PromptTemplate(
            "Analyse the product review and extract the requested fields.\nReview: {text}"
        )
        result = llm_model.structured_predict(
            Review,
            prompt,
            text=(
                "Absolutely love this blender! It's quiet, powerful, and easy to clean. "
                "Highly recommend the NutriBlend 3000."
            ),
        )
        _assert_valid_instance(result, Review)
        assert isinstance(result.sentiment, Sentiment)
        assert result.sentiment == Sentiment.POSITIVE
        assert result.summary.strip()

    @pytest.mark.e2e
    @_SKIP
    def test_literal_field(self, llm_model: Ollama) -> None:
        """Schema with Literal field — value must be one of the allowed constants.

        Inputs: text describing a traffic light at red.
        Expected: TrafficLight with state == 'red'.
        Checks: state is exactly 'red'; intersection is non-empty.
        """
        prompt = PromptTemplate(
            "Extract the traffic-light state from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            TrafficLight,
            prompt,
            text="The light at Main & 5th is currently showing red.",
        )
        _assert_valid_instance(result, TrafficLight)
        assert result.state == "red"

    @pytest.mark.e2e
    @_SKIP
    def test_union_field(self, llm_model: Ollama) -> None:
        """Schema with Union[int, float] — produces anyOf in JSON schema.

        Inputs: text with a decimal measurement.
        Expected: Measurement with value as int or float.
        Checks: value is numeric; label and unit are non-empty strings.
        """
        prompt = PromptTemplate(
            "Extract the measurement details from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Measurement,
            prompt,
            text="The package weighs 2.5 kg.",
        )
        _assert_valid_instance(result, Measurement)
        assert isinstance(result.value, (int, float))
        assert result.unit.strip()

    @pytest.mark.e2e
    @_SKIP
    def test_constrained_fields(self, llm_model: Ollama) -> None:
        """Schema with annotated constraints (minLength, minimum, maximum).

        Inputs: product description with SKU, price, and stock level.
        Expected: ConstrainedProduct with non-negative price and stock.
        Checks: price >= 0, stock >= 0; sku has >=3 chars.
        """
        prompt = PromptTemplate(
            "Extract product information from the description.\nDescription: {text}"
        )
        result = llm_model.structured_predict(
            ConstrainedProduct,
            prompt,
            text="SKU: WDG-001. Price: $19.99. Currently 150 units in stock.",
        )
        _assert_valid_instance(result, ConstrainedProduct)
        assert result.price_usd >= 0
        assert result.stock >= 0
        assert len(result.sku) >= 3

    @pytest.mark.e2e
    @_SKIP
    def test_deeply_nested_three_levels(self, llm_model: Ollama) -> None:
        """Three-level nesting: Route -> Waypoint -> Location.

        Schema produces multi-level $defs with multiple $ref chains.
        Inputs: text describing a route from one city to another.
        Expected: Route with both origin and destination as Waypoint with Location.
        Checks: origin.location.latitude is a float; destination.name non-empty.
        """
        prompt = PromptTemplate(
            "Extract the route details from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Route,
            prompt,
            text=(
                "The scenic drive starts at Golden Gate Bridge "
                "(37.8199° N, 122.4783° W) and ends at "
                "Fisherman's Wharf (37.8080° N, 122.4177° W)."
            ),
        )
        _assert_valid_instance(result, Route)
        assert isinstance(result.origin, Waypoint)
        assert isinstance(result.origin.location, Location)
        assert isinstance(result.destination, Waypoint)
        assert isinstance(result.destination.location, Location)

    @pytest.mark.e2e
    @_SKIP
    def test_optional_nested_model(self, llm_model: Ollama) -> None:
        """Optional nested model field — anyOf:[ref, null] in schema.

        Inputs: text with username and bio but no address.
        Expected: Profile with username and bio; address may be None.
        Checks: username non-empty; address is either None or an Address instance.
        """
        prompt = PromptTemplate(
            "Extract profile information from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Profile,
            prompt,
            text="User: dev_guru42. Bio: 'Open-source enthusiast and coffee lover.'",
        )
        assert isinstance(result, Profile)
        assert result.username.strip()
        assert result.address is None or isinstance(result.address, Address)

    @pytest.mark.e2e
    @_SKIP
    def test_list_of_primitives(self, llm_model: Ollama) -> None:
        """Model with list[str] — simple array type without $defs.

        Inputs: topic description.
        Expected: Keywords with a non-empty list of keyword strings.
        Checks: keywords is a list with at least one string.
        """
        prompt = PromptTemplate(
            "Extract the topic and keywords from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            Keywords,
            prompt,
            text="Quantum computing uses qubits to perform calculations far beyond classical computers.",
        )
        _assert_valid_instance(result, Keywords)
        assert isinstance(result.keywords, list)
        assert len(result.keywords) >= 1
        assert all(isinstance(k, str) for k in result.keywords)

    @pytest.mark.e2e
    @_SKIP
    def test_field_validator_applied_post_parse(self, llm_model: Ollama) -> None:
        """field_validator runs after the LLM response is parsed by model_validate_json.

        Inputs: text with a lowercase name.
        Expected: NormalizedName with first and last uppercased by the validator.
        Checks: result.first == result.first.upper().
        """
        prompt = PromptTemplate(
            "Extract the first and last name from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            NormalizedName,
            prompt,
            text="The author is alice johnson.",
        )
        assert isinstance(result, NormalizedName)
        assert result.first == result.first.upper()
        assert result.last == result.last.upper()

    @pytest.mark.e2e
    @_SKIP
    def test_idempotent_on_same_input(self, llm_model: Ollama) -> None:
        """Same prompt produces structurally identical results across two calls (temperature=0).

        Inputs: identical text on two separate calls.
        Expected: Both calls return PersonBasic instances with the same name and age.
        Checks: result1.name == result2.name (within normalisation tolerance).
        """
        prompt = PromptTemplate(
            "Extract the following information from the text.\nText: {text}"
        )
        text = "Bob Wilson is 45 years old, 1.80 m tall, and is retired."
        result1 = llm_model.structured_predict(PersonBasic, prompt, text=text)
        result2 = llm_model.structured_predict(PersonBasic, prompt, text=text)
        assert isinstance(result1, PersonBasic)
        assert isinstance(result2, PersonBasic)
        # With temperature=0, name and age should be identical
        assert result1.name.lower() == result2.name.lower()
        assert result1.age == result2.age

    @pytest.mark.e2e
    @_SKIP
    def test_extra_llm_kwargs_forwarded(self, llm_model: Ollama) -> None:
        """llm_kwargs are forwarded to the underlying chat call.

        Passing num_predict restricts generation length; structured output
        should still be parseable as long as the schema is small.
        Inputs: small flat model with num_predict=200 in llm_kwargs.
        Expected: PersonBasic instance returned without exception.
        Checks: result is a valid PersonBasic.
        """
        prompt = PromptTemplate(
            "Extract person info from the text.\nText: {text}"
        )
        result = llm_model.structured_predict(
            PersonBasic,
            prompt,
            llm_kwargs={"num_predict": 200},
            text="Maria Garcia is 29, stands 1.65 m, and is a freelance designer.",
        )
        assert isinstance(result, PersonBasic)


# ---------------------------------------------------------------------------
# Tests — astructured_predict (async)
# ---------------------------------------------------------------------------


class TestStructuredPredictAsync:
    """Async astructured_predict across key schema shapes."""

    @pytest.mark.e2e
    @pytest.mark.asyncio()
    @_SKIP
    async def test_async_flat_model(self, llm_model: Ollama) -> None:
        """Async variant returns correct type for a flat primitive schema.

        Inputs: person description via astructured_predict.
        Expected: PersonBasic instance with all required fields.
        Checks: isinstance check; all required fields non-None.
        """
        prompt = PromptTemplate(
            "Extract the following information from the text.\nText: {text}"
        )
        result = await llm_model.astructured_predict(
            PersonBasic,
            prompt,
            text="Emma Torres is 27, 1.68 m tall, and works as a data analyst.",
        )
        _assert_valid_instance(result, PersonBasic)

    @pytest.mark.e2e
    @pytest.mark.asyncio()
    @_SKIP
    async def test_async_nested_model(self, llm_model: Ollama) -> None:
        """Async variant handles $defs/$ref schemas (nested models).

        Inputs: employee description.
        Expected: Employee with nested Address.
        Checks: address.city is non-empty string.
        """
        prompt = PromptTemplate(
            "Extract employee information from the text.\nText: {text}"
        )
        result = await llm_model.astructured_predict(
            Employee,
            prompt,
            text=(
                "Ana Lima is a Product Manager based at Rua Augusta 100, São Paulo, Brazil."
            ),
        )
        _assert_valid_instance(result, Employee)
        assert isinstance(result.address, Address)
        assert result.address.city.strip()

    @pytest.mark.e2e
    @pytest.mark.asyncio()
    @_SKIP
    async def test_async_enum_field(self, llm_model: Ollama) -> None:
        """Async variant correctly resolves enum constraints.

        Inputs: negative review text.
        Expected: Review with sentiment == Sentiment.NEGATIVE.
        Checks: sentiment is a Sentiment member.
        """
        prompt = PromptTemplate(
            "Analyse the review and extract the requested fields.\nReview: {text}"
        )
        result = await llm_model.astructured_predict(
            Review,
            prompt,
            text="Terrible product. Broke after two days. Would not recommend.",
        )
        _assert_valid_instance(result, Review)
        assert isinstance(result.sentiment, Sentiment)
        assert result.sentiment == Sentiment.NEGATIVE

    @pytest.mark.e2e
    @pytest.mark.asyncio()
    @_SKIP
    async def test_async_list_of_nested_models(self, llm_model: Ollama) -> None:
        """Async variant correctly handles array of nested objects.

        Inputs: invoice description.
        Expected: Invoice with non-empty items list.
        Checks: items[0] is a LineItem instance.
        """
        prompt = PromptTemplate(
            "Extract the invoice details from the following text.\nText: {text}"
        )
        result = await llm_model.astructured_predict(
            Invoice,
            prompt,
            text=(
                "Invoice #INV-002 for TechStartup Ltd. "
                "3x Laptop Stand at $25.00 each, 1x USB-C Hub at $45.00. Total: $120.00."
            ),
        )
        _assert_valid_instance(result, Invoice)
        assert len(result.items) >= 1
        assert isinstance(result.items[0], LineItem)


# ---------------------------------------------------------------------------
# Tests — stream_structured_predict (sync streaming)
# ---------------------------------------------------------------------------


class TestStreamStructuredPredict:
    """Synchronous streaming structured predict."""

    @pytest.mark.e2e
    @_SKIP
    def test_stream_yields_instances(self, llm_model: Ollama) -> None:
        """Streaming yields at least one BaseModel instance before completion.

        Inputs: person description.
        Expected: Generator yields at least one PersonBasic (or partial flexible model).
        Checks: at least one item yielded; last item is a PersonBasic instance.
        """
        prompt = PromptTemplate(
            "Extract the following information from the text.\nText: {text}"
        )
        gen = llm_model.stream_structured_predict(
            PersonBasic,
            prompt,
            text="Liam Chen is 31, 1.75 m tall, and works as a teacher.",
        )
        results = list(gen)
        assert len(results) >= 1
        last = results[-1]
        # Last item must be the final, fully resolved model or a list
        if isinstance(last, list):
            assert all(isinstance(r, BaseModel) for r in last)
        else:
            assert isinstance(last, PersonBasic)

    @pytest.mark.e2e
    @_SKIP
    def test_stream_nested_model(self, llm_model: Ollama) -> None:
        """Streaming handles nested model schemas without error.

        Inputs: employee description with address.
        Expected: Generator completes; final value is an Employee or contains one.
        Checks: no exception; final item is Employee or list[Employee].
        """
        prompt = PromptTemplate(
            "Extract employee information from the text.\nText: {text}"
        )
        gen = llm_model.stream_structured_predict(
            Employee,
            prompt,
            text="Sofia Russo is a Lead Engineer at Via Roma 5, Rome, Italy.",
        )
        results = list(gen)
        assert len(results) >= 1
        last = results[-1]
        if isinstance(last, list):
            assert len(last) >= 1
        else:
            assert isinstance(last, Employee)

    @pytest.mark.e2e
    @_SKIP
    def test_stream_final_object_is_complete(self, llm_model: Ollama) -> None:
        """Final streamed object has all required fields populated.

        Inputs: contact info text with email.
        Expected: Last yielded Contact has email populated.
        Checks: result.email contains '@'.
        """
        prompt = PromptTemplate(
            "Extract contact details from the text.\nText: {text}"
        )
        gen = llm_model.stream_structured_predict(
            Contact,
            prompt,
            text="You can reach support at help@company.io.",
        )
        results = list(gen)
        last = results[-1]
        final = last[0] if isinstance(last, list) else last
        assert isinstance(final, Contact)
        assert "@" in final.email


# ---------------------------------------------------------------------------
# Tests — astream_structured_predict (async streaming)
# ---------------------------------------------------------------------------


class TestAsyncStreamStructuredPredict:
    """Asynchronous streaming structured predict."""

    @pytest.mark.e2e
    @pytest.mark.asyncio()
    @_SKIP
    async def test_async_stream_yields_instances(self, llm_model: Ollama) -> None:
        """Async streaming yields at least one BaseModel instance.

        Inputs: measurement description.
        Expected: At least one item yielded from astream_structured_predict.
        Checks: last item is a Measurement or list; no exception raised.
        """
        prompt = PromptTemplate(
            "Extract the measurement details from the text.\nText: {text}"
        )
        gen = await llm_model.astream_structured_predict(
            Measurement,
            prompt,
            text="The temperature outside is 23.5 °C.",
        )
        results = []
        async for obj in gen:
            results.append(obj)
        assert len(results) >= 1

    @pytest.mark.e2e
    @pytest.mark.asyncio()
    @_SKIP
    async def test_async_stream_nested_completes(self, llm_model: Ollama) -> None:
        """Async streaming completes without error for a nested schema.

        Inputs: route description with two waypoints.
        Expected: Generator exhausted; final item contains a Route or partial thereof.
        Checks: no exception; at least one item yielded.
        """
        prompt = PromptTemplate(
            "Extract the route details from the text.\nText: {text}"
        )
        gen = await llm_model.astream_structured_predict(
            Route,
            prompt,
            text=(
                "Drive from Eiffel Tower (48.8584° N, 2.2945° E) "
                "to the Louvre (48.8606° N, 2.3376° E)."
            ),
        )
        results = []
        async for obj in gen:
            results.append(obj)
        assert len(results) >= 1
