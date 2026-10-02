import os

# ---------------------------------------------------------
# FIX FOR CREWAI + GROQ cache_breakpoint ERROR
# ---------------------------------------------------------
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg


from crewai import Agent, LLM


MODEL_NAME = "groq/openai/gpt-oss-120b"


def get_llm():
    """
    Create the Groq LLM used by CrewAI.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please add it to Streamlit Secrets."
        )

    return LLM(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=3000,
        reasoning_effort="medium",
    )


def create_agents():

    llm = get_llm()

    requirement_agent = Agent(
        role="Shopping Requirement Analyst",
        goal=(
            "Understand the customer's shopping request and convert it "
            "into structured shopping requirements."
        ),
        backstory=(
            "You specialize in understanding shopping requirements, "
            "budget, delivery city, payment methods, product condition, "
            "ingredients and specifications."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    search_agent = Agent(
        role="Product Search Coordinator",
        goal=(
            "Identify relevant products from the supplied product sources."
        ),
        backstory=(
            "You specialize in searching and identifying products "
            "across multiple shopping sources."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    filter_agent = Agent(
        role="Product Filtering Specialist",
        goal=(
            "Filter products according to the customer's requirements."
        ),
        backstory=(
            "You carefully check price, category, condition, delivery, "
            "payment method, ingredients and specifications."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    comparison_agent = Agent(
        role="Product Comparison Analyst",
        goal=(
            "Compare matching products using factual product information."
        ),
        backstory=(
            "You compare prices, delivery information, payment methods, "
            "ingredients, specifications and sources."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    recommendation_agent = Agent(
        role="Shopping Recommendation Assistant",
        goal=(
            "Present the product comparison clearly to the customer."
        ),
        backstory=(
            "You provide clear shopping information and never invent "
            "prices, delivery dates, ingredients or specifications."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return {
        "requirement": requirement_agent,
        "search": search_agent,
        "filter": filter_agent,
        "comparison": comparison_agent,
        "recommendation": recommendation_agent,
    }
