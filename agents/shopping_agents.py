import os

from crewai import Agent, LLM


MODEL_NAME = "groq/openai/gpt-oss-120b"


def get_llm():
    """
    Create the Groq-powered CrewAI LLM.
    GROQ_API_KEY must be stored in Streamlit Secrets/environment.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured.")

    return LLM(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=3000,
    )


def create_agents():
    llm = get_llm()

    requirement_agent = Agent(
        role="Shopping Requirement Analyst",
        goal=(
            "Understand the customer's shopping request and convert it "
            "into precise structured requirements."
        ),
        backstory=(
            "You specialize in understanding customer shopping needs, "
            "budgets, delivery requirements, specifications and product conditions."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    search_agent = Agent(
        role="Product Search Coordinator",
        goal=(
            "Analyze available product-source data and identify products "
            "that could satisfy the user's requirements."
        ),
        backstory=(
            "You specialize in product discovery across multiple shopping sources. "
            "You must rely only on supplied product-source information."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    filter_agent = Agent(
        role="Product Filtering Specialist",
        goal=(
            "Filter products according to budget, category, condition, "
            "delivery city, payment method, deadline and specifications."
        ),
        backstory=(
            "You are a strict shopping filter. Never claim that a product "
            "matches a requirement unless the supplied data supports it."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    comparison_agent = Agent(
        role="Product Comparison Analyst",
        goal=(
            "Compare matching products using price, delivery, payment, "
            "ingredients, specifications and source information."
        ),
        backstory=(
            "You are a product comparison analyst who creates clear "
            "side-by-side comparisons."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    recommendation_agent = Agent(
        role="Shopping Recommendation Assistant",
        goal=(
            "Present the matching products clearly and explain the important "
            "differences without inventing unavailable information."
        ),
        backstory=(
            "You are a helpful shopping assistant. You present factual "
            "product information and clearly identify missing information."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return {
        "requirement": requirement_agent,
        "search": search_agent,
        "filter": filter_agent,
        "comparison": comparison_agent,
        "recommendation": recommendation_agent,
    }
