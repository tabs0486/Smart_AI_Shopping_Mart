import os
import pandas as pd
import streamlit as st

from workflow.shopping_crew import run_shopping_crew
from connectors.product_sources import load_products
from vector_store.faiss_store import ProductVectorStore


st.set_page_config(
    page_title="AI Smart Shopping Mart",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 AI Smart Shopping Mart")
st.write(
    "Search and compare products using a CrewAI multi-agent shopping system."
)


# --------------------------------------------------
# GROQ API KEY
# --------------------------------------------------

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
    )
    st.stop()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.sidebar.header("Shopping Requirements")

category = st.sidebar.selectbox(
    "Category",
    [
        "Electronics",
        "Grocery",
        "Baby Products",
        "Home & Kitchen",
    ],
)

product = st.sidebar.text_input(
    "Product",
    placeholder="e.g. laptop, baby cereal, rice"
)

budget = st.sidebar.number_input(
    "Maximum Budget (PKR)",
    min_value=0,
    value=100000,
    step=1000,
)

condition = st.sidebar.selectbox(
    "Condition",
    ["New", "Used", "Any"]
)

city = st.sidebar.selectbox(
    "Delivery City",
    [
        "Karachi",
        "Lahore",
        "Islamabad",
        "Rawalpindi",
        "Faisalabad",
        "Multan",
        "Peshawar",
        "Other",
    ],
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Any",
        "Cash on Delivery",
        "Card",
    ],
)

deadline = st.sidebar.number_input(
    "Delivery Deadline (days)",
    min_value=1,
    value=3,
)

ingredients = st.sidebar.text_input(
    "Ingredients",
    placeholder="e.g. milk, wheat, iron"
)

specifications = st.sidebar.text_input(
    "Specifications",
    placeholder="e.g. 16GB RAM, 512GB SSD"
)


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

if st.button("🔎 Search Products", use_container_width=True):

    user_request = f"""
    Category: {category}
    Product: {product}
    Maximum Budget: PKR {budget}
    Condition: {condition}
    Delivery City: {city}
    Payment Method: {payment}
    Delivery Deadline: {deadline} days
    Ingredients: {ingredients}
    Specifications: {specifications}
    """

    with st.spinner("AI shopping agents are working..."):

        products = load_products()

        # Semantic retrieval through FAISS
        vector_store = ProductVectorStore()

        query = (
            f"{category} {product} {ingredients} "
            f"{specifications}"
        )

        retrieved = vector_store.search(
            query,
            k=10
        )

        st.subheader("Retrieved Products")

        st.dataframe(
            retrieved,
            use_container_width=True,
        )

        # CrewAI
        with st.spinner("Running CrewAI agents..."):

            result = run_shopping_crew(
                user_request=user_request,
                products=retrieved.to_dict(
                    orient="records"
                ),
            )

        st.subheader("🤖 AI Shopping Analysis")

        st.write(result.raw)

else:
    st.info(
        "Enter your shopping requirements and click Search Products."
    )
